"""Uses LangGraph's functional API to build an agent."""

from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
import base64
import io
import json
import logging
import uuid
from typing import Any, TypedDict, TypeVar
import os
from dotenv import load_dotenv
load_dotenv()
import langsmith as ls
import PIL
from arcengine import FrameData, GameAction
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAI
from langchain_anthropic import ChatAnthropic
from openai.types.chat import ChatCompletionMessage

from agents.templates.llm_agents import LLM

from ..agent import Agent

logger = logging.getLogger(__name__)

class State(TypedDict, total=False):
    frames: list[FrameData]
    latest_frame: FrameData
    

MESSAGES = TypeVar("MESSAGES", bound=list[dict[str, Any] | ChatCompletionMessage])

###TODO: better system prompt
SYS_PROMPT = """# CONTEXT:
You are an agent playing a dynamic ARC-AGI game. Your objective is to
WIN and avoid GAME_OVER while minimizing actions.

# TURN:
Call exactly one action.
"""


class ResearchBaseline(LLM, Agent):
    """An agent that always selects actions at random."""

    MAX_ACTIONS = 80
    MODEL_REQUIRES_TOOLS = True

    MODEL: str = "gpt-5.2"
    PROVIDER: str = "openai" # "openai" | "anthropic" | "openrouter"
    REASONING_EFFORT: str | None = "low"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._thread_id = uuid.uuid5(uuid.NAMESPACE_DNS, self.game_id)
        self.available_actions = self.arc_env.action_space
        self.current_games_actions = [action.name for action in self.arc_env.action_space]
        tools = self.build_tools()
        self.llm = self._build_llm(tools)
        self.previous_tool_id = None # NOTE, we're doing this because we haven't have any parallel tool calls.
        ###TODO: better context management to be done
        self.thread_messages = [SystemMessage(SYS_PROMPT)]

    def _build_llm(self, tools: list[dict[str, Any]]):
        if self.PROVIDER == "openai":
            return ChatOpenAI(model=self.MODEL
            , api_key=os.getenv("OPENAI_API_KEY")
            , tool_choice="required"
            , model_kwargs={"reasoning": {"effort": self.REASONING_EFFORT, "summary": "auto"}}).bind_tools(tools)
        elif self.PROVIDER == "anthropic":
            ###TODO: Anthropic's reasoning has different kwargs
            return ChatAnthropic(model=self.MODEL
            , api_key=os.getenv("ANTHROPIC_API_KEY")
            , model_kwargs={"reasoning": {"effort": self.REASONING_EFFORT, "summary": "auto"}}).bind_tools(tools)
        elif self.PROVIDER == "openrouter":
            ###TODO: OpenRouter's reasoning is different for different providers
            return ChatOpenAI(model=self.MODEL
            , base_url="https://openrouter.ai/api/v1"
            , api_key=os.getenv("OPENROUTER_API_KEY")
            , model_kwargs={"reasoning": {"effort": self.REASONING_EFFORT, "summary": "auto"}}).bind_tools(tools)
        else:
            raise ValueError(f"Unknown provider: {self.PROVIDER}")
    
    @ls.traceable  # type: ignore[misc]
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        ###TODO: no game_over and reset handling. 
        prev_tool_result, frame_and_next_action = format_frame(latest_frame, as_image=True)
        if len(self.thread_messages) >= 2:
            # print(self.thread_messages)
            self.thread_messages.append(ToolMessage(content=prev_tool_result, tool_call_id=self.previous_tool_id))
        messages = self.thread_messages + [HumanMessage(content=frame_and_next_action)]
        msg = self.llm.invoke(messages)
        tool_calls = msg.tool_calls
        reasoning = msg.additional_kwargs.get("reasoning", {}).get("summary", [])
        try:
            self.thread_messages.append(AIMessage(content=f"(Reasoning summary)\n{reasoning}"))
            self.thread_messages.append(msg)
            print("this is msg tokens", msg.usage_metadata)
            func = tool_calls[0]
            self.previous_tool_id = func["id"]
            action = GameAction.from_name(func["name"])
            args = json.loads(func["args"]) if func["args"] else {}
        except Exception as e:
            logger.exception(f"Tool calling still failed: {e}")
            raise e
        action.set_data(args)
        action.reasoning = msg.model_dump()
        self.track_tokens(msg)
        return action

    def main(self) -> None:
        with ls.trace(
            "LangGraph Agent",
            input={"state": self.state},
            metadata={
                "game_id": self.game_id,
                "card_id": self.card_id,
                "agent_name": self.agent_name,
                "thread_id": self._thread_id,
            },
        ) as rt:
            super().main()
            rt.end(outputs={"state": self.state})

    def track_tokens(self, msg: AIMessage) -> None:
        ###TODO: implement different loggings
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record({
                "reasoning": msg.additional_kwargs.get("reasoning", {}),
                "tool_calls": msg.tool_calls[0],
                # "action_counter": self.action_counter,
            })

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        ###TODO: implement
        return """No func resp for now"""

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        ###TODO: implement
        return """No user prompt for now"""
    
    def build_tools(self) -> list[dict[str, Any]]:
        tools = super().build_tools()
        tools = [tool for tool in tools if tool["function"]["name"] in self.current_games_actions]
        return tools

# uv run main.py --agent=researchbaseline --game=ls20
def format_frame(latest_frame: FrameData, as_image: bool) -> list[dict[str, Any]]: 
    img = g2im(latest_frame.frame) if latest_frame.frame else None
    frame_block = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img).decode('ascii')}"},
    }
    prev_tool_result = [{
            "type": "text",
            "text": f"""# State:
        {latest_frame.state.name}

        # Score:
        {latest_frame.levels_completed}

        """}]
    frame_and_next_action = [
        frame_block,
        {
            "type": "text",
            "text": """
# TURN:
Reply with a few sentences of plain-text strategy observation about the frame to inform your next action.""",
        },
    ]
    return prev_tool_result, frame_and_next_action


def g2im(g: list[list[list[int]]]) -> bytes:
    from arc_agi.rendering import COLOR_MAP, hex_to_rgb
    C = [hex_to_rgb(COLOR_MAP[i]) for i in range(16)]

    h, w = len(g[0]), len(g[0][0])
    good = [block for block in g if len(block) == h and len(block[0]) == w]
    n = len(good)
    s = 5 * (n > 1)
    W = w * n + s * (n - 1)

    im = PIL.Image.new("RGB", (W, h), "white")
    px = im.load()
    for i, block in enumerate(good):
        ox = i * (w + s)
        for y, row in enumerate(block):
            for x, val in enumerate(row):
                px[ox + x, y] = C[val & 15]

    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()

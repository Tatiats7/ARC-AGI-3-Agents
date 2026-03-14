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
from arcengine.enums import GameState
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

##TODO: better system prompt
SYS_PROMPT = """# CONTEXT
You are an agent solving ARC-AGI puzzles.

Your objective is to reach the **WIN state** on each level while avoiding **GAME_OVER**.
Moves are limited, so actions must be deliberate and efficient.

You interact with the environment through **actions (tool calls)**.
Each step may contain **exactly one tool call**.

# OBSERVATIONS

You receive **frames** as input, all previous actions, plans, reflections, hypotheses, etc. as history.

Important:

* Latest Frame includes the latest state.
* Earlier assumptions you made may be incorrect when compared to latest frame/frames.
* Always re-examine the situation to verify whether your hypotheses still hold.

Do not assume your first interpretation is correct, it might be but it is not guaranteed.
Be ready to **revise your understanding of the environment** when new evidence or thought appears.

# REASONING STRATEGY

Approach the problem using an iterative loop:

1. **Understand the environment**
   * Identify the player, goal location, obstacles, and interactive objects.
   * Determine how orientation, rotations, or interactions affect the puzzle.

2. **Form a hypothesis**
   * Infer what the puzzle rules are.
   * Predict what actions should move the system closer to the goal.

3. **Plan**
   * Decide the next best step toward the objective.
   * Consider move efficiency since moves are limited.

4. **Act**
   * Execute exactly **one action** via a tool call.

5. **Reflect**
   * Compare the result of the action with your expectations.
   * If the outcome contradicts your hypothesis, revise your understanding.

Reflection is important.
Your earlier reasoning may be wrong, incomplete, or based on misinterpreted visual cues.

# OUTPUT RULES

Each response represents one step of the agent.

Your response may contain **visible textual guidance** such as:

• **Hypothesis (optional)**  
A short description of your current understanding of the puzzle.

• **Reflection (optional)**  
Brief notes about what happened after previous actions or whether your hypothesis might be wrong.

• **Plan (optional)**  
A short description of the next move or short sequence of moves you intend to try.

Important:
Internal reasoning tokens are temporary and may not persist across steps.  
If a hypothesis, discovery, or plan is important for future steps, write it in the assistant message so it becomes part of the history.

Each step may also include:
• **Exactly one tool call representing the next action**

Rules:
- At most one tool call per step
- Text guidance should be concise
- Prefer acting rather than producing long explanations

# IMPORTANT RULE

Reason briefly.  
Do not produce long chains of reasoning.
Your primary job is to take actions that move the player toward WIN.

Each response represents one step of the agent.

Most steps should end with exactly one action (tool call).
Text reasoning should be short and only used when necessary.

# PRINCIPLES

* Be **adaptive**: update your understanding as new frames appear.
* Be **efficient**: minimize unnecessary moves.
* Be **critical of your own reasoning**: early conclusions may be wrong.
* Be **curious**: explore the environment to understand it better. Identify objects, their properties, their interactions, and the rules of the puzzle.
* Always think about **how the current step contributes to reaching the WIN state**.
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
        ###TODO: add planning/thinking/analyzing steps
        tools = self.build_tools()
        self.llm = self._build_llm(tools)
        self.previous_tool_id = None # NOTE, we're doing this because we haven't have any parallel tool calls.
        self.previous_levels_completed = 0
        ###TODO: better context management to be done
        self.thread_messages = [SystemMessage(SYS_PROMPT)]

    def _build_llm(self, tools: list[dict[str, Any]]):
        if self.PROVIDER == "openai":
            return ChatOpenAI(model=self.MODEL
            , api_key=os.getenv("OPENAI_API_KEY")
            # , tool_choice="required"
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
        prev_tool_result, current_frame = format_frame(latest_frame,  self.previous_levels_completed, as_image=True)
        self.previous_levels_completed = latest_frame.levels_completed
        if len(self.thread_messages) >= 2:
            self.thread_messages.append(ToolMessage(content=prev_tool_result, tool_call_id=self.previous_tool_id))
        self.thread_messages.append(HumanMessage(content=current_frame))

        while True:
            msg = self.llm.invoke(self.thread_messages)
            tool_calls = msg.tool_calls
            if tool_calls:
                break
            self.thread_messages.append(msg)
            print("this is msg tokens", msg.usage_metadata)
            self.track_tokens(msg)
            print("no action calls, running again")
        try:
            self.thread_messages.append(msg)
            print("this is msg tokens", msg.usage_metadata)
            func = tool_calls[0]
            self.previous_tool_id = func["id"]
            action = GameAction.from_name(func["name"])
            args = func["args"] if func["args"] else {}
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
                "tool_calls": msg.tool_calls[0] if msg.tool_calls else {},
                "model_name": self.MODEL,
                "text_msg_if_any": self.thread_messages[-1].content if self.thread_messages[-1].content else "",
                # "action_counter": self.action_counter,
            })

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        """This function is just here to prevent incorrect logging"""
        return """No func resp for now"""

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        """This function is just here to prevent incorrect logging"""
        return """No user prompt for now"""
    
    def build_tools(self) -> list[dict[str, Any]]:
        tools = super().build_tools()
        tools = [tool for tool in tools if tool["function"]["name"] in self.current_games_actions]
        tools.append(test_tool)
        return tools

# uv run main.py --agent=researchbaseline --game=ls20
def format_frame(latest_frame: FrameData, previous_levels_completed: int, as_image: bool) -> list[dict[str, Any]]: 
    img_ls = g2im(latest_frame.frame) if latest_frame.frame else None

    ###TODO: add one level completion celebration.
    # if previous_levels_completed < latest_frame.levels_completed:
    #     text_to_append = "Congrats! you just won one level. do continue on the next"
    if img_ls is None:
        raise ValueError("No image for the frame")
    if len(img_ls) == 1:
        frame_block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img_ls[0]).decode('ascii')}"},
        }
    else:
        frame_block = [{
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img).decode('ascii')}"},
        } for img in img_ls]
    
    prev_tool_result = [{
            "type": "text",
            "text": f"""# State:
        {latest_frame.state.name}

        # Score:
        {latest_frame.levels_completed}

        """}]
    if len(img_ls) == 2:
        current_frame = [
            {"type": "text", "text": """Current frame:"""},
            frame_block[0],
            frame_block[1]
        ]
    else:
        current_frame = [
            {"type": "text", "text": """Current frame:"""},
            frame_block
        ]
    return prev_tool_result, current_frame


def g2im(g: list[list[list[int]]]) -> list[bytes]:
    from arc_agi.rendering import COLOR_MAP, hex_to_rgb
    C = [hex_to_rgb(COLOR_MAP[i]) for i in range(16)]

    CELL = 8
    GRID_COLOR = (58, 58, 60)

    h, w = len(g[0]), len(g[0][0])
    good = [block for block in g if len(block) == h and len(block[0]) == w]

    img_w = w * CELL + (w - 1)
    img_h = h * CELL + (h - 1)

    images = []
    for block in good:
        tile = PIL.Image.new("RGB", (img_w, img_h), GRID_COLOR)
        px = tile.load()
        for y, row in enumerate(block):
            for x, val in enumerate(row):
                color = C[val & 15]
                px0 = x * (CELL + 1)
                py0 = y * (CELL + 1)
                for dy in range(CELL):
                    for dx in range(CELL):
                        px[px0 + dx, py0 + dy] = color
        buf = io.BytesIO()
        tile.save(buf, "PNG")
        images.append(buf.getvalue())

    return images

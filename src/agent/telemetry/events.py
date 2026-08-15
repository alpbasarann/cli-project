from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from agent.protocol import StopReason


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BaseEvent(BaseModel):
    timestamp: datetime = Field(default_factory=_utcnow)
    session_id: str = ""


class TurnStarted(BaseEvent):
    type: Literal["turn_started"] = "turn_started"


class LLMCallCompleted(BaseEvent):
    type: Literal["llm_call_completed"] = "llm_call_completed"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    stop_reason: StopReason = "end_turn"


class ToolExecuted(BaseEvent):
    type: Literal["tool_executed"] = "tool_executed"
    tool_name: str
    is_error: bool = False
    duration_ms: float = 0.0


class ContextCompacted(BaseEvent):
    type: Literal["context_compacted"] = "context_compacted"
    tokens_before: int
    tokens_after: int


class TurnCompleted(BaseEvent):
    type: Literal["turn_completed"] = "turn_completed"
    steps: int
    total_tokens: int
    estimated_cost_usd: float | None = None


Event = Annotated[
    Union[TurnStarted, LLMCallCompleted, ToolExecuted, ContextCompacted, TurnCompleted],
    Field(discriminator="type"),
]

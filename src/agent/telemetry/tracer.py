from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter

from agent.telemetry.events import (
    ContextCompacted,
    Event,
    LLMCallCompleted,
    ToolExecuted,
    TurnCompleted,
    TurnStarted,
)

MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

FALLBACK_PRICE = (3.0, 15.0)

_EVENT_ADAPTER = TypeAdapter(Event)


def price_for(model: str) -> tuple[float, float]:
    price = MODEL_PRICES.get(model)
    if price is not None:
        return price
    for name, known in MODEL_PRICES.items():
        if model.startswith(name):
            return known
    return FALLBACK_PRICE


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = price_for(model)
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


@dataclass
class TraceTotals:
    turns: int = 0
    steps: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    llm_duration_ms: float = 0.0
    tool_duration_ms: float = 0.0
    tool_calls: dict[str, int] = field(default_factory=dict)
    tool_errors: int = 0
    compactions: int = 0
    tokens_reclaimed: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tool_call_count(self) -> int:
        return sum(self.tool_calls.values())

    def record(self, event: Event) -> None:
        if isinstance(event, TurnStarted):
            self.turns += 1
        elif isinstance(event, LLMCallCompleted):
            self.llm_calls += 1
            self.input_tokens += event.input_tokens
            self.output_tokens += event.output_tokens
            self.llm_duration_ms += event.duration_ms
            self.estimated_cost_usd += estimate_cost(
                event.model, event.input_tokens, event.output_tokens
            )
        elif isinstance(event, ToolExecuted):
            self.tool_calls[event.tool_name] = self.tool_calls.get(event.tool_name, 0) + 1
            self.tool_duration_ms += event.duration_ms
            if event.is_error:
                self.tool_errors += 1
        elif isinstance(event, ContextCompacted):
            self.compactions += 1
            self.tokens_reclaimed += event.tokens_before - event.tokens_after
        elif isinstance(event, TurnCompleted):
            self.steps += event.steps


class Tracer(Protocol):
    def emit(self, event: Event) -> None: ...


class NullTracer:
    def emit(self, event: Event) -> None:
        return None


class JsonlTracer:
    def __init__(self, trace_dir: Path, session_id: str, model: str = "") -> None:
        self._path = Path(trace_dir) / f"{session_id}.jsonl"
        self._session_id = session_id
        self._model = model
        self._totals = TraceTotals()
        self._turn_cost = 0.0
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def totals(self) -> TraceTotals:
        return self._totals

    def emit(self, event: Event) -> None:
        if isinstance(event, TurnStarted):
            self._turn_cost = 0.0

        event = self._enrich(event)

        if isinstance(event, LLMCallCompleted):
            self._turn_cost += estimate_cost(
                event.model, event.input_tokens, event.output_tokens
            )

        self._totals.record(event)

        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def _enrich(self, event: Event) -> Event:
        updates: dict = {}

        if not event.session_id:
            updates["session_id"] = self._session_id
        if isinstance(event, LLMCallCompleted) and not event.model:
            updates["model"] = self._model
        if isinstance(event, TurnCompleted) and event.estimated_cost_usd is None:
            updates["estimated_cost_usd"] = round(self._turn_cost, 6)

        return event.model_copy(update=updates) if updates else event


def read_events(path: Path) -> list[Event]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [_EVENT_ADAPTER.validate_json(line) for line in lines if line.strip()]


def summarize(events: list[Event]) -> TraceTotals:
    totals = TraceTotals()
    for event in events:
        totals.record(event)
    return totals

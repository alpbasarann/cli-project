import time
from dataclasses import dataclass, field

from agent.errors import MaxStepsExceeded
from agent.llm.base import LLMProvider
from agent.protocol import Message, Usage
from agent.session.context import ContextManager
from agent.session.state import Session
from agent.telemetry.events import (
    ContextCompacted,
    LLMCallCompleted,
    TurnCompleted,
    TurnStarted,
)
from agent.telemetry.tracer import NullTracer, Tracer
from agent.tools.dispatcher import Dispatcher


@dataclass
class RunResult:
    output: str
    steps: int
    usage: Usage = field(default_factory=Usage)
    history: list[Message] = field(default_factory=list)


def run(
    provider: LLMProvider,
    dispatcher: Dispatcher,
    session: Session,
    user_input: str,
    system: str | None = None,
    max_steps: int = 20,
    context: ContextManager | None = None,
    tracer: Tracer | None = None,
) -> RunResult:
    context = context or ContextManager()
    tracer = tracer or NullTracer()
    session.start_turn(user_input)
    schemas = dispatcher.schemas()

    turn_usage = Usage()
    tracer.emit(TurnStarted(session_id=session.id))

    for step in range(1, max_steps + 1):
        started = time.perf_counter()
        response = provider.send(session.history, schemas, system)
        duration_ms = (time.perf_counter() - started) * 1000

        session.record_response(response)

        turn_usage.input_tokens += response.usage.input_tokens
        turn_usage.output_tokens += response.usage.output_tokens

        tracer.emit(
            LLMCallCompleted(
                session_id=session.id,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                duration_ms=duration_ms,
                stop_reason=response.stop_reason,
            )
        )

        if not response.wants_tools:
            tracer.emit(
                TurnCompleted(
                    session_id=session.id,
                    steps=step,
                    total_tokens=turn_usage.total,
                )
            )
            return RunResult(response.text, step, turn_usage, session.history)

        results = [dispatcher.dispatch(block) for block in response.tool_uses]
        session.record_tool_results(results)

        tokens_before = context.estimate(session.history)
        compacted = context.compact(session.history)
        session.replace_history(compacted)
        tokens_after = context.estimate(compacted)

        if tokens_after != tokens_before:
            tracer.emit(
                ContextCompacted(
                    session_id=session.id,
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                )
            )

    raise MaxStepsExceeded(f"Did not finish within {max_steps} steps")

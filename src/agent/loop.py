from dataclasses import dataclass, field

from agent.errors import MaxStepsExceeded
from agent.llm.base import LLMProvider
from agent.protocol import Message, Usage
from agent.session.context import ContextManager
from agent.session.state import Session
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
) -> RunResult:
    context = context or ContextManager()
    session.start_turn(user_input)
    schemas = dispatcher.schemas()

    turn_usage = Usage()

    for step in range(1, max_steps + 1):
        response = provider.send(session.history, schemas, system)
        session.record_response(response)

        turn_usage.input_tokens += response.usage.input_tokens
        turn_usage.output_tokens += response.usage.output_tokens

        if not response.wants_tools:
            return RunResult(response.text, step, turn_usage, session.history)

        results = [dispatcher.dispatch(block) for block in response.tool_uses]
        session.record_tool_results(results)

        session.replace_history(context.compact(session.history))

    raise MaxStepsExceeded(f"Did not finish within {max_steps} steps")
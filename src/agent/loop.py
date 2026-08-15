from dataclasses import dataclass, field

from agent.errors import MaxStepsExceeded
from agent.llm.base import LLMProvider
from agent.protocol import Message, Usage, UserMessage
from agent.tools import ToolRegistry, dispatch


@dataclass
class RunResult:
    output: str
    history: list[Message]
    steps: int
    usage: Usage = field(default_factory=Usage)


def run(
    provider: LLMProvider,
    registry: ToolRegistry,
    user_input: str,
    system: str | None = None,
    max_steps: int = 10,
) -> RunResult:
    history: list[Message] = [UserMessage.from_text(user_input)]
    schemas = registry.schemas()
    total = Usage()

    for step in range(1, max_steps + 1):
        response = provider.send(history, schemas, system)
        history.append(response)

        total.input_tokens += response.usage.input_tokens
        total.output_tokens += response.usage.output_tokens

        if not response.wants_tools:
            return RunResult(response.text, history, step, total)

        results = [dispatch(block, registry) for block in response.tool_uses]
        history.append(UserMessage.from_tool_results(results))

    raise MaxStepsExceeded(f"Did not finish within {max_steps} steps")
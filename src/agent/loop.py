from dataclasses import dataclass

from agent.llm.base import LLMProvider
from agent.protocol import Message, UserMessage
from agent.tools import ToolRegistry, dispatch


class MaxStepsExceeded(Exception):
    pass


@dataclass
class RunResult:
    output: str
    history: list[Message]
    steps: int


def run(
    provider: LLMProvider,
    registry: ToolRegistry,
    user_input: str,
    max_steps: int = 10,
) -> RunResult:
    history: list[Message] = [UserMessage.from_text(user_input)]
    schemas = registry.schemas()

    for step in range(1, max_steps + 1):
        response = provider.send(history, schemas)
        history.append(response)

        if not response.wants_tools:
            return RunResult(output=response.text, history=history, steps=step)

        results = [dispatch(block, registry) for block in response.tool_uses]
        history.append(UserMessage.from_tool_results(results))

    raise MaxStepsExceeded(f"Did not finish within {max_steps} steps")
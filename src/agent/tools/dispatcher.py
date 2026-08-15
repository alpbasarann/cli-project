from typing import Callable

from pydantic import ValidationError

from agent.errors import SandboxViolation
from agent.protocol import ToolResultBlock, ToolUseBlock
from agent.tools.base import ToolContext
from agent.tools.registry import ToolRegistry

ApprovalFn = Callable[[str, dict], bool]


def always_approve(tool_name: str, args: dict) -> bool:
    return True


class Dispatcher:
    def __init__(
        self,
        registry: ToolRegistry,
        context: ToolContext,
        approve: ApprovalFn = always_approve,
    ) -> None:
        self._registry = registry
        self._context = context
        self._approve = approve

    def schemas(self) -> list[dict]:
        return self._registry.schemas()

    def dispatch(self, block: ToolUseBlock) -> ToolResultBlock:
        def failure(message: str) -> ToolResultBlock:
            return ToolResultBlock(tool_use_id=block.id, content=message, is_error=True)

        tool = self._registry.get(block.name)
        if tool is None:
            return failure(f"Unknown tool: '{block.name}'")

        try:
            args = tool.Input(**block.input)
        except ValidationError as exc:
            return failure(f"Invalid arguments for '{block.name}':\n{exc}")

        if tool.mutating and not self._approve(tool.name, block.input):
            return failure(f"The user declined to run '{tool.name}'.")

        try:
            result = tool.execute(args, self._context)
        except SandboxViolation as exc:
            return failure(str(exc))
        except Exception as exc:
            return failure(f"{type(exc).__name__}: {exc}")

        return ToolResultBlock(
            tool_use_id=block.id,
            content=result.content,
            is_error=result.is_error,
        )
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ValidationError

from agent.protocol import ToolResultBlock, ToolUseBlock


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


class Tool(Protocol):
    name: str
    description: str
    Input: type[BaseModel]

    def execute(self, args: BaseModel) -> ToolResult: ...


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    def schemas(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.Input.model_json_schema(),
            }
            for t in self._tools.values()
        ]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)


def dispatch(block: ToolUseBlock, registry: ToolRegistry) -> ToolResultBlock:
    def failure(message: str) -> ToolResultBlock:
        return ToolResultBlock(tool_use_id=block.id, content=message, is_error=True)

    tool = registry.get(block.name)
    if tool is None:
        return failure(f"Unknown tool: '{block.name}'")

    try:
        args = tool.Input(**block.input)
    except ValidationError as exc:
        return failure(f"Invalid arguments for '{block.name}':\n{exc}")

    try:
        result = tool.execute(args)
    except Exception as exc:
        return failure(f"{type(exc).__name__}: {exc}")

    return ToolResultBlock(
        tool_use_id=block.id,
        content=result.content,
        is_error=result.is_error,
    )
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from agent.sandbox import Sandbox


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class ToolContext:
    sandbox: Sandbox

    def resolve(self, raw: str) -> Path:
        return self.sandbox.resolve(raw)


class Tool(Protocol):
    name: str
    description: str
    Input: type[BaseModel]
    mutating: bool

    def execute(self, args: BaseModel, ctx: ToolContext) -> ToolResult: ...
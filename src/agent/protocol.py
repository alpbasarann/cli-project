from typing import Self, Annotated, Literal, Union #Annotated = adds metadata, Literal = allows only defined values, Union = merges data types (union type) 
from pydantic import BaseModel, Field

class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    Union[TextBlock, ToolUseBlock, ToolResultBlock],
    Field(discriminator="type"),
]

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

StopReason = Literal["end_turn", "tool_use", "max_tokens"]

class BaseMessage(BaseModel):
    role: str
    content: list[ContentBlock]

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock))


class UserMessage(BaseMessage):
    role: Literal["user"] = "user"

    @classmethod
    def from_text(cls, text: str) -> Self:
        return cls(content=[TextBlock(text=text)])

    @classmethod
    def from_tool_results(cls, results: list[ToolResultBlock]) -> Self:
        return cls(content=list(results))

    @property
    def tool_results(self) -> list[ToolResultBlock]:
        return [b for b in self.content if isinstance(b, ToolResultBlock)]


class AssistantMessage(BaseMessage):
    role: Literal["assistant"] = "assistant"
    stop_reason: StopReason = "end_turn"
    usage: Usage = Field(default_factory=Usage)

    @classmethod
    def from_text(cls, text: str, stop_reason: StopReason = "end_turn") -> Self:
        return cls(content=[TextBlock(text=text)], stop_reason=stop_reason)

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def wants_tools(self) -> bool:
        return self.stop_reason == "tool_use"


Message = Annotated[
    Union[UserMessage, AssistantMessage],
    Field(discriminator="role"),
]
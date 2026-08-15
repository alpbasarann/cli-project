from pydantic import BaseModel, Field

from agent.llm.fake import FakeLLMProvider
from agent.loop import run
from agent.protocol import AssistantMessage, ToolUseBlock
from agent.tools import ToolRegistry, ToolResult

FAKE_FS = {"main.py": "def add(a, b):\n    return a - b\n"}


class ReadFileInput(BaseModel):
    path: str = Field(description="Path of the file to read")


class ReadFileTool:
    name = "read_file"
    description = "Returns the contents of the file at the given path."
    Input = ReadFileInput

    def execute(self, args: ReadFileInput) -> ToolResult:
        if args.path not in FAKE_FS:
            return ToolResult(content=f"File not found: {args.path}", is_error=True)
        return ToolResult(content=FAKE_FS[args.path])


def tool_call(id_: str, name: str, **kwargs) -> AssistantMessage:
    return AssistantMessage(
        content=[ToolUseBlock(id=id_, name=name, input=kwargs)],
        stop_reason="tool_use",
    )


def main() -> None:
    registry = ToolRegistry([ReadFileTool()])

    provider = FakeLLMProvider([
        tool_call("t1", "read_file", path="missing.py"),
        tool_call("t2", "read_file", path="main.py"),
        AssistantMessage.from_text("The add function subtracts instead of adding."),
    ])

    result = run(provider, registry, "find the bug in main.py")

    print(f"Output : {result.output}")
    print(f"Steps  : {result.steps}")
    print(f"Calls  : {len(provider.calls)}\n")

    for msg in result.history:
        for block in msg.content:
            detail = getattr(block, "text", None) or getattr(block, "content", None) or block.input
            flag = " [ERROR]" if getattr(block, "is_error", False) else ""
            print(f"[{msg.role:9}] {block.type:12}{flag} {str(detail)[:70]}")


if __name__ == "__main__":
    main()
from pydantic import BaseModel, Field

from agent.tools.base import ToolContext, ToolResult


class WriteFileInput(BaseModel):
    path: str = Field(description="File path relative to the working directory")
    content: str = Field(description="Full contents to write to the file")


class WriteFileTool:
    name = "write_file"
    description = (
        "Create a new file or overwrite an existing one. "
        "To change part of an existing file, use edit_file instead."
    )
    Input = WriteFileInput
    mutating = True

    def execute(self, args: WriteFileInput, ctx: ToolContext) -> ToolResult:
        target = ctx.resolve(args.path)
        existed = target.is_file()

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")

        verb = "Overwrote" if existed else "Created"
        return ToolResult(content=f"{verb} {args.path} ({len(args.content.splitlines())} lines)")
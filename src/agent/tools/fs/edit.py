from pydantic import BaseModel, Field

from agent.tools.base import ToolContext, ToolResult


class EditFileInput(BaseModel):
    path: str = Field(description="File path relative to the working directory")
    old_string: str = Field(description="Exact text to replace; must appear exactly once")
    new_string: str = Field(description="Replacement text")


class EditFileTool:
    name = "edit_file"
    description = (
        "Replace an exact string in a file. old_string must match the file content "
        "exactly and occur exactly once, so include enough surrounding context to "
        "make it unique. Read the file before editing it."
    )
    Input = EditFileInput
    mutating = True

    def execute(self, args: EditFileInput, ctx: ToolContext) -> ToolResult:
        target = ctx.resolve(args.path)

        if not target.is_file():
            return ToolResult(content=f"File not found: {args.path}", is_error=True)

        original = target.read_text(encoding="utf-8")
        count = original.count(args.old_string)

        if count == 0:
            return ToolResult(
                content=f"old_string not found in {args.path}. Read the file and copy the exact text.",
                is_error=True,
            )
        if count > 1:
            return ToolResult(
                content=f"old_string appears {count} times in {args.path}. Add more surrounding context to make it unique.",
                is_error=True,
            )

        target.write_text(original.replace(args.old_string, args.new_string), encoding="utf-8")
        return ToolResult(content=f"Edited {args.path}")
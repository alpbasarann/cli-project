from pydantic import BaseModel, Field

from agent.tools.base import ToolContext, ToolResult


class ReadFileInput(BaseModel):
    path: str = Field(description="File path relative to the working directory")
    offset: int = Field(default=0, ge=0, description="First line to read (0-based)")
    limit: int = Field(default=2000, gt=0, description="Maximum number of lines to read")


class ReadFileTool:
    name = "read_file"
    description = (
        "Read a text file and return its contents with line numbers. "
        "Always read a file before editing it. "
        "For large files, use offset and limit to read in chunks."
    )
    Input = ReadFileInput
    mutating = False

    def execute(self, args: ReadFileInput, ctx: ToolContext) -> ToolResult:
        target = ctx.resolve(args.path)

        if not target.is_file():
            return ToolResult(content=f"File not found: {args.path}", is_error=True)

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        window = lines[args.offset : args.offset + args.limit]

        if not window:
            return ToolResult(content=f"No content at offset {args.offset}.")

        numbered = "\n".join(
            f"{args.offset + i + 1:6}\t{line}" for i, line in enumerate(window)
        )

        end = args.offset + len(window)
        if end < len(lines):
            numbered += f"\n\n[showing lines {args.offset + 1}-{end} of {len(lines)}]"

        return ToolResult(content=numbered)
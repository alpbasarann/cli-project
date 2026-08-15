from pathlib import Path

from pydantic import BaseModel, Field

from agent.llm.config import ProviderConfig
from agent.llm.registry import create_provider
from agent.loop import run
from agent.tools import ToolRegistry, ToolResult
from agent.errors import ProviderError
from dotenv import load_dotenv

load_dotenv()

SYSTEM = "You are a CLI coding agent. Use the available tools to inspect files before answering."


class ReadFileInput(BaseModel):
    path: str = Field(description="Path of the file to read, relative to the working directory")


class ReadFileTool:
    name = "read_file"
    description = "Read the contents of a text file. Use this before answering questions about code."
    Input = ReadFileInput

    def execute(self, args: ReadFileInput) -> ToolResult:
        target = Path(args.path)
        if not target.is_file():
            return ToolResult(content=f"File not found: {args.path}", is_error=True)
        return ToolResult(content=target.read_text(encoding="utf-8"))


def main() -> None:
    try:
        provider = create_provider(ProviderConfig())
        registry = ToolRegistry([ReadFileTool()])

        result = run(
            provider,
            registry,
            "Read demo.py and explain in two sentences what it does.",
            system=SYSTEM,
        )

        print(result.output)
        print(f"\nsteps={result.steps} tokens={result.usage.total}")
    except ProviderError as exc:
        print(f"Provider error: {exc}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
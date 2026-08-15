from dotenv import load_dotenv

load_dotenv()

import json
from pathlib import Path

from agent.errors import AgentError
from agent.llm.config import ProviderConfig
from agent.llm.registry import create_provider
from agent.loop import run
from agent.sandbox import Sandbox
from agent.tools.base import ToolContext
from agent.tools.dispatcher import Dispatcher
from agent.tools.fs.edit import EditFileTool
from agent.tools.fs.read import ReadFileTool
from agent.tools.fs.write import WriteFileTool
from agent.tools.registry import ToolRegistry

SYSTEM = (
    "You are a CLI coding agent working inside a sandboxed directory. "
    "Read files before editing them, and make the smallest change that solves the task."
)


def console_approve(tool_name: str, args: dict) -> bool:
    print(f"\n{tool_name} -> {json.dumps(args)[:300]}")
    return input("approve? [y/N] ").strip().lower() == "y"


def main() -> None:
    workdir = Path("./sandbox").resolve()
    workdir.mkdir(exist_ok=True)

    dispatcher = Dispatcher(
        registry=ToolRegistry([ReadFileTool(), WriteFileTool(), EditFileTool()]),
        context=ToolContext(sandbox=Sandbox(workdir)),
        approve=console_approve,
    )

    try:
        result = run(
            provider=create_provider(ProviderConfig()),
            dispatcher=dispatcher,
            user_input=(
                "Create calc.py containing an add function that mistakenly subtracts, "
                "then read it back and fix the bug with edit_file."
            ),
            system=SYSTEM,
        )
    except AgentError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    print(f"\n{result.output}")
    print(f"steps={result.steps} tokens={result.usage.total}")


if __name__ == "__main__":
    main()
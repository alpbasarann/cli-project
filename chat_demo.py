from dotenv import load_dotenv

load_dotenv()

import json
from pathlib import Path

from agent.llm.config import ProviderConfig
from agent.errors import AgentError
from agent.llm.registry import create_provider
from agent.loop import run
from agent.sandbox import Sandbox
from agent.session.context import ContextManager
from agent.session.state import Session
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


def approve(name: str, args: dict) -> bool:
    print(f"\n{name} -> {json.dumps(args, ensure_ascii=False)[:300]}")
    return input("approve? [y/N] ").strip().lower() == "y"


def main() -> None:
    workdir = Path("./workspace").resolve()
    workdir.mkdir(exist_ok=True)

    provider = create_provider(ProviderConfig())
    dispatcher = Dispatcher(
        registry=ToolRegistry([ReadFileTool(), WriteFileTool(), EditFileTool()]),
        context=ToolContext(sandbox=Sandbox(workdir)),
        approve=approve,
    )
    session = Session()
    context = ContextManager(max_tokens=40_000, keep_recent=6)

    print(f"session {session.id} — type 'exit' to quit, 'reset' to clear history\n")

    while True:
        prompt = input("> ").strip()
        if prompt in {"exit", "quit"}:
            break
        if prompt == "reset":
            session.reset()
            print("history cleared")
            continue
        if not prompt:
            continue

        try:
            result = run(provider, dispatcher, session, prompt, SYSTEM, context=context)
        except AgentError as exc:
            print(f"error: {exc}")
            continue

        print(f"\n{result.output}")
        print(
            f"[turn {session.turns}] steps={result.steps} "
            f"turn_tokens={result.usage.total} session_tokens={session.usage.total} "
            f"ctx≈{context.estimate(session.history)}\n"
        )

    session.save(Path(f"./sessions/{session.id}.jsonl"))
    print(f"saved to sessions/{session.id}.jsonl")


if __name__ == "__main__":
    main()
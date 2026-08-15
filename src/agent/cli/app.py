from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv

from agent.cli import render
from agent.config.loader import load_settings
from agent.config.schema import Settings
from agent.errors import AgentError, ConfigError
from agent.llm.base import LLMProvider
from agent.llm.registry import create_provider
from agent.loop import run as run_turn
from agent.protocol import AssistantMessage, Message, ToolResultBlock, ToolUseBlock
from agent.sandbox import Sandbox
from agent.session.context import ContextManager
from agent.session.state import Session
from agent.telemetry.tracer import JsonlTracer, NullTracer, Tracer, read_events, summarize
from agent.tools.base import ToolContext
from agent.tools.dispatcher import Dispatcher, always_approve
from agent.tools.fs.edit import EditFileTool
from agent.tools.fs.read import ReadFileTool
from agent.tools.fs.write import WriteFileTool
from agent.tools.registry import ToolRegistry

load_dotenv()

SESSIONS_DIR = Path("sessions")

app = typer.Typer(help="A sandboxed CLI coding agent.", no_args_is_help=True)
config_app = typer.Typer(help="Inspect configuration.", no_args_is_help=True)
trace_app = typer.Typer(help="Inspect saved traces.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(trace_app, name="trace")

ModelOption = Annotated[str | None, typer.Option("--model", help="Model to send requests to")]
WorkdirOption = Annotated[
    Path | None, typer.Option("--workdir", help="Sandbox directory the tools may touch")
]
MaxStepsOption = Annotated[
    int | None, typer.Option("--max-steps", help="Maximum provider calls per turn")
]
ConfigOption = Annotated[
    Path | None, typer.Option("--config", help="Extra YAML config file to merge on top")
]
YesOption = Annotated[
    bool, typer.Option("--yes", "-y", help="Approve every tool call without prompting")
]


class SpinnerProvider:
    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner

    def send(
        self,
        messages: list[Message],
        tools: list[dict],
        system: str | None = None,
    ) -> AssistantMessage:
        with render.waiting():
            return self._inner.send(messages, tools, system)


class RenderingDispatcher(Dispatcher):
    def dispatch(self, block: ToolUseBlock) -> ToolResultBlock:
        render.tool_call(block.name, block.input)
        result = super().dispatch(block)
        render.tool_result(result)
        return result


class Approver:
    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    def __call__(self, tool_name: str, args: dict) -> bool:
        return render.approve(tool_name, args, self._preview(tool_name, args))

    def _preview(self, tool_name: str, args: dict):
        try:
            path = args["path"]
            target = self._sandbox.resolve(path)
            before = target.read_text(encoding="utf-8") if target.is_file() else ""

            if tool_name == "write_file":
                after = args["content"]
            elif tool_name == "edit_file":
                if before.count(args["old_string"]) != 1:
                    return None
                after = before.replace(args["old_string"], args["new_string"], 1)
            else:
                return None
        except Exception:
            return None

        return render.diff(path, before, after)


@dataclass
class Runtime:
    settings: Settings
    session: Session
    provider: LLMProvider
    dispatcher: Dispatcher
    context: ContextManager
    tracer: Tracer


def _settings(
    model: str | None = None,
    workdir: Path | None = None,
    max_steps: int | None = None,
    config: Path | None = None,
    yes: bool = False,
) -> Settings:
    overrides: dict[str, Any] = {}

    if model is not None:
        overrides["provider"] = {"model": model}

    agent_overrides: dict[str, Any] = {}
    if workdir is not None:
        agent_overrides["workdir"] = str(workdir)
    if max_steps is not None:
        agent_overrides["max_steps"] = max_steps
    if agent_overrides:
        overrides["agent"] = agent_overrides

    if yes:
        overrides["tools"] = {"auto_approve": True}

    try:
        return load_settings(config, overrides)
    except ConfigError as exc:
        render.error(str(exc))
        raise typer.Exit(1)


def _build(settings: Settings, session: Session) -> Runtime:
    workdir = settings.agent.workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    sandbox = Sandbox(workdir)

    tracer: Tracer = NullTracer()
    if settings.telemetry.enabled:
        tracer = JsonlTracer(
            trace_dir=settings.telemetry.trace_dir.expanduser(),
            session_id=session.id,
            model=settings.provider.model,
        )

    approve = always_approve if settings.tools.auto_approve else Approver(sandbox)

    dispatcher = RenderingDispatcher(
        registry=ToolRegistry([ReadFileTool(), WriteFileTool(), EditFileTool()]),
        context=ToolContext(sandbox=sandbox),
        approve=approve,
        tracer=tracer,
    )

    return Runtime(
        settings=settings,
        session=session,
        provider=SpinnerProvider(create_provider(settings.provider)),
        dispatcher=dispatcher,
        context=ContextManager(
            max_tokens=settings.context.max_tokens,
            keep_recent=settings.context.keep_recent,
        ),
        tracer=tracer,
    )


def _cost(tracer: Tracer) -> float | None:
    if isinstance(tracer, JsonlTracer):
        return tracer.totals().estimated_cost_usd
    return None


def _turn(runtime: Runtime, prompt: str) -> bool:
    try:
        result = run_turn(
            provider=runtime.provider,
            dispatcher=runtime.dispatcher,
            session=runtime.session,
            user_input=prompt,
            system=runtime.settings.agent.system_prompt,
            max_steps=runtime.settings.agent.max_steps,
            context=runtime.context,
            tracer=runtime.tracer,
        )
    except AgentError as exc:
        render.error(str(exc))
        return False

    render.answer(result.output)
    render.usage(
        steps=result.steps,
        turn_tokens=result.usage.total,
        session_tokens=runtime.session.usage.total,
        cost=_cost(runtime.tracer),
    )
    return True


@app.command()
def run(
    prompt: Annotated[str, typer.Argument(help="Task for the agent to complete")],
    model: ModelOption = None,
    workdir: WorkdirOption = None,
    max_steps: MaxStepsOption = None,
    config: ConfigOption = None,
    yes: YesOption = False,
) -> None:
    settings = _settings(model, workdir, max_steps, config, yes)
    session = Session()
    runtime = _build(settings, session)

    render.info(f"session {session.id} · {settings.provider.model}")

    if not _turn(runtime, prompt):
        raise typer.Exit(1)


@app.command()
def chat(
    model: ModelOption = None,
    workdir: WorkdirOption = None,
    max_steps: MaxStepsOption = None,
    config: ConfigOption = None,
    yes: YesOption = False,
) -> None:
    settings = _settings(model, workdir, max_steps, config, yes)
    session = Session()
    runtime = _build(settings, session)

    render.info(
        f"session {session.id} · {settings.provider.model} · "
        "type 'exit' to quit, 'reset' to clear history"
    )

    while True:
        try:
            prompt = render.console.input("[bold cyan]› [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            render.console.print()
            break

        if prompt in {"exit", "quit"}:
            break
        if prompt == "reset":
            session.reset()
            render.info("history cleared")
            continue
        if not prompt:
            continue

        _turn(runtime, prompt)

    path = SESSIONS_DIR / f"{session.id}.jsonl"
    session.save(path)
    render.info(f"session saved to {path}")


@config_app.command("show")
def config_show(
    model: ModelOption = None,
    workdir: WorkdirOption = None,
    max_steps: MaxStepsOption = None,
    config: ConfigOption = None,
    yes: YesOption = False,
) -> None:
    render.settings_table(_settings(model, workdir, max_steps, config, yes))


@trace_app.command("show")
def trace_show(
    session_id: Annotated[str, typer.Argument(help="Session id of a saved trace")],
    config: ConfigOption = None,
) -> None:
    settings = _settings(config=config)
    path = settings.telemetry.trace_dir.expanduser() / f"{session_id}.jsonl"

    if not path.is_file():
        render.error(f"No trace found at {path}")
        raise typer.Exit(1)

    render.trace_table(session_id, summarize(read_events(path)))


if __name__ == "__main__":
    app()

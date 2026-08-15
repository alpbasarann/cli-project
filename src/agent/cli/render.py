import difflib
import json

from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.status import Status
from rich.table import Table
from rich.text import Text

from agent.config.schema import Settings
from agent.protocol import ToolResultBlock
from agent.telemetry.tracer import TraceTotals

console = Console()

MAX_ARG_CHARS = 300
MAX_RESULT_LINES = 8
MAX_DIFF_LINES = 40

_DIFF_STYLES = {"+": "green", "-": "red", "@": "cyan"}


def _format_args(args: dict) -> str:
    rendered = json.dumps(args, ensure_ascii=False)
    if len(rendered) > MAX_ARG_CHARS:
        rendered = rendered[:MAX_ARG_CHARS] + "…"
    return rendered


def tool_call(name: str, args: dict) -> None:
    line = Text("• ", style="bold cyan")
    line.append(name, style="bold")
    line.append(f" {_format_args(args)}", style="dim")
    console.print(line)


def tool_result(result: ToolResultBlock) -> None:
    lines = result.content.splitlines() or [""]
    shown = lines[:MAX_RESULT_LINES]
    body = Text("\n".join(shown))

    if len(lines) > MAX_RESULT_LINES:
        body.append(f"\n… {len(lines) - MAX_RESULT_LINES} more lines", style="dim")

    style = "red" if result.is_error else "green"
    title = "error" if result.is_error else "result"
    console.print(Panel(body, title=title, border_style=style, padding=(0, 1)))


def diff(path: str, before: str, after: str) -> Text | None:
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    if not lines:
        return None

    rendered = Text()
    for index, line in enumerate(lines[:MAX_DIFF_LINES]):
        style = "dim" if index < 2 else _DIFF_STYLES.get(line[:1], "")
        rendered.append(line + "\n", style=style)

    if len(lines) > MAX_DIFF_LINES:
        rendered.append(f"… {len(lines) - MAX_DIFF_LINES} more diff lines\n", style="dim")

    return rendered


def approve(name: str, args: dict, preview: Text | None = None) -> bool:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(overflow="fold")

    for key, value in args.items():
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if len(rendered) > MAX_ARG_CHARS:
            rendered = rendered[:MAX_ARG_CHARS] + "…"
        table.add_row(key, rendered)

    body: list[RenderableType] = [table]
    if preview is not None:
        body.append(preview)

    console.print(
        Panel(
            Group(*body),
            title=f"approve {name}",
            border_style="yellow",
            padding=(0, 1),
        )
    )
    return Confirm.ask("Run this tool?", default=False, console=console)


def answer(text: str) -> None:
    console.print(Panel(Markdown(text or "(no output)"), border_style="blue", padding=(0, 1)))


def error(message: str) -> None:
    console.print(Panel(Text(message), title="error", border_style="red", padding=(0, 1)))


def info(message: str) -> None:
    console.print(Text(message, style="dim"))


def waiting(message: str = "Waiting for the provider…") -> Status:
    return console.status(message, spinner="dots")


def usage(steps: int, turn_tokens: int, session_tokens: int, cost: float | None) -> None:
    parts = [f"steps={steps}", f"turn_tokens={turn_tokens}", f"session_tokens={session_tokens}"]
    if cost is not None:
        parts.append(f"cost=${cost:.4f}")
    console.print(Text("  ".join(parts), style="dim"))


def settings_table(settings: Settings) -> None:
    table = Table(title="resolved settings", title_style="bold", box=None, padding=(0, 2))
    table.add_column("setting", style="cyan", no_wrap=True)
    table.add_column("value", overflow="fold")

    for section, values in settings.model_dump(mode="json").items():
        for key, value in values.items():
            table.add_row(f"{section}.{key}", str(value))

    console.print(table)


def trace_table(session_id: str, totals: TraceTotals) -> None:
    table = Table(title=f"trace {session_id}", title_style="bold", box=None, padding=(0, 2))
    table.add_column("metric", style="cyan", no_wrap=True)
    table.add_column("value", justify="right")

    rows = [
        ("turns", str(totals.turns)),
        ("steps", str(totals.steps)),
        ("llm calls", str(totals.llm_calls)),
        ("input tokens", f"{totals.input_tokens:,}"),
        ("output tokens", f"{totals.output_tokens:,}"),
        ("total tokens", f"{totals.total_tokens:,}"),
        ("estimated cost", f"${totals.estimated_cost_usd:.4f}"),
        ("llm time", f"{totals.llm_duration_ms / 1000:.1f}s"),
        ("tool time", f"{totals.tool_duration_ms / 1000:.1f}s"),
        ("tool calls", str(totals.tool_call_count)),
        ("tool errors", str(totals.tool_errors)),
        ("compactions", str(totals.compactions)),
        ("tokens reclaimed", f"{totals.tokens_reclaimed:,}"),
    ]
    for label, value in rows:
        table.add_row(label, value)

    console.print(table)

    if not totals.tool_calls:
        return

    breakdown = Table(title="tool calls", title_style="bold", box=None, padding=(0, 2))
    breakdown.add_column("tool", style="cyan", no_wrap=True)
    breakdown.add_column("count", justify="right")
    for name, count in sorted(totals.tool_calls.items(), key=lambda item: -item[1]):
        breakdown.add_row(name, str(count))

    console.print(breakdown)

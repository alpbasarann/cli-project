from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from evals.harness import run_evaluation
from evals.models import RunReport

app = typer.Typer(no_args_is_help=False)
console = Console()
RESULTS_DIR = Path(__file__).parent / "results"


def _render_report(report: RunReport) -> None:
    table = Table(title=f"Agent evaluation: {report.label}")
    table.add_column("Task")
    table.add_column("Passes", justify="right")
    table.add_column("Mean steps", justify="right")
    table.add_column("Mean tokens", justify="right")
    table.add_column("Mean duration", justify="right")
    table.add_column("Last failure")
    for summary in report.tasks:
        failures = [
            attempt.reason
            for attempt in report.attempts
            if attempt.task == summary.name and not attempt.passed
        ]
        table.add_row(
            summary.name,
            f"{summary.passes}/{summary.attempts}",
            f"{summary.mean_steps:.1f}",
            f"{summary.mean_tokens:.1f}",
            f"{summary.mean_duration_seconds:.2f}s",
            failures[-1] if failures else "",
        )
    console.print(table)
    console.print(f"Overall pass rate: {report.pass_rate:.1%}")


def _render_comparison(current: RunReport, previous: RunReport) -> None:
    def rate(passes: int, attempts: int) -> float:
        return passes / attempts if attempts else 0.0

    def display(passes: int, attempts: int) -> str:
        return f"{passes}/{attempts} ({rate(passes, attempts):.1%})"

    before = {task.name: task for task in previous.tasks}
    after = {task.name: task for task in current.tasks}
    for name in sorted(set(before) | set(after)):
        if name not in before:
            console.print(f"Only in current run: {name}")
        elif name not in after:
            console.print(f"Only in compared run: {name}")
        elif rate(after[name].passes, after[name].attempts) > rate(before[name].passes, before[name].attempts):
            console.print(
                f"Improved: {name} ({display(before[name].passes, before[name].attempts)} -> "
                f"{display(after[name].passes, after[name].attempts)})"
            )
        elif rate(after[name].passes, after[name].attempts) < rate(before[name].passes, before[name].attempts):
            console.print(
                f"Regressed: {name} ({display(before[name].passes, before[name].attempts)} -> "
                f"{display(after[name].passes, after[name].attempts)})"
            )
        else:
            console.print(
                f"Unchanged: {name} ({display(before[name].passes, before[name].attempts)} -> "
                f"{display(after[name].passes, after[name].attempts)})"
            )


@app.callback(invoke_without_command=True)
def main(
    repeat: Annotated[int, typer.Option("--repeat", "-r")] = 1,
    task: Annotated[list[str] | None, typer.Option("--task", "-t")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    label: Annotated[str, typer.Option("--label")] = "run",
    compare: Annotated[Path | None, typer.Option("--compare")] = None,
) -> None:
    if repeat < 1:
        raise typer.BadParameter("repeat must be at least 1")
    try:
        report = run_evaluation(repeat=repeat, selected_names=task, model=model, label=label)
    except ValueError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=2) from exc
    _render_report(report)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = RESULTS_DIR / f"{label}-{timestamp}.json"
    destination.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print(f"Wrote results: {destination}")
    if compare is not None:
        previous = RunReport.model_validate_json(compare.read_text(encoding="utf-8"))
        _render_comparison(report, previous)


if __name__ == "__main__":
    app()

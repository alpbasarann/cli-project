import importlib.util
import shutil
import tempfile
import time
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import yaml
from dotenv import load_dotenv

from agent.config.loader import load_settings
from agent.errors import AgentError, MaxStepsExceeded
from agent.llm.registry import create_provider
from agent.loop import run
from agent.protocol import AssistantMessage, UserMessage
from agent.sandbox import Sandbox
from agent.session.state import Session
from agent.tools.base import ToolContext
from agent.tools.dispatcher import Dispatcher, always_approve
from agent.tools.fs.edit import EditFileTool
from agent.tools.fs.read import ReadFileTool
from agent.tools.fs.write import WriteFileTool
from agent.tools.registry import ToolRegistry
from evals.models import Attempt, RunReport, ScoreResult, TaskDefinition, TaskSummary
from evals.scoring import ScoreContext

load_dotenv()

TASKS_DIR = Path(__file__).parent / "tasks"


def load_tasks(selected_names: list[str] | None = None) -> list[tuple[Path, TaskDefinition]]:
    tasks: list[tuple[Path, TaskDefinition]] = []
    for task_dir in sorted(TASKS_DIR.iterdir()):
        task_file = task_dir / "task.yaml"
        if not task_file.is_file():
            continue
        raw = yaml.safe_load(task_file.read_text(encoding="utf-8"))
        task = TaskDefinition.model_validate(raw)
        if selected_names is None or task.name in selected_names:
            tasks.append((task_dir, task))
    if selected_names is not None:
        found = {task.name for _, task in tasks}
        missing = sorted(set(selected_names) - found)
        if missing:
            raise ValueError(f"Unknown task: {', '.join(missing)}")
    return tasks


def _trajectory(session: Session) -> tuple[list[str], int]:
    tool_calls: list[str] = []
    tool_errors = 0
    for message in session.history:
        if isinstance(message, AssistantMessage):
            tool_calls.extend(block.name for block in message.tool_uses)
        elif isinstance(message, UserMessage):
            tool_errors += sum(block.is_error for block in message.tool_results)
    return tool_calls, tool_errors


def _load_scorer(task_dir: Path) -> ModuleType:
    module_name = f"evals_task_{task_dir.name}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, task_dir / "scorer.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load scorer for {task_dir.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_attempt(task_dir: Path, task: TaskDefinition, model: str | None) -> Attempt:
    workdir = Path(tempfile.mkdtemp()).resolve()
    fixture_dir = task_dir / "fixture"
    session = Session()
    started = time.perf_counter()
    error: str | None = None
    max_steps_exceeded = False
    output = ""
    steps = 0
    try:
        if fixture_dir.is_dir():
            shutil.copytree(fixture_dir, workdir, dirs_exist_ok=True)
        overrides: dict[str, dict[str, str | int]] = {
            "agent": {"workdir": str(workdir), "max_steps": task.max_steps}
        }
        if model is not None:
            overrides["provider"] = {"model": model}
        settings = load_settings(overrides=overrides)
        provider = create_provider(settings.provider)
        dispatcher = Dispatcher(
            registry=ToolRegistry([ReadFileTool(), WriteFileTool(), EditFileTool()]),
            context=ToolContext(sandbox=Sandbox(workdir)),
            approve=always_approve,
        )
        try:
            result = run(
                provider=provider,
                dispatcher=dispatcher,
                session=session,
                user_input=task.prompt,
                system=settings.agent.system_prompt,
                max_steps=task.max_steps,
            )
            output = result.output
            steps = result.steps
        except MaxStepsExceeded as exc:
            error = str(exc)
            max_steps_exceeded = True
            output = session.last_output
            steps = task.max_steps
        except AgentError as exc:
            error = f"{type(exc).__name__}: {exc}"
            output = session.last_output
            steps = len([message for message in session.history if isinstance(message, AssistantMessage)])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            output = session.last_output
            steps = len([message for message in session.history if isinstance(message, AssistantMessage)])
        tool_calls, tool_errors = _trajectory(session)
        context = ScoreContext(
            workdir=workdir,
            fixture_dir=fixture_dir if fixture_dir.is_dir() else None,
            output=output,
            steps=steps,
            tool_calls=tool_calls,
            tool_errors=tool_errors,
        )
        try:
            score = _load_scorer(task_dir).score(context)
        except Exception as exc:
            score = ScoreResult(passed=False, reason=f"Scorer error: {type(exc).__name__}: {exc}")
        if max_steps_exceeded:
            score = ScoreResult(passed=False, reason=f"{error}; scorer: {score.reason}")
        elif error is not None:
            score = ScoreResult(passed=False, reason=error)
        return Attempt(
            task=task.name,
            passed=score.passed,
            reason=score.reason,
            steps=steps,
            tokens=session.usage.total,
            duration_seconds=time.perf_counter() - started,
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            error=error,
        )
    except Exception as exc:
        tool_calls, tool_errors = _trajectory(session)
        return Attempt(
            task=task.name,
            passed=False,
            reason=f"{type(exc).__name__}: {exc}",
            steps=steps,
            tokens=session.usage.total,
            duration_seconds=time.perf_counter() - started,
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_evaluation(
    repeat: int,
    selected_names: list[str] | None,
    model: str | None,
    label: str,
) -> RunReport:
    tasks = load_tasks(selected_names)
    attempts = [
        run_attempt(task_dir=task_dir, task=task, model=model)
        for _ in range(repeat)
        for task_dir, task in tasks
    ]
    summaries = []
    for _, task in tasks:
        task_attempts = [attempt for attempt in attempts if attempt.task == task.name]
        count = len(task_attempts)
        summaries.append(
            TaskSummary(
                name=task.name,
                passes=sum(attempt.passed for attempt in task_attempts),
                attempts=count,
                mean_steps=sum(attempt.steps for attempt in task_attempts) / count,
                mean_tokens=sum(attempt.tokens for attempt in task_attempts) / count,
                mean_duration_seconds=sum(attempt.duration_seconds for attempt in task_attempts) / count,
            )
        )
    return RunReport(
        label=label,
        model=model or load_settings().provider.model,
        repeat=repeat,
        pass_rate=sum(attempt.passed for attempt in attempts) / len(attempts) if attempts else 0.0,
        tasks=summaries,
        attempts=attempts,
    )

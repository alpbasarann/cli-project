import subprocess
import sys
from collections.abc import Callable

from evals.models import ScoreResult
from evals.scoring import ScoreContext

Check = Callable[[ScoreContext], ScoreResult]


def file_exists(path: str) -> Check:
    def check(context: ScoreContext) -> ScoreResult:
        if context.read_workdir(path) is None:
            return ScoreResult(passed=False, reason=f"Missing file: {path}")
        return ScoreResult(passed=True, reason="passed")

    return check


def file_contains(path: str, needle: str) -> Check:
    def check(context: ScoreContext) -> ScoreResult:
        content = context.read_workdir(path)
        if content is None:
            return ScoreResult(passed=False, reason=f"Missing file: {path}")
        if needle not in content:
            return ScoreResult(passed=False, reason=f"Expected content not found in {path}")
        return ScoreResult(passed=True, reason="passed")

    return check


def file_unchanged(path: str) -> Check:
    def check(context: ScoreContext) -> ScoreResult:
        expected = context.read_fixture(path)
        actual = context.read_workdir(path)
        if expected is None:
            return ScoreResult(passed=False, reason=f"Fixture file missing: {path}")
        if actual != expected:
            return ScoreResult(passed=False, reason=f"File changed: {path}")
        return ScoreResult(passed=True, reason="passed")

    return check


def pytest_passes(timeout: float = 20.0) -> Check:
    def check(context: ScoreContext) -> ScoreResult:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                cwd=context.workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ScoreResult(passed=False, reason="pytest timed out")
        if result.returncode == 0:
            return ScoreResult(passed=True, reason="passed")
        tail = result.stdout.strip()[-1000:]
        return ScoreResult(passed=False, reason=f"pytest failed: {tail}")

    return check


def all_of(*checks: Check) -> Check:
    def check(context: ScoreContext) -> ScoreResult:
        for candidate in checks:
            result = candidate(context)
            if not result.passed:
                return result
        return ScoreResult(passed=True, reason="passed")

    return check

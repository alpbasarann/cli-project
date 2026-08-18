import re

from evals.models import ScoreResult


def score(ctx):
    expected = {
        "greeter.py": "def salute(name: str) -> str:",
        "morning.py": "from greeter import salute",
        "evening.py": "from greeter import salute",
    }
    for path, needle in expected.items():
        content = ctx.read_workdir(path)
        if content is None:
            return ScoreResult(passed=False, reason=f"Missing file: {path}")
        if needle not in content:
            return ScoreResult(passed=False, reason=f"Rename missing in {path}")
        if re.search(r"\bgreet\b", content):
            return ScoreResult(passed=False, reason=f"Old name remains in {path}")
    return ScoreResult(passed=True, reason="passed")

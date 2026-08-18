from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScoreContext:
    workdir: Path
    fixture_dir: Path | None
    output: str
    steps: int
    tool_calls: list[str]
    tool_errors: int

    def read_workdir(self, path: str) -> str | None:
        target = self.workdir / path
        if not target.is_file():
            return None
        return target.read_text(encoding="utf-8")

    def read_fixture(self, path: str) -> str | None:
        if self.fixture_dir is None:
            return None
        target = self.fixture_dir / path
        if not target.is_file():
            return None
        return target.read_text(encoding="utf-8")

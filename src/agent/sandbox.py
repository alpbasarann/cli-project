from pathlib import Path

from agent.errors import SandboxViolation


class Sandbox:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, raw: str) -> Path:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self._root / candidate

        resolved = candidate.resolve()
        if resolved != self._root and not resolved.is_relative_to(self._root):
            raise SandboxViolation(f"Path '{raw}' resolves outside the working directory")
        return resolved
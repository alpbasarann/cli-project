import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from pydantic import TypeAdapter

from agent.protocol import AssistantMessage, Message, ToolResultBlock, Usage, UserMessage

_MESSAGE_ADAPTER = TypeAdapter(Message)


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    history: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    turns: int = 0

    def start_turn(self, prompt: str) -> None:
        self.history.append(UserMessage.from_text(prompt))
        self.turns += 1

    def record_response(self, message: AssistantMessage) -> None:
        self.history.append(message)
        self.usage.input_tokens += message.usage.input_tokens
        self.usage.output_tokens += message.usage.output_tokens

    def record_tool_results(self, results: list[ToolResultBlock]) -> None:
        self.history.append(UserMessage.from_tool_results(results))

    def replace_history(self, history: list[Message]) -> None:
        self.history = history

    @property
    def last_output(self) -> str:
        for message in reversed(self.history):
            if isinstance(message, AssistantMessage):
                return message.text
        return ""

    def reset(self) -> None:
        self.history.clear()
        self.usage = Usage()
        self.turns = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": self.id, "turns": self.turns}) + "\n")
            for message in self.history:
                handle.write(message.model_dump_json() + "\n")

    @classmethod
    def load(cls, path: Path) -> "Session":
        lines = path.read_text(encoding="utf-8").splitlines()
        meta = json.loads(lines[0])
        history = [_MESSAGE_ADAPTER.validate_json(line) for line in lines[1:]]
        return cls(id=meta["id"], history=history, turns=meta["turns"])
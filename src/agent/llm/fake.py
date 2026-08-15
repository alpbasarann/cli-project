from copy import deepcopy

from agent.protocol import AssistantMessage, Message

class FakeLLMProvider:
    def __init__(self, scripted: list[AssistantMessage]) -> None:
        self._queue = list(scripted)
        self.calls: list[list[Message]] = []

    def send(
        self,
        messages: list[Message],
        tools: list[dict],
        system: str | None = None,
    ) -> AssistantMessage:
        self.calls.append(deepcopy(messages))
        if not self._queue:
            raise RuntimeError("Loop requested more responses than were scripted")
        return self._queue.pop(0)
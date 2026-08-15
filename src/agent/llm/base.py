from typing import Protocol

from agent.protocol import AssistantMessage, Message


class LLMProvider(Protocol):
    def send(
        self,
        messages: list[Message],
        tools: list[dict],
    ) -> AssistantMessage: ...
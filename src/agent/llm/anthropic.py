import time

import anthropic

from agent.errors import PermanentProviderError, TransientProviderError
from agent.llm.config import ProviderConfig
from agent.protocol import (
    AssistantMessage,
    ContentBlock,
    Message,
    StopReason,
    TextBlock,
    ToolUseBlock,
    Usage,
)

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "end_turn",
}


class AnthropicProvider:
    def __init__(self, config: ProviderConfig, client: anthropic.Anthropic | None = None) -> None:
        self._config = config
        self._client = client or anthropic.Anthropic(timeout=config.timeout_seconds)

    def send(
        self,
        messages: list[Message],
        tools: list[dict],
        system: str | None = None,
    ) -> AssistantMessage:
        payload: dict = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "messages": [self._encode(m) for m in messages],
        }
        if tools:
            payload["tools"] = tools
        if system:
            payload["system"] = system

        response = self._request_with_retry(payload)
        return self._decode(response)

    def _request_with_retry(self, payload: dict):
        delay = 1.0
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                return self._client.messages.create(**payload)
            except (anthropic.RateLimitError, anthropic.APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                if status not in (429, 500, 502, 503, 529):
                    raise PermanentProviderError(f"API error {status}: {exc}") from exc
                last_error = exc
            except anthropic.APIConnectionError as exc:
                last_error = exc

            if attempt < self._config.max_retries:
                time.sleep(delay)
                delay *= 2

        raise TransientProviderError(
            f"Failed after {self._config.max_retries + 1} attempts: {last_error}"
        ) from last_error

    def _encode(self, message: Message) -> dict:
        return {
            "role": message.role,
            "content": [b.model_dump() for b in message.content],
        }

    def _decode(self, response) -> AssistantMessage:
        blocks: list[ContentBlock] = []
        for block in response.content:
            if block.type == "text":
                blocks.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                blocks.append(ToolUseBlock(id=block.id, name=block.name, input=dict(block.input)))

        stop_reason = _STOP_REASONS.get(response.stop_reason or "end_turn", "end_turn")

        return AssistantMessage(
            content=blocks,
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )
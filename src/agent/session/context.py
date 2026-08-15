from typing import Protocol

from agent.protocol import (
    AssistantMessage,
    ContentBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

CHARS_PER_TOKEN = 4
MIN_TRIMMABLE_CHARS = 500


def estimate_tokens(history: list[Message]) -> int:
    chars = 0
    for message in history:
        for block in message.content:
            if isinstance(block, TextBlock):
                chars += len(block.text)
            elif isinstance(block, ToolResultBlock):
                chars += len(block.content)
            elif isinstance(block, ToolUseBlock):
                chars += len(str(block.input))
    return chars // CHARS_PER_TOKEN


class CompactionStrategy(Protocol):
    def apply(self, history: list[Message], keep_recent: int) -> list[Message]: ...


class TrimToolResults:
    def apply(self, history: list[Message], keep_recent: int) -> list[Message]:
        cutoff = max(0, len(history) - keep_recent)
        compacted: list[Message] = []

        for index, message in enumerate(history):
            if index >= cutoff or not isinstance(message, UserMessage):
                compacted.append(message)
                continue

            blocks: list[ContentBlock] = []
            changed = False

            for block in message.content:
                if isinstance(block, ToolResultBlock) and len(block.content) > MIN_TRIMMABLE_CHARS:
                    lines = block.content.count("\n") + 1
                    blocks.append(
                        ToolResultBlock(
                            tool_use_id=block.tool_use_id,
                            content=f"[trimmed: {lines} lines removed to save context; re-read if needed]",
                            is_error=block.is_error,
                        )
                    )
                    changed = True
                else:
                    blocks.append(block)

            compacted.append(UserMessage(content=blocks) if changed else message)

        return compacted


class ContextManager:
    def __init__(
        self,
        max_tokens: int = 60_000,
        keep_recent: int = 6,
        strategy: CompactionStrategy | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._keep_recent = keep_recent
        self._strategy = strategy or TrimToolResults()

    def estimate(self, history: list[Message]) -> int:
        return estimate_tokens(history)

    def needs_compaction(self, history: list[Message]) -> bool:
        return self.estimate(history) > self._max_tokens

    def compact(self, history: list[Message]) -> list[Message]:
        if not self.needs_compaction(history):
            return history
        return self._strategy.apply(history, self._keep_recent)
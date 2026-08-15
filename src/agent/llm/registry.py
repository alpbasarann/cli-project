from agent.errors import AgentError
from agent.llm.base import LLMProvider
from agent.llm.config import ProviderConfig


def create_provider(config: ProviderConfig) -> LLMProvider:
    if config.provider == "anthropic":
        from agent.llm.anthropic import AnthropicProvider

        return AnthropicProvider(config)
    raise AgentError(f"Unknown provider: '{config.provider}'")
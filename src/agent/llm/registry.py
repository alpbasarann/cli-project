from agent.config.schema import ProviderSettings
from agent.errors import AgentError
from agent.llm.base import LLMProvider


def create_provider(config: ProviderSettings) -> LLMProvider:
    if config.name == "anthropic":
        from agent.llm.anthropic import AnthropicProvider

        return AnthropicProvider(config)
    raise AgentError(f"Unknown provider: '{config.name}'")

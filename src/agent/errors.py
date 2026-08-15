class AgentError(Exception):
    pass


class ProviderError(AgentError):
    pass


class TransientProviderError(ProviderError):
    pass


class PermanentProviderError(ProviderError):
    pass


class ConfigError(AgentError):
    pass


class MaxStepsExceeded(AgentError):
    pass


class SandboxViolation(AgentError):
    pass
class AgentError(Exception):
    pass


class ProviderError(AgentError):
    pass


class TransientProviderError(ProviderError):
    pass


class PermanentProviderError(ProviderError):
    pass


class MaxStepsExceeded(AgentError):
    pass
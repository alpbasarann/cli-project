from pydantic import BaseModel, Field

class ProviderConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5"
    max_tokens: int = Field(default=4096, gt=0)
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=120.0, gt=0)
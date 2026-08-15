from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_SYSTEM_PROMPT = (
    "You are a CLI coding agent working inside a sandboxed directory. "
    "Read files before editing them, and make the smallest change that solves the task."
)


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderSettings(Section):
    name: str = "anthropic"
    model: str = "claude-sonnet-4-5"
    max_tokens: int = Field(default=4096, gt=0)
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=120.0, gt=0)


class AgentSettings(Section):
    workdir: Path = Path("workspace")
    max_steps: int = Field(default=20, gt=0)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class ContextSettings(Section):
    max_tokens: int = Field(default=60_000, gt=0)
    keep_recent: int = Field(default=6, ge=0)


class ToolSettings(Section):
    auto_approve: bool = False


class TelemetrySettings(Section):
    enabled: bool = True
    trace_dir: Path = Path("traces")


class Settings(Section):
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)

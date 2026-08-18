from pydantic import BaseModel


class TaskDefinition(BaseModel):
    name: str
    prompt: str
    max_steps: int


class ScoreResult(BaseModel):
    passed: bool
    reason: str


class Attempt(BaseModel):
    task: str
    passed: bool
    reason: str
    steps: int
    tokens: int
    duration_seconds: float
    tool_calls: list[str]
    tool_errors: int
    error: str | None = None


class TaskSummary(BaseModel):
    name: str
    passes: int
    attempts: int
    mean_steps: float
    mean_tokens: float
    mean_duration_seconds: float


class RunReport(BaseModel):
    label: str
    model: str
    repeat: int
    pass_rate: float
    tasks: list[TaskSummary]
    attempts: list[Attempt]

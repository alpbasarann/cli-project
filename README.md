# cli-agent

A minimal CLI coding agent built from scratch in Python. It connects to an LLM,
exposes file read/write/edit as tools, and runs a multi-step loop that lets the
model inspect and modify files inside a sandboxed working directory.

Deliberately small: the core mechanics of an agent — tool calling, an agent
loop, error feedback, context management, permissions and telemetry — with no
framework in between.

## Setup

Requires Python 3.12+ and an Anthropic API key. Create a `.env` in the repo
root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

This is the only required variable. `.env` is gitignored and excluded from the
Docker image.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Running

```bash
agent run "create hello.py that prints a greeting"
agent run "find and fix the bug in calc.py" --max-steps 10 --yes
agent chat                    # interactive, multi-turn
agent config show
agent trace show <session_id>
```

In `chat`, context carries across turns so follow-ups can refer to earlier work.
`reset` clears history, `exit` saves the session to `sessions/{id}.jsonl`.

Via Docker:

```bash
docker compose build
docker compose run --rm agent chat
```

The container runs as a non-root user. Only mounted volumes persist:
`workspace/`, `traces/`, `sessions/`.

## Configuration

Settings resolve in layers, each overriding the last: `config/default.yaml` →
`~/.agent/config.yaml` → `--config <file>` → environment variables
(`AGENT_MODEL`, `AGENT_WORKDIR`, `AGENT_MAX_STEPS`) → CLI flags. Merging is
deep, so overriding `provider.model` leaves the rest of the block intact.

```yaml
provider:
  model: claude-sonnet-4-5
  max_tokens: 4096
  max_retries: 3

agent:
  workdir: workspace
  max_steps: 20
  system_prompt: >-
    You are a CLI coding agent working inside a sandboxed directory.

context:
  max_tokens: 60000
  keep_recent: 6

telemetry:
  enabled: true
  trace_dir: traces
```

## How a message flows

An LLM is a stateless function: text in, text out. Tool calling lets it emit a
structured request to run a named tool, but the host still executes it and feeds
the result back. The model is the brain; this codebase is the hands.

1. **Entry** — `cli/app.py` resolves settings and builds the provider,
   dispatcher, session and tracer.
2. **Turn** — `Session.start_turn()` appends a `UserMessage`; `loop.run()` takes
   over.
3. **Schemas** — `ToolRegistry.schemas()` derives each tool's JSON schema from
   its Pydantic input model, so the schema the model sees and the validation
   applied to its output can never drift apart.
4. **Encode** — `AnthropicProvider.send()` converts internal messages to the
   wire format. The system prompt goes in as a separate field, not a message,
   because that is this vendor's shape. SDK-shaped data exists nowhere else.
5. **Transport** — 429/500/502/503/529 are retried with exponential backoff;
   anything else raises `PermanentProviderError` immediately, since retrying a
   bad key wastes time.
6. **Decode** — the response becomes an `AssistantMessage`. Unknown block types
   are skipped so a new API feature cannot crash the loop. Token counts are
   captured into `Usage`.
7. **Branch** — if `stop_reason` is not `tool_use`, the turn ends. Otherwise the
   requested tools run.
8. **Dispatch** — look up the tool, validate arguments, request approval if it
   mutates state, resolve paths through the sandbox, execute. Every failure
   along this path — unknown tool, bad arguments, rejected approval, escaped
   path, unexpected exception — returns a `ToolResultBlock` with `is_error=True`
   instead of raising. The model reads the message next turn and corrects
   itself. This feedback path is what separates an agent from a brittle
   pipeline.
9. **Results** — all results from one assistant message go into a single
   `UserMessage`. The protocol requires every `tool_use` id to have a matching
   `tool_result` in the very next message; because the loop builds them
   together, that holds structurally.
10. **Compact** — if the history exceeds its budget, older large tool results
    have their content replaced with a placeholder. Blocks are shortened, never
    deleted — deleting would break the tool_use/tool_result pairing. Recent
    turns stay untouched.
11. **Repeat** — until the model stops requesting tools or `max_steps` is hit.

Two design notes. Paths are checked with `Path.resolve()`, which follows
symlinks, so a link planted inside the sandbox pointing out of it is still
caught; inspecting the string for `..` would not be. And `edit_file` requires
`old_string` to occur exactly once: if the model has not read the current file,
the edit is refused rather than applied to the wrong place.

## Project structure

```
src/agent/
├── protocol.py           message and content block types
├── sandbox.py            path containment
├── errors.py             exception hierarchy
├── loop.py               orchestration
├── llm/
│   ├── base.py           LLMProvider protocol
│   ├── anthropic.py      encoding, transport, retry, decoding
│   ├── fake.py           scripted provider for offline runs
│   └── registry.py       provider selection from config
├── tools/
│   ├── base.py           Tool protocol, ToolResult, ToolContext
│   ├── registry.py       schema generation and lookup
│   ├── dispatcher.py     validation, approval, execution, error capture
│   └── fs/               read, write, edit
├── session/
│   ├── state.py          history, usage, persistence
│   └── context.py        token estimation and compaction
├── telemetry/
│   ├── events.py         structured event models
│   └── tracer.py         JSONL tracer and cost accounting
├── config/
│   ├── schema.py         settings models
│   └── loader.py         layered resolution
└── cli/
    ├── app.py            Typer commands
    └── render.py         Rich output
```

Dependencies run one way: `protocol` ← `tools`/`llm`/`session` ← `loop` ←
`cli`. Nothing under `src/agent/` imports from `cli/`, and the loop imports only
protocols — which is why it runs unchanged against a fake provider with no API
calls at all.

## Telemetry

Each session writes `traces/{session_id}.jsonl`, one JSON object per line:

```json
{"type": "llm_call_completed", "model": "claude-sonnet-4-5", "input_tokens": 1048, "output_tokens": 91, "duration_ms": 1794.5, "stop_reason": "tool_use"}
{"type": "tool_executed", "tool_name": "write_file", "is_error": false, "duration_ms": 1.19}
{"type": "turn_completed", "steps": 2, "total_tokens": 2318, "estimated_cost_usd": 0.008286}
```

Two patterns show up in almost every trace: input tokens grow each step because
the full history is resent, so cost rises faster than step count; and LLM
latency dominates tool latency by three orders of magnitude, which is where any
optimisation belongs.
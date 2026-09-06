# Step 7 — The Dispatcher — Retry Policy

> [Back to index](README.md) · Previous: [The Dispatcher — Validating Calls](07-the-dispatcher-validating-calls.md) · Next: [The ReAct Loop](09-the-react-loop.md)

## Goal

Finish `dispatch_tool_call()`: add the rule that a side-effecting tool is never blind-retried after it's already failed once this task, and classify execution failures as infrastructure (retryable only if the tool is idempotent) or a correctable argument problem (always retryable).

## Why this matters

This is the app's central design decision, and the whole reason `ToolError` carries a `retryable_by_model` flag instead of just a message string. Two different kinds of "the tool didn't run" need two different responses:

- **`calculator` raises on `'1/0'`** — that's a reasoning error. The model sent bad input; if it's told exactly what was wrong, it can plausibly fix its own next `Action Input`. Safe to feed back and retry, no matter how many times it's happened.
- **`save_note` fails once** — maybe the write actually landed and only the confirmation was lost, maybe it didn't. Either way, retrying blindly risks writing the note twice. This isn't something the model can reason its way out of by trying again with different arguments; the right answer is to refuse the second attempt entirely, not just discourage it.

The same asymmetry applies to infrastructure failures (`TimeoutError`, `ConnectionError`): a `web_search` that times out is worth retrying, because searching again changes nothing about state — but if a hypothetical side-effecting tool timed out, you'd have no way to know whether the effect landed before the connection dropped. That's exactly what `IDEMPOTENT_TOOLS` from Step 3 exists to answer with a simple set-membership check, instead of guessing per failure.

## 1. Update the imports

```python
from tools import IDEMPOTENT_TOOLS, SIDE_EFFECTING_TOOLS, TOOLS, tools_prompt_block
```

## 2. Add `tool_failure_counts` and the side-effecting guard

Replace the Step 6 version of `dispatch_tool_call()` with:

```python
def dispatch_tool_call(tool_name: str, raw_args, tool_failure_counts: dict[str, int]):
    """Returns either the tool's result, or a ToolError describing why it
    didn't run and whether the model should be allowed to try again."""
    if tool_name not in TOOLS:
        # Suggest the closest valid name, never auto-substitute — the model
        # must confirm its own intent on the next turn.
        suggestion = get_close_matches(tool_name, TOOLS.keys(), n=1, cutoff=0.6)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        return ToolError(message=f"Unknown tool '{tool_name}'. Available: {list(TOOLS)}.{hint}", retryable_by_model=True)

    if isinstance(raw_args, str):
        return ToolError(message=f"Arguments for '{tool_name}' were not valid JSON: {raw_args}", retryable_by_model=True)

    spec = TOOLS[tool_name]
    try:
        validated = spec["schema"].model_validate(raw_args)
    except ValidationError as e:
        return ToolError(message=f"Invalid arguments for '{tool_name}': {e}", retryable_by_model=True)

    # A side-effecting tool that already failed once this task is never
    # blind-retried — a second identical call could double an effect (e.g.
    # two saved copies of the same note) if the first partially landed.
    if tool_name in SIDE_EFFECTING_TOOLS and tool_failure_counts.get(tool_name, 0) > 0:
        return ToolError(
            message=f"'{tool_name}' already failed once this task and is not safe to auto-retry.",
            retryable_by_model=False,
        )

    try:
        return spec["execute"](**validated.model_dump())
    except (TimeoutError, ConnectionError) as e:
        # Infrastructure failure, not a reasoning error — only worth another
        # attempt if the tool is idempotent.
        return ToolError(message=f"'{tool_name}' unavailable: {e}", retryable_by_model=tool_name in IDEMPOTENT_TOOLS)
    except ValueError as e:
        # A predictable, tool-raised argument problem (e.g. calculator's bad
        # expression) — the model can plausibly correct it.
        return ToolError(message=str(e), retryable_by_model=True)
```

What changed from Step 6, in order:

- The signature gains `tool_failure_counts: dict[str, int]` — a per-task tally the caller (`run_agent()`, built next step) maintains and passes in on every call. The dispatcher itself is stateless; it only ever reads this dict, never owns it.
- After schema validation passes, a new guard checks two things at once: is this tool side-effecting, *and* has it already failed at least once this task? Only when both are true does the dispatcher refuse outright, with `retryable_by_model=False` — the one value in this whole function that isn't `True`.
- `spec["execute"](...)` is now wrapped in `try`/`except`, splitting failures into two typed buckets: `(TimeoutError, ConnectionError)` for infrastructure, whose `retryable_by_model` is computed from `IDEMPOTENT_TOOLS` rather than hardcoded; and `ValueError` for a tool-raised argument problem (this is exactly what `calculator` raises), always retryable since it's information the model can act on.

Notice the ordering: the side-effecting guard runs *before* `execute()` is ever called, using the failure count from *previous* calls. The `except` blocks handle a failure from *this* call, which is what increments the count the *next* call will see (that increment happens in `run_agent()`, not here — the dispatcher only decides, it doesn't track).

## Try it

Contrast an idempotent tool against a side-effecting one, both under a nonzero failure count:

```bash
uv run python -c "
from agent import dispatch_tool_call
print(dispatch_tool_call('save_note', {'title': 'x', 'content': 'y'}, {'save_note': 1}))
print(dispatch_tool_call('calculator', {'expression': '1/0'}, {'calculator': 5}))
"
```

Expected output:

```text
message="'save_note' already failed once this task and is not safe to auto-retry." retryable_by_model=False
message="could not evaluate '1/0': division by zero" retryable_by_model=True
```

`save_note` is refused outright because it's in `SIDE_EFFECTING_TOOLS` and its failure count is nonzero — it never even reaches `execute()`. `calculator` runs anyway, despite a failure count of `5`, because `calculator` isn't in `SIDE_EFFECTING_TOOLS` at all; its `ValueError` is still reported as retryable regardless of how many times it's happened. One line of code — `tool_name in SIDE_EFFECTING_TOOLS` — is the entire difference in behavior between the two tools.

## Checkpoint

<details>
<summary>Full <code>agent.py</code> (through the complete <code>dispatch_tool_call</code>)</summary>

```python
#!/usr/bin/env python3
"""Step 3 — a small ReAct agent with a real dispatcher and guardrails.

This generalizes the single-tool-hop chat app from Step 2 of this series
into the full ReAct loop: `Thought -> Action -> Observation`, repeated
until the model itself decides it has enough to answer, capped by a
`max_steps` guardrail. The dispatcher below handles four failure modes:
hallucinated tool names, malformed/truncated arguments, execution errors,
and (new in this app) never blind-retrying a side-effecting tool.

Run:
    uv run agent.py "What is 17 * 12.99, and is that under 220?"
    uv run agent.py            # interactive: one goal per line
"""
import json
import os
import re
from dataclasses import dataclass
from difflib import get_close_matches

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from tools import IDEMPOTENT_TOOLS, SIDE_EFFECTING_TOOLS, TOOLS, tools_prompt_block

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")
DEFAULT_MAX_STEPS = 6  # a step cap is a guardrail, not decoration

# --------------------------------------------------------------------------
# Profile — the fixed identity/constraints shaping every decision. Includes
# a worked example so a small local model has something concrete to
# imitate, since it can't be trusted to infer a strict output format from a
# description alone.
# --------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are a careful problem-solving agent. Work step by step and use tools instead of guessing.

{tools_prompt_block()}

Respond using EXACTLY this format, one block per turn:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: {{"key": "value"}}

After you receive an Observation, continue with another Thought, and either
another Action, or finish with:

Thought: <your reasoning for why you're done>
Final Answer: <your answer to the user's request>

Example:

Thought: I need to compute 17 * 12.99 before I can answer.
Action: calculator
Action Input: {{"expression": "17 * 12.99"}}

Rules:
- Never compute arithmetic yourself — always call calculator.
- Only call save_note when the user explicitly asks you to save or remember something.
- If a tool call fails, read the error and correct your Action Input on the next turn.
- Give a Final Answer as soon as you genuinely have enough information — don't call tools you don't need.
"""

# --------------------------------------------------------------------------
# Parsing the model's Thought / Action / Action Input / Final Answer reply.
# Split into separate regexes (rather than one big pattern) so we can tell
# apart "no action at all", "action with no parseable JSON" (truncated or
# malformed), and a clean call.
# --------------------------------------------------------------------------
THOUGHT_PATTERN = re.compile(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer):|\Z)", re.DOTALL)
FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)
ACTION_NAME_PATTERN = re.compile(r"Action:\s*(\w+)", re.DOTALL)
ACTION_INPUT_LABEL_PATTERN = re.compile(r"Action Input:\s*(.*)", re.DOTALL)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ParsedReply:
    thought: str
    action: str | None = None
    action_input: dict | str | None = None  # str means "malformed/truncated"
    final_answer: str | None = None


def parse_reply(text: str) -> ParsedReply:
    thought_match = THOUGHT_PATTERN.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    final_match = FINAL_ANSWER_PATTERN.search(text)
    if final_match:
        return ParsedReply(thought=thought, final_answer=final_match.group(1).strip())

    action_match = ACTION_NAME_PATTERN.search(text)
    if not action_match:
        # The model didn't follow the format at all. Rather than looping on
        # an unparseable turn, the caller treats the raw reply as the answer.
        return ParsedReply(thought=thought)

    action_name = action_match.group(1)
    input_match = ACTION_INPUT_LABEL_PATTERN.search(text)
    search_space = input_match.group(1) if input_match else text[action_match.end() :]
    json_match = JSON_OBJECT_PATTERN.search(search_space)

    if not json_match:
        return ParsedReply(thought=thought, action=action_name, action_input=search_space.strip())

    try:
        action_input = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        action_input = json_match.group(0)

    return ParsedReply(thought=thought, action=action_name, action_input=action_input)


# --------------------------------------------------------------------------
# Dispatcher — validates and executes a call, and decides whether a failure
# is worth feeding back to the model for another attempt.
# --------------------------------------------------------------------------
class ToolError(BaseModel):
    message: str
    retryable_by_model: bool  # True: the model can plausibly fix this itself


def dispatch_tool_call(tool_name: str, raw_args, tool_failure_counts: dict[str, int]):
    """Returns either the tool's result, or a ToolError describing why it
    didn't run and whether the model should be allowed to try again."""
    if tool_name not in TOOLS:
        # Suggest the closest valid name, never auto-substitute — the model
        # must confirm its own intent on the next turn.
        suggestion = get_close_matches(tool_name, TOOLS.keys(), n=1, cutoff=0.6)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        return ToolError(message=f"Unknown tool '{tool_name}'. Available: {list(TOOLS)}.{hint}", retryable_by_model=True)

    if isinstance(raw_args, str):
        return ToolError(message=f"Arguments for '{tool_name}' were not valid JSON: {raw_args}", retryable_by_model=True)

    spec = TOOLS[tool_name]
    try:
        validated = spec["schema"].model_validate(raw_args)
    except ValidationError as e:
        return ToolError(message=f"Invalid arguments for '{tool_name}': {e}", retryable_by_model=True)

    # A side-effecting tool that already failed once this task is never
    # blind-retried — a second identical call could double an effect (e.g.
    # two saved copies of the same note) if the first partially landed.
    if tool_name in SIDE_EFFECTING_TOOLS and tool_failure_counts.get(tool_name, 0) > 0:
        return ToolError(
            message=f"'{tool_name}' already failed once this task and is not safe to auto-retry.",
            retryable_by_model=False,
        )

    try:
        return spec["execute"](**validated.model_dump())
    except (TimeoutError, ConnectionError) as e:
        # Infrastructure failure, not a reasoning error — only worth another
        # attempt if the tool is idempotent.
        return ToolError(message=f"'{tool_name}' unavailable: {e}", retryable_by_model=tool_name in IDEMPOTENT_TOOLS)
    except ValueError as e:
        # A predictable, tool-raised argument problem (e.g. calculator's bad
        # expression) — the model can plausibly correct it.
        return ToolError(message=str(e), retryable_by_model=True)


def call_model(client: OpenAI, model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def main() -> None:
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key="not-needed")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is 17 * 12.99, and is that under 220?"},
    ]
    print(call_model(client, DEFAULT_MODEL, messages))


if __name__ == "__main__":
    main()
```

</details>

This matches [../03-small-agent/agent.py](../03-small-agent/agent.py)'s `dispatch_tool_call()` (and everything above it) exactly — only `call_model` and the temporary `main()` below it still differ from the final file, and Step 8 replaces `main()`.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `save_note` gets blocked on its very first call, before it's ever failed | `tool_failure_counts.get(tool_name, 0) > 0` written as `>= 0`, which is always true | Use `> 0` — a fresh task with no prior failures must still allow the first attempt |
| `calculator` stops being retryable after enough failures | `IDEMPOTENT_TOOLS` check accidentally combined with the side-effecting guard instead of kept in the separate `except (TimeoutError, ConnectionError)` branch | Only `SIDE_EFFECTING_TOOLS` membership blocks a retry outright; `IDEMPOTENT_TOOLS` only affects infrastructure-failure retryability |
| A `TimeoutError` from a tool crashes the app instead of returning a `ToolError` | `execute()` call not wrapped in `try`/`except`, or the exception tuple doesn't include `TimeoutError` | Match the `except (TimeoutError, ConnectionError) as e:` line exactly |

Next: **[The ReAct Loop](09-the-react-loop.md)** — assemble `parse_reply` and `dispatch_tool_call` into the actual loop that runs a goal to completion.

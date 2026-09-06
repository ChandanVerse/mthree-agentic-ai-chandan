# Step 6 — The Dispatcher — Validating Calls

> [Back to index](README.md) · Previous: [Parsing the Reply](06-parsing-the-reply.md) · Next: [The Dispatcher — Retry Policy](08-the-dispatcher-retry-policy.md)

## Goal

Build the first half of `dispatch_tool_call()`: turn a parsed `(tool_name, raw_args)` pair into either a real tool result or a specific `ToolError`, covering an unknown tool name, malformed arguments, and a schema mismatch — the same three checks [02-chat-with-web-search's `resolve_tool_call()`](../02-chat-with-web-search-walkthrough/06-validating-and-executing.md) makes.

## Why this matters

If you've built Step 2's dispatcher, none of these three checks are new ideas — they're the same reasoning, just returning a typed `ToolError` object instead of a `(result, error_message)` tuple. That change in shape matters for what comes next: this app needs to attach more than just an error message to a failure — it needs to say whether the *model* can plausibly fix it. `ToolError` carries that as an explicit field, `retryable_by_model`, rather than the caller having to infer it from the error text.

This step deliberately stops short of the full function. Everything here treats a failure as retryable by default (`retryable_by_model=True`), because at this stage the dispatcher doesn't yet know about the idempotent-vs-side-effecting distinction from Step 3 — that's what Step 7 adds. Getting the validation checks right first, independent of the retry policy, mirrors how you'd actually build and test this in isolation.

## 1. New imports

Update the top of `agent.py`:

```python
import json
import os
import re
from dataclasses import dataclass
from difflib import get_close_matches

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from tools import TOOLS, tools_prompt_block
```

`BaseModel` here is for `ToolError` itself, not a tool schema — `ToolError` is a Pydantic model for the same reason the tool argument schemas are: a fixed shape (`message: str`, `retryable_by_model: bool`) that's easy to construct correctly and easy to pattern-match on later (`isinstance(result, ToolError)`).

## 2. `ToolError`

Add this after `parse_reply()`:

```python
# --------------------------------------------------------------------------
# Dispatcher — validates and executes a call, and decides whether a failure
# is worth feeding back to the model for another attempt.
# --------------------------------------------------------------------------
class ToolError(BaseModel):
    message: str
    retryable_by_model: bool  # True: the model can plausibly fix this itself
```

## 3. `dispatch_tool_call()` — validation only

```python
def dispatch_tool_call(tool_name: str, raw_args):
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

    return spec["execute"](**validated.model_dump())
```

Three checks, each guarding one specific failure, same order as Step 2's `resolve_tool_call`:

1. **Unknown tool name** — `get_close_matches` suggests, but the dispatcher never auto-substitutes. If it silently rewrote `calculater` to `calculator` and ran it, the model would get an answer from a tool it didn't confidently name — with no error to notice or learn from. Surfacing "did you mean X?" keeps the model in the loop.
2. **Malformed JSON** — recognized cheaply, because `parse_reply` already marked this case by returning a `str` instead of a `dict` for `action_input`. No re-parsing needed here, just a type check.
3. **Schema mismatch** — `spec["schema"].model_validate(raw_args)` runs the tool's Pydantic schema; the `ValidationError`'s own message is passed straight through, since it already names the exact field and problem.

Once all three pass, `spec["execute"](**validated.model_dump())` runs the tool directly and returns whatever it returns — no error handling around execution yet. That's exactly what Step 7 adds.

## Try it

Test all three error paths plus the success path directly, without a model involved:

```bash
uv run python -c "
from agent import dispatch_tool_call
print(dispatch_tool_call('calculater', {'expression': '1+1'}))
print(dispatch_tool_call('calculator', '{\"expression\":'))
print(dispatch_tool_call('calculator', {'wrong_field': '1+1'}))
print(dispatch_tool_call('calculator', {'expression': '17 * 12.99'}))
"
```

Expected output:

```text
message="Unknown tool 'calculater'. Available: ['calculator', 'web_search', 'save_note']. Did you mean 'calculator'?" retryable_by_model=True
message='Arguments for \'calculator\' were not valid JSON: {"expression":' retryable_by_model=True
message="Invalid arguments for 'calculator': 1 validation error for CalculatorArgs\nexpression\n  Field required [type=missing, input_value={'wrong_field': '1+1'}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.13/v/missing"
220.83
```

The three error cases print as `ToolError` objects (Pydantic's default `repr`); the last line is a bare `float`, returned straight from `calculator()` with no wrapping at all — the dispatcher hasn't yet unified "what a caller gets back" into one consistent shape. Nothing in `run_agent()` needs that yet either, since it isn't built until Step 8.

## Checkpoint

<details>
<summary>Full <code>agent.py</code> (through <code>dispatch_tool_call</code>, validation only)</summary>

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

from tools import TOOLS, tools_prompt_block

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


def dispatch_tool_call(tool_name: str, raw_args):
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

    return spec["execute"](**validated.model_dump())


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

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| Fuzzy-match hint never appears, even for an obvious typo like `calc` | `cutoff=0.6` treats `"calc"` as too dissimilar from `"calculator"` by character-overlap ratio | Expected — `difflib`'s similarity metric isn't the same as "obviously the same word to a human"; try a closer typo like `calculater` |
| `dispatch_tool_call` raises `pydantic.ValidationError` instead of returning a `ToolError` | The `try`/`except ValidationError` block is missing or catches the wrong exception | Confirm the `except ValidationError as e:` line matches the checkpoint exactly |
| Calling `dispatch_tool_call('calculator', {'expression': '1/0'})` raises instead of returning a `ToolError` | Expected at this step — `calculator`'s `ValueError` isn't caught yet | This is exactly what Step 7 adds; not a bug in this step's code |

Next: **[The Dispatcher — Retry Policy](08-the-dispatcher-retry-policy.md)** — add the side-effecting no-retry rule and classify execution failures as retryable or not.

# Step 9 — The max_steps Guardrail

> [Back to index](README.md) · Previous: [The ReAct Loop](09-the-react-loop.md) · Next: [CLI and Interactive Mode](11-cli-and-interactive-mode.md)

## Goal

Replace Step 8's `while True` with `for step_num in range(1, max_steps + 1)`, add a `max_steps` parameter, and return an honest fallback message if the loop ever exhausts it without reaching a `Final Answer`.

## Why this matters

This is the guardrail that's easiest to skip on a first pass, precisely because the happy path never needs it — a model that reliably converges in two or three steps will never notice a step cap exists. It only matters the moment something goes wrong: a model that keeps calling `web_search` with slight query variations, never quite satisfied; a model that gets stuck in a loop calling `calculator` on the same expression twice; a genuinely hard goal that needs more reasoning than the model can produce cleanly. Without a cap, every one of those situations means `run_agent()` never returns at all — the difference between "the model might get stuck" and "the app might hang forever" is entirely this one guardrail.

A bounded loop with an honest fallback message converts an unbounded liability into a predictable, worst-case behavior: at most `max_steps` model calls, then a clear "I didn't finish" response instead of a silent hang. That predictability is worth having even though — especially because — you can't always trigger it on demand with a live model that happens to converge quickly.

## 1. Rewrite `run_agent()` with the step cap

Replace the Step 8 version with:

```python
def run_agent(
    client: OpenAI,
    model: str,
    goal: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    verbose: bool = True,
) -> str:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    tool_failure_counts: dict[str, int] = {}

    for step_num in range(1, max_steps + 1):
        reply = call_model(client, model, messages)
        parsed = parse_reply(reply)

        if verbose and parsed.thought:
            print(f"[step {step_num}] Thought: {parsed.thought}")

        if parsed.final_answer is not None:
            return parsed.final_answer

        if parsed.action is None:
            # Format wasn't followed at all — don't spin on an unparseable
            # turn, just surface what the model actually said.
            return reply

        if verbose:
            print(f"[step {step_num}] Action: {parsed.action}  Input: {parsed.action_input}")

        result = dispatch_tool_call(parsed.action, parsed.action_input, tool_failure_counts)
        messages.append({"role": "assistant", "content": reply})

        if isinstance(result, ToolError):
            tool_failure_counts[parsed.action] = tool_failure_counts.get(parsed.action, 0) + 1
            observation = f"Error: {result.message}"
            if not result.retryable_by_model:
                # This is the guardrail the `retryable_by_model` flag exists
                # for: an infra or side-effecting failure stops the loop
                # instead of being fed back for another (potentially harmful
                # or futile) attempt.
                return f"Stopped — {result.message}"
        else:
            observation = result if isinstance(result, str) else json.dumps(result)

        if verbose:
            print(f"[step {step_num}] Observation: {observation}\n")

        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return f"Stopped: exceeded max_steps ({max_steps}) without reaching a final answer."
```

What changed from Step 8, and why each change matters:

- `max_steps: int = DEFAULT_MAX_STEPS` joins the signature — callers can override it (the CLI in Step 10 exposes this as `--max-steps`), but every call has a cap by default, not just the ones that remember to ask for one.
- `while True:` becomes `for step_num in range(1, max_steps + 1):` — the loop can now run at most `max_steps` times, full stop, no matter what the model does inside it.
- `step_num` is a genuinely new piece of information, not just bookkeeping: the three `print()` calls now prefix every line with `[step {step_num}]`, so a verbose trace shows exactly which iteration produced which thought, action, or observation — invaluable when a trainee is trying to figure out *where* a long-running goal went wrong.
- If the `for` loop exhausts every iteration without hitting a `return` inside it — no `Final Answer`, no unparseable reply, no non-retryable error — control falls through to the final `return`, a fixed, honest message naming the exact `max_steps` value that was hit.

## Try it

Force the guardrail to fire by giving it too little room to work with, on a goal that genuinely needs more than one step:

```bash
uv run python -c "
from openai import OpenAI
from agent import run_agent, DEFAULT_MODEL, DEFAULT_BASE_URL

client = OpenAI(base_url=DEFAULT_BASE_URL, api_key='not-needed')
goal = 'A store sells 17 units at \$12.99 each with 8% discount and 6% tax on the discounted amount. Is the total under \$216?'
print(run_agent(client, DEFAULT_MODEL, goal, max_steps=1))
"
```

Expected output (a three-step calculation like this one cannot finish in a single step):

```text
[step 1] Thought: I need the subtotal first.
[step 1] Action: calculator  Input: {'expression': '17 * 12.99'}
[step 1] Observation: 220.83

Stopped: exceeded max_steps (1) without reaching a final answer.
```

Re-run the same call with `max_steps=6` (or omit the argument to use `DEFAULT_MAX_STEPS`) and it should reach a real `Final Answer` instead — the exact same goal this series' [../03-small-agent/README.md](../03-small-agent/README.md#example-trace) walks through as a worked example, including the trap of computing tax on the discounted amount rather than the original subtotal.

## Checkpoint

<details>
<summary>Full <code>agent.py</code> (through <code>run_agent</code>, capped, temporary <code>main()</code>)</summary>

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


# --------------------------------------------------------------------------
# Controller — the ordinary code that runs the loop: call the model, parse
# its decision, dispatch it, feed back the result.
# --------------------------------------------------------------------------
def call_model(client: OpenAI, model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def run_agent(
    client: OpenAI,
    model: str,
    goal: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    verbose: bool = True,
) -> str:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    tool_failure_counts: dict[str, int] = {}

    for step_num in range(1, max_steps + 1):
        reply = call_model(client, model, messages)
        parsed = parse_reply(reply)

        if verbose and parsed.thought:
            print(f"[step {step_num}] Thought: {parsed.thought}")

        if parsed.final_answer is not None:
            return parsed.final_answer

        if parsed.action is None:
            # Format wasn't followed at all — don't spin on an unparseable
            # turn, just surface what the model actually said.
            return reply

        if verbose:
            print(f"[step {step_num}] Action: {parsed.action}  Input: {parsed.action_input}")

        result = dispatch_tool_call(parsed.action, parsed.action_input, tool_failure_counts)
        messages.append({"role": "assistant", "content": reply})

        if isinstance(result, ToolError):
            tool_failure_counts[parsed.action] = tool_failure_counts.get(parsed.action, 0) + 1
            observation = f"Error: {result.message}"
            if not result.retryable_by_model:
                # This is the guardrail the `retryable_by_model` flag exists
                # for: an infra or side-effecting failure stops the loop
                # instead of being fed back for another (potentially harmful
                # or futile) attempt.
                return f"Stopped — {result.message}"
        else:
            observation = result if isinstance(result, str) else json.dumps(result)

        if verbose:
            print(f"[step {step_num}] Observation: {observation}\n")

        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return f"Stopped: exceeded max_steps ({max_steps}) without reaching a final answer."


def main() -> None:
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key="not-needed")
    goal = "A store sells 17 units at $12.99 each. What's the total, and is it under $220?"
    print(f"\nFinal Answer: {run_agent(client, DEFAULT_MODEL, goal)}")


if __name__ == "__main__":
    main()
```

</details>

This matches [../03-small-agent/agent.py](../03-small-agent/agent.py)'s `run_agent()` (and everything above it) exactly — only the temporary `main()` at the bottom still differs from the final file, and Step 10 replaces it.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| Loop still never terminates on a persistently non-converging model | `for step_num in range(1, max_steps + 1)` changed back to `while True`, or `max_steps` hardcoded to an enormous number | Keep the bounded range — this is the entire point of the guardrail |
| Fallback message never appears even when it should | The final `return` placed *inside* the `for` loop instead of after it (so it fires every iteration, or never reaches unindented code) | It must be the function's last statement, reached only once the loop is exhausted |
| `[step N]` prefixes missing from some but not all trace lines | Only some of the three `print()` calls were updated | All three (`Thought`, `Action`/`Input`, `Observation`) should share the same `[step {step_num}]` prefix |

Next: **[CLI and Interactive Mode](11-cli-and-interactive-mode.md)** — wrap `run_agent()` in a real CLI with flags, and an interactive loop for one goal per line.

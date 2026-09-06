# Step 8 — The ReAct Loop

> [Back to index](README.md) · Previous: [The Dispatcher — Retry Policy](08-the-dispatcher-retry-policy.md) · Next: [The max_steps Guardrail](10-the-max-steps-guardrail.md)

## Goal

Assemble `parse_reply` and `dispatch_tool_call` into `run_agent()`: call the model, act on its decision, feed the observation back, and repeat — for now, with no limit on how many times the loop can run.

## Why this matters

This is the step where the pieces actually become a *loop* instead of a collection of separately-tested functions, and it's worth building it once without a step cap before adding one. Without the cap, the mechanics are easier to see clearly: call the model, parse what it said, either return (`Final Answer`, or an unparseable reply) or dispatch the action and append both the assistant's reply and the resulting observation to `messages`, then go around again. Every version of this app you've studied in this series shares that same shape at its core — this is the generalization of Step 2's single `run_turn()` call into something that can run itself as many times as the model decides.

Deliberately leaving out the guardrail here — building it in Step 9 instead — is the point: a model that never converges would make this version of `run_agent()` loop forever. That's not a defect to quietly work around while building; it's the exact failure mode the next step's guardrail exists to prevent, and it's worth understanding what's missing before the fix is added on top of it.

## 1. `run_agent()` — unbounded

Add this after `dispatch_tool_call()`:

```python
# --------------------------------------------------------------------------
# Controller — the ordinary code that runs the loop: call the model, parse
# its decision, dispatch it, feed back the result.
# --------------------------------------------------------------------------
def run_agent(client: OpenAI, model: str, goal: str, verbose: bool = True) -> str:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    tool_failure_counts: dict[str, int] = {}

    while True:
        reply = call_model(client, model, messages)
        parsed = parse_reply(reply)

        if verbose and parsed.thought:
            print(f"Thought: {parsed.thought}")

        if parsed.final_answer is not None:
            return parsed.final_answer

        if parsed.action is None:
            # Format wasn't followed at all — don't spin on an unparseable
            # turn, just surface what the model actually said.
            return reply

        if verbose:
            print(f"Action: {parsed.action}  Input: {parsed.action_input}")

        result = dispatch_tool_call(parsed.action, parsed.action_input, tool_failure_counts)
        messages.append({"role": "assistant", "content": reply})

        if isinstance(result, ToolError):
            tool_failure_counts[parsed.action] = tool_failure_counts.get(parsed.action, 0) + 1
            observation = f"Error: {result.message}"
            if not result.retryable_by_model:
                return f"Stopped — {result.message}"
        else:
            observation = result if isinstance(result, str) else json.dumps(result)

        if verbose:
            print(f"Observation: {observation}\n")

        messages.append({"role": "user", "content": f"Observation: {observation}"})
```

Walk through one full iteration in order:

1. **Call the model, parse the reply.** Same two calls every turn, regardless of how many turns have already happened.
2. **`Final Answer` present** → return it immediately. This is the loop's only "successful" exit.
3. **No `Action` parsed either** → return the raw reply. Same fallback `parse_reply` was built to signal in Step 5 — don't spin on a turn the model didn't format at all.
4. **Dispatch the action.** `tool_failure_counts` — empty on the first call, growing as failures accumulate — is threaded through exactly as Step 7's `dispatch_tool_call` expects.
5. **Append the assistant's own reply to `messages` before the observation.** This has to happen in this order: the conversation history must show the model's `Action` before it shows the result of that action, or the next `call_model()` call would be reasoning from a transcript that doesn't match what actually happened.
6. **A `ToolError` with `retryable_by_model=False`** → return a `"Stopped — ..."` message immediately, without appending an observation the model would never get to see. This is Step 7's side-effecting no-retry rule finally taking effect at the control-flow level: the loop itself ends, not just this one call.
7. **Every other outcome** — a successful result or a retryable error — becomes an `Observation` message, and the loop goes around again.

## 2. Scratch code: run one goal to completion

Replace the Step 4 scratch `main()` with:

```python
def main() -> None:
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key="not-needed")
    goal = "A store sells 17 units at $12.99 each. What's the total, and is it under $220?"
    print(f"\nFinal Answer: {run_agent(client, DEFAULT_MODEL, goal)}")


if __name__ == "__main__":
    main()
```

Still not the finished CLI — that's Step 10. This is scratch code to prove the whole multi-step round trip actually works for one hardcoded goal.

## Try it

```bash
uv run agent.py
```

Expected shape of the output (exact wording and step count depend on your model — don't expect it verbatim):

```text
Thought: I need to find the total cost first.
Action: calculator
Action Input: {"expression": "17 * 12.99"}
Observation: 220.83

Thought: 220.83 is over $220, so the answer is no.
Final Answer: The total comes to $220.83, which is not under $220.
```

Two full iterations of the loop happened here with no cap in sight — that's the mechanism working. If your model instead keeps calling `calculator` with slightly different expressions and never produces a `Final Answer`, let it run for a minute, then interrupt it with Ctrl-C. That's not a bug in this code — it's the unbounded loop doing exactly what unbounded means, and precisely the situation Step 9 exists to prevent.

## Checkpoint

<details>
<summary>Full <code>agent.py</code> (through <code>run_agent</code>, unbounded, temporary <code>main()</code>)</summary>

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


def run_agent(client: OpenAI, model: str, goal: str, verbose: bool = True) -> str:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    tool_failure_counts: dict[str, int] = {}

    while True:
        reply = call_model(client, model, messages)
        parsed = parse_reply(reply)

        if verbose and parsed.thought:
            print(f"Thought: {parsed.thought}")

        if parsed.final_answer is not None:
            return parsed.final_answer

        if parsed.action is None:
            # Format wasn't followed at all — don't spin on an unparseable
            # turn, just surface what the model actually said.
            return reply

        if verbose:
            print(f"Action: {parsed.action}  Input: {parsed.action_input}")

        result = dispatch_tool_call(parsed.action, parsed.action_input, tool_failure_counts)
        messages.append({"role": "assistant", "content": reply})

        if isinstance(result, ToolError):
            tool_failure_counts[parsed.action] = tool_failure_counts.get(parsed.action, 0) + 1
            observation = f"Error: {result.message}"
            if not result.retryable_by_model:
                return f"Stopped — {result.message}"
        else:
            observation = result if isinstance(result, str) else json.dumps(result)

        if verbose:
            print(f"Observation: {observation}\n")

        messages.append({"role": "user", "content": f"Observation: {observation}"})


def main() -> None:
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key="not-needed")
    goal = "A store sells 17 units at $12.99 each. What's the total, and is it under $220?"
    print(f"\nFinal Answer: {run_agent(client, DEFAULT_MODEL, goal)}")


if __name__ == "__main__":
    main()
```

</details>

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| Observation appended to `messages` before the assistant's own `Action` reply | Lines reordered by accident | The assistant's raw reply must be appended first — the transcript has to reflect what actually happened, in order |
| Loop runs forever on a model that never converges | Expected at this step — there is no cap yet | Interrupt with Ctrl-C; Step 9 adds the fix, not this step |
| `NameError: json is not defined` inside `run_agent` | `import json` missing or removed while editing imports in an earlier step | Confirm the imports block matches the checkpoint |

Next: **[The max_steps Guardrail](10-the-max-steps-guardrail.md)** — cap the loop so a model that never converges can't hang the app.

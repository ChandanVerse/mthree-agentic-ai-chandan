# Step 10 — CLI and Interactive Mode

> [Back to index](README.md) · Previous: [The max_steps Guardrail](10-the-max-steps-guardrail.md) · Next: [Recap and Exercises](12-recap-and-exercises.md)

## Goal

Replace the scratch `main()` with the real CLI: `argparse` flags for the model, endpoint, step cap, and a `--quiet` trace toggle, plus both a one-shot mode (`agent.py "<goal>"`) and an interactive mode (one fresh goal per line).

## Why this matters

Most of this is the same pattern as every other app in this series — `--model`/`--base-url` flags falling back to `DMR_MODEL`/`DMR_BASE_URL` env vars, `APIError` handled around the call that can raise it, Ctrl-D/Ctrl-C/`exit`/`quit` all treated as a clean exit. What's worth calling out is what's deliberately *different* here from the chat apps earlier in the series: this app has **no memory across goals**. Each CLI invocation, and each line typed in interactive mode, starts a brand-new `messages` list inside `run_agent()` — there's no persistent conversation the way [01-basic-chat-app](../01-basic-chat-app-walkthrough/README.md)'s REPL keeps one. That's a real design choice, not an oversight: a ReAct agent finishing one bounded goal and starting fresh for the next is a different shape than an open-ended chat, and mixing the two would mean an old goal's `Thought`/`Action`/`Observation` history bleeding into a new, unrelated task's reasoning.

## 1. `main()`

Replace the temporary `main()` from Step 9 with:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="A small ReAct agent with calculator, web_search, and save_note tools.")
    parser.add_argument("goal", nargs="?", help="The task to hand the agent. Omit for interactive mode.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help=f"Step-limit guardrail (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument("--quiet", action="store_true", help="Only print the final answer, not the Thought/Action/Observation trace")
    args = parser.parse_args()

    # Docker Model Runner doesn't check the API key, but the client requires one.
    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    def run_one(goal: str) -> None:
        try:
            answer = run_agent(client, args.model, goal, max_steps=args.max_steps, verbose=not args.quiet)
        except APIError as e:
            print(f"[error] {e}. Is Docker Model Runner running? See ../00-local-model-setup/README.md")
            sys.exit(1)
        print(f"\nFinal Answer: {answer}")

    if args.goal:
        run_one(args.goal)
        return

    print(f"Small agent ready ({args.model}). Each line is a fresh goal — no memory carries over between them. Type 'exit' to quit.\n")
    while True:
        try:
            goal = input("Goal: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if goal.lower() in {"exit", "quit"}:
            break
        if not goal:
            continue
        run_one(goal)
        print()


if __name__ == "__main__":
    main()
```

Add the two missing imports at the top of the file, and update the `openai` import to include `APIError`:

```python
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from difflib import get_close_matches

from openai import APIError, OpenAI
```

(`argparse` and `sys` are the only genuinely new imports; `APIError` joins the existing `openai` import line.)

Three things worth pointing out even though most of the shape is familiar from earlier apps in the series:

- `goal` is a **positional, optional** argument (`nargs="?"`) — supplying it on the command line runs one-shot mode; omitting it falls through to the interactive loop below. That single `if args.goal:` branch is the entire fork between the two modes.
- `run_one()` is a closure defined inside `main()`, capturing `client` and `args` from the enclosing scope, so both the one-shot path and every line of the interactive loop share identical error handling instead of duplicating the `try`/`except APIError` block twice.
- The interactive loop's banner says outright, to the person typing, that "each line is a fresh goal — no memory carries over" — surfacing the no-memory design decision as an explicit part of the UI, not something a trainee has to discover by noticing the agent forgot something.

## Try it

One-shot mode, reproducing the trap from this app's own reference README — computing tax on the *discounted* amount rather than the original subtotal:

```bash
uv run agent.py "A store sells 17 units at \$12.99 each with 8% discount and 6% tax on the discounted amount. Is the total under \$216?"
```

Expected shape of the output (exact step count and wording depend on your model):

```text
[step 1] Thought: I need the subtotal first.
[step 1] Action: calculator  Input: {'expression': '17 * 12.99'}
[step 1] Observation: 220.83

[step 2] Thought: Now apply the 8% discount.
[step 2] Action: calculator  Input: {'expression': '220.83 * 0.92'}
[step 2] Observation: 203.1636

[step 3] Thought: Tax is 6% of the DISCOUNTED amount, not the original subtotal.
[step 3] Action: calculator  Input: {'expression': '203.1636 * 1.06'}
[step 3] Observation: 215.35

[step 4] Thought: 215.35 is under 216, so the answer is yes.
Final Answer: Yes — the total comes to $215.35, which is under the $216 budget.
```

Whether your model tag gets the discount-before-tax ordering right on the first pass, or computes tax on the original subtotal instead, is worth watching for — it's the same trap this series' notes on Reflexion address as something a self-check step could catch.

Then try `--quiet` (only the final line prints) and interactive mode:

```bash
uv run agent.py --quiet "What's 2^10?"
uv run agent.py
```

```text
Small agent ready (docker.io/ai/gemma4:E4B). Each line is a fresh goal — no memory carries over between them. Type 'exit' to quit.

Goal: What's 12 * 7?
[step 1] Thought: I need to compute 12 * 7.
[step 1] Action: calculator  Input: {'expression': '12 * 7'}
[step 1] Observation: 84

Final Answer: 84.

Goal: exit
```

## Checkpoint

<details>
<summary>Full <code>agent.py</code></summary>

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
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from difflib import get_close_matches

from openai import APIError, OpenAI
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
    parser = argparse.ArgumentParser(description="A small ReAct agent with calculator, web_search, and save_note tools.")
    parser.add_argument("goal", nargs="?", help="The task to hand the agent. Omit for interactive mode.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help=f"Step-limit guardrail (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument("--quiet", action="store_true", help="Only print the final answer, not the Thought/Action/Observation trace")
    args = parser.parse_args()

    # Docker Model Runner doesn't check the API key, but the client requires one.
    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    def run_one(goal: str) -> None:
        try:
            answer = run_agent(client, args.model, goal, max_steps=args.max_steps, verbose=not args.quiet)
        except APIError as e:
            print(f"[error] {e}. Is Docker Model Runner running? See ../00-local-model-setup/README.md")
            sys.exit(1)
        print(f"\nFinal Answer: {answer}")

    if args.goal:
        run_one(args.goal)
        return

    print(f"Small agent ready ({args.model}). Each line is a fresh goal — no memory carries over between them. Type 'exit' to quit.\n")
    while True:
        try:
            goal = input("Goal: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if goal.lower() in {"exit", "quit"}:
            break
        if not goal:
            continue
        run_one(goal)
        print()


if __name__ == "__main__":
    main()
```

</details>

This matches [../03-small-agent/agent.py](../03-small-agent/agent.py) exactly.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `[error] Could not reach model` immediately on start | Docker Model Runner not running / model not pulled | See [00-local-model-setup](../00-local-model-setup/README.md) |
| Interactive mode seems to remember a previous goal's tool results | `messages` accidentally hoisted outside `run_one`/`run_agent` and reused across calls | Each `run_agent()` call must build its own fresh `messages` list — that's the entire no-memory design |
| `--quiet` still prints the `[step N]` trace | `verbose=not args.quiet` not wired through, or `run_agent`'s own `verbose` parameter default overriding it | Confirm `run_one` passes `verbose=not args.quiet` explicitly |

Next: **[Recap and Exercises](12-recap-and-exercises.md)**.

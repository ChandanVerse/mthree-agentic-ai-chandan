# Step 4 — The Profile and Reply Format

> [Back to index](README.md) · Previous: [Web Search, Save Note, and the Registry](04-web-search-save-note-and-the-registry.md) · Next: [Parsing the Reply](06-parsing-the-reply.md)

## Goal

Write `SYSTEM_PROMPT` — the profile that teaches the model the `Thought`/`Action`/`Action Input`/`Final Answer` format, including a worked example — and look at a real, raw, multi-block reply before writing any parsing code.

## Why this matters

Step 2's chat app taught the model one line: `ACTION: web_search {...}`. This app needs the model to reliably produce a whole small block of structured text, every turn, for as many turns as the task needs. That's a harder ask of a small local model, and a *description* of the format ("respond with a Thought, then an Action...") isn't enough on its own — small models are much better at pattern-matching a concrete example than at following an abstract specification. That's why `SYSTEM_PROMPT` below includes a full worked example of one `Thought`/`Action`/`Action Input` block, using the exact tool names and JSON syntax the model is expected to reproduce.

As in Step 2, before writing a single regex, confirm with your own eyes that the model actually produces this format for a question that needs a tool. If it doesn't, no amount of downstream parsing logic fixes that — you'd need a different model tag, or the fallback and guardrail behavior built in later steps to contain it gracefully.

Create the file:

```bash
touch agent.py
```

## 1. Module docstring, imports, and configuration

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
import os

from openai import OpenAI

from tools import tools_prompt_block

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")
DEFAULT_MAX_STEPS = 6  # a step cap is a guardrail, not decoration
```

Same `DEFAULT_MODEL`/`DEFAULT_BASE_URL` environment-variable pattern as every other app in this series. `DEFAULT_MAX_STEPS` is declared now even though nothing reads it until Step 9 — it's a constant that belongs next to the other configuration, not buried inside the loop that will eventually use it.

## 2. The system prompt, with a worked example

```python
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
```

Four things worth calling out to trainees:

- `{tools_prompt_block()}` embeds Step 3's registry rendering directly, same as Step 2 — the model's view of what tools exist and the code's view (`TOOLS`) share one source of truth.
- The `Example:` block is the concrete instance the model imitates. It's not decorative — cut it and watch how much less reliably a small model reproduces the exact `Action Input: {"key": "value"}` syntax.
- The rules list encodes two of this app's guardrails directly into the prompt, in plain language, alongside the structural enforcement built in later steps: never compute arithmetic mentally (so the calculator tool is actually exercised), and only call `save_note` when explicitly asked (a first line of defense against an unwanted side effect, on top of the classification from Step 3).
- Note the doubled braces `{{"key": "value"}}` — this is an f-string, so a literal `{` or `}` in the output text has to be escaped as `{{`/`}}`.

## 3. A minimal, non-streaming model call

```python
def call_model(client: OpenAI, model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content
```

Same non-streaming shape as Step 2's `call_model()`, for the same reason: the full reply has to be inspected — for a `Final Answer`, an `Action`, or neither — before deciding what to do with it. You can't act on a decision that's still streaming in.

## 4. Scratch code: inspect one raw multi-block decision

```python
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

Throwaway code, replaced in Step 8 once `run_agent()` exists.

## Try it

```bash
uv run agent.py
```

Expected shape of the output (your model's exact wording and numbers may differ — what matters is the three labeled lines):

```text
Thought: I need to compute 17 * 12.99 before I can answer.
Action: calculator
Action Input: {"expression": "17 * 12.99"}
```

Point out to trainees exactly what's *not* here yet: no `Observation`, no `Final Answer` — this is only the model's first move, one `Thought`/`Action`/`Action Input` block, produced from a single `call_model()` call with no loop around it at all. Turning this into a full conversation is everything from Step 6 onward.

## Checkpoint

<details>
<summary>Full <code>agent.py</code> (raw decision inspection — temporary <code>main()</code>)</summary>

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
import os

from openai import OpenAI

from tools import tools_prompt_block

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

`SYSTEM_PROMPT` and `call_model()` are final as written here — the temporary `main()` above is fully replaced in [Step 10](11-cli-and-interactive-mode.md).

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `SyntaxError` on the `SYSTEM_PROMPT` f-string | A literal `{` or `}` in the prompt text (e.g. in `Action Input: {"key": "value"}`) written as a single brace instead of doubled | Escape every literal brace as `{{`/`}}` inside an f-string |
| Model answers `84` directly with no `Thought`/`Action` at all | Some models over-comply with "give a Final Answer as soon as you have enough information" and skip straight to arithmetic despite the "never compute arithmetic yourself" rule | Expected occasionally with small local models — this is exactly the kind of format-noncompliance the parser (Step 5) and the guardrails (Steps 7-9) are built to tolerate |
| `openai.APIConnectionError` | Docker Model Runner isn't running, or the model wasn't pulled | See [00-local-model-setup](../00-local-model-setup/README.md) |

Next: **[Parsing the Reply](06-parsing-the-reply.md)** — turn this raw multi-block text into a structured `ParsedReply` your code can act on.

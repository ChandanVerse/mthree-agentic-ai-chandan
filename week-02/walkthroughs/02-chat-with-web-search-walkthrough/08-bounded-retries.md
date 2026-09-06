# Step 7 — Bounded Retries

> [Back to index](README.md) · Previous: [The Round Trip](07-the-round-trip.md) · Next: [Assembling the CLI and Loop](09-assembling-the-cli-and-loop.md)

## Goal

Replace Step 6's placeholder `return f"Tool call failed: {error}"` with a real self-correction loop: feed the exact error back to the model as a new message, let it try again, and cap the number of attempts.

## Why this matters

Silently giving up after one bad call wastes a recoverable situation — a model that produced `ACTION: web_serach {"query": "..."}` (a typo) or slightly malformed JSON is usually one nudge away from getting it right, if it's told *specifically* what was wrong. That's why `resolve_tool_call()` was built in Step 5 to return the exact validation or lookup message instead of a generic "that didn't work" — this step is where that specificity finally gets used.

The other half of this step is just as important: the retry has to be **bounded**. Without `MAX_TOOL_RETRIES`, a model that keeps producing the same malformed call would loop forever, since nothing here changes the model's own behavior between attempts — it's still the same model, the same weights, being asked again. A fixed cap plus a graceful fallback message is what turns "the model might get stuck" into "the app never hangs, worst case it just answers without the tool."

## 1. The retry constant

Add this after `SYSTEM_PROMPT`:

```python
# Give the model a bounded number of chances to self-correct a malformed
# call before we give up and answer without the tool.
MAX_TOOL_RETRIES = 2
```

## 2. Rewrite `run_turn()` with the loop

Replace the Step 6 version with:

```python
def run_turn(client: OpenAI, model: str, messages: list[dict]) -> str:
    """Handle one user turn: at most one tool hop, then a final answer."""
    for _ in range(MAX_TOOL_RETRIES + 1):
        reply = call_model(client, model, messages)
        tool_name, raw_args = try_parse_action(reply)

        if tool_name is None:
            return reply  # model chose to answer directly — no tool needed

        result, error = resolve_tool_call(tool_name, raw_args)
        if error:
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {"role": "user", "content": f"Tool call failed: {error}. Try again or answer without the tool."}
            )
            continue

        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Observation from {tool_name}: {json.dumps(result)}\n\n"
                    "Now answer the original question using this observation. Do not call another tool."
                ),
            }
        )
        return call_model(client, model, messages)

    return "I couldn't complete that tool call correctly after a few tries — could you rephrase?"
```

What changed from Step 6, and why:

- The whole body now lives inside `for _ in range(MAX_TOOL_RETRIES + 1)` — one initial attempt plus `MAX_TOOL_RETRIES` retries, so `MAX_TOOL_RETRIES = 2` means up to three attempts total.
- On error, the failed reply is appended as the `assistant` turn, then the *exact* error string is appended as a `user`-role message ("Tool call failed: {error}. Try again or answer without the tool."), and the loop `continue`s — the model sees precisely what it got wrong on its next attempt.
- The success path is unchanged from Step 6, except it now `return`s out of the loop, ending the retry cycle the moment a call actually works.
- If every attempt in the loop fails, the `for` exhausts and falls through to the final `return` — a fixed, honest message rather than an exception or an infinite loop.

Note also what's implicit here: `web_search` is read-only and idempotent, so retrying it blindly (running it again with corrected arguments) is safe. That wouldn't be true of a tool with side effects — sending an email, writing to a database — where blindly retrying could repeat an action that already partially succeeded. This app only has to make this call because its one tool happens to be safe to retry.

## Try it

Run the same scratch `main()` from Step 6 again — the success path should look identical. To see the retry path fire, temporarily feed a message list that already contains a broken tool call, bypassing the model:

```bash
uv run python -c "
from openai import OpenAI
from chat_with_tools import run_turn, DEFAULT_MODEL, DEFAULT_BASE_URL, SYSTEM_PROMPT

client = OpenAI(base_url=DEFAULT_BASE_URL, api_key='not-needed')
messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': 'Use the web_serach tool (note the typo) to look up the current weather in Tokyo.'},
]
print(run_turn(client, DEFAULT_MODEL, messages))
"
```

Whether this actually triggers the retry path depends on whether your model reproduces the typo verbatim — small models often self-correct the spelling on their own, in which case you'll just see a normal answer. Either outcome is worth showing trainees: the retry logic isn't dead code, but you can't always force it on demand with a live model.

## Checkpoint

<details>
<summary>Full <code>chat_with_tools.py</code> (through <code>run_turn</code>, temporary <code>main()</code>)</summary>

```python
#!/usr/bin/env python3
"""Step 2 — a chat app augmented with one tool (web search) the model may
choose to use.

Run:
    uv run chat_with_tools.py
"""
import json
import os
import re
from difflib import get_close_matches

from openai import APIError, OpenAI
from pydantic import ValidationError

from tools import TOOLS, tools_prompt_block

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")

# The text protocol a model must follow to call a tool: a line reading
# "ACTION: <tool_name> {json args}". Matching the tool name and the JSON
# blob as two separate steps (below) lets us tell "no tool call at all"
# apart from "tried to call a tool but the JSON is truncated/broken".
ACTION_NAME_PATTERN = re.compile(r"ACTION:\s*(\w+)", re.DOTALL)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = f"""You are a helpful assistant with access to one tool.

{tools_prompt_block()}

To call a tool, reply with ONLY this line (no other text):
ACTION: web_search {{"query": "your search query"}}

If you don't need the tool, just answer the user's question directly in plain text.
Only call the tool when you genuinely need current or external information —
not for things you already know.
"""

# Give the model a bounded number of chances to self-correct a malformed
# call before we give up and answer without the tool.
MAX_TOOL_RETRIES = 2


def call_model(client: OpenAI, model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def try_parse_action(text: str):
    """Extract a tool call from a model reply.

    Returns one of:
    - (None, None)               the model didn't try to call a tool
    - (tool_name, args_dict)     a syntactically valid call
    - (tool_name, raw_str)       the model *tried* to call a tool, but no
                                  parseable JSON object followed — covers
                                  both a truncated call (no closing brace)
                                  and one with broken JSON syntax — distinct
                                  from "no call" so the caller can react.
    """
    name_match = ACTION_NAME_PATTERN.search(text)
    if not name_match:
        return None, None

    tool_name = name_match.group(1)
    remainder = text[name_match.end() :]
    json_match = JSON_OBJECT_PATTERN.search(remainder)
    if not json_match:
        return tool_name, remainder.strip()

    try:
        return tool_name, json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return tool_name, json_match.group(0)


def resolve_tool_call(tool_name: str, raw_args) -> tuple[dict | None, str | None]:
    """Validate and execute one tool call.

    Returns (result, error_message) — exactly one is set. An error is
    handed back to the model as the *exact* validation/lookup problem,
    rather than crashing the app or silently guessing.
    """
    if tool_name not in TOOLS:
        # Suggest, never auto-substitute — the model must confirm a fuzzy
        # match itself rather than the dispatcher silently rerouting it.
        suggestion = get_close_matches(tool_name, TOOLS.keys(), n=1, cutoff=0.6)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        return None, f"Unknown tool '{tool_name}'. Available tools: {list(TOOLS)}.{hint}"

    spec = TOOLS[tool_name]
    if isinstance(raw_args, str):  # JSON failed to parse in try_parse_action
        return None, f"Arguments for '{tool_name}' were not valid JSON: {raw_args}"

    try:
        validated = spec["schema"].model_validate(raw_args)
    except ValidationError as e:
        return None, f"Invalid arguments for '{tool_name}': {e}"

    result = spec["execute"](**validated.model_dump())
    return result, None


def run_turn(client: OpenAI, model: str, messages: list[dict]) -> str:
    """Handle one user turn: at most one tool hop, then a final answer."""
    for _ in range(MAX_TOOL_RETRIES + 1):
        reply = call_model(client, model, messages)
        tool_name, raw_args = try_parse_action(reply)

        if tool_name is None:
            return reply  # model chose to answer directly — no tool needed

        result, error = resolve_tool_call(tool_name, raw_args)
        if error:
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {"role": "user", "content": f"Tool call failed: {error}. Try again or answer without the tool."}
            )
            continue

        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Observation from {tool_name}: {json.dumps(result)}\n\n"
                    "Now answer the original question using this observation. Do not call another tool."
                ),
            }
        )
        return call_model(client, model, messages)

    return "I couldn't complete that tool call correctly after a few tries — could you rephrase?"


def main() -> None:
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key="not-needed")
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What's the latest stable version of Python?"},
    ]
    answer = run_turn(client, DEFAULT_MODEL, messages)
    print(f"Assistant: {answer}")


if __name__ == "__main__":
    main()
```

</details>

This matches [../02-chat-with-web-search/chat_with_tools.py](../02-chat-with-web-search/chat_with_tools.py)'s `run_turn()` (and everything above it) exactly — only the temporary `main()` at the bottom still differs from the final file, and Step 8 replaces it.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| Loop never terminates on a persistently broken tool call | `range(MAX_TOOL_RETRIES + 1)` changed to `while True` | Keep the bounded range — this is the entire point of the guardrail |
| Error message fed back to the model is generic ("something went wrong") instead of specific | Not using `error` from `resolve_tool_call()` directly in the fed-back message | Always forward the exact string `resolve_tool_call` returned |
| App raises instead of returning the fallback message after repeated failures | Fallback `return` placed inside the `for` loop instead of after it | It must be the function's last statement, reached only once the loop is exhausted |

Next: **[Assembling the CLI and Loop](09-assembling-the-cli-and-loop.md)** — wrap `run_turn()` in the same REPL loop and CLI-flags pattern from the rest of the series.

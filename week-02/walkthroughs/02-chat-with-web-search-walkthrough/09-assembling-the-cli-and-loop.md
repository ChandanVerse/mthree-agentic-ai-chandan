# Step 8 — Assembling the CLI and Loop

> [Back to index](README.md) · Previous: [Bounded Retries](08-bounded-retries.md) · Next: [Recap and Exercises](10-recap-and-exercises.md)

## Goal

Replace the scratch `main()` with the real interactive REPL loop: `argparse` flags for model/endpoint, a persistent `messages` list across turns, and graceful handling of a connection failure.

## Why this matters

None of this is new material if you've completed [01-basic-chat-app-walkthrough](../01-basic-chat-app-walkthrough/README.md) — the exit conditions (Ctrl-D, Ctrl-C, `exit`/`quit`, empty line), the `--model`/`--base-url` flags falling back to `DMR_MODEL`/`DMR_BASE_URL` env vars, and the `APIError` handling are the identical pattern taught there. What's worth calling out is what's *missing* compared to that app: this loop doesn't stream (Step 3 already covered why — the full reply has to be inspected before deciding whether it's a tool call), and it doesn't call a `trim_history()`-style bounded-window function, so `messages` grows for the entire session. That's a real simplification, not an oversight — extending it to trim history is one of this walkthrough's exercises.

## 1. `main()`

Replace the temporary `main()` from Step 7 with:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Chat app with an optional web-search tool.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

    # Docker Model Runner doesn't check the API key, but the client requires one.
    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Chatting with {args.model} (web search enabled) — type 'exit' to leave.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            answer = run_turn(client, args.model, messages)
        except APIError as e:
            print(f"[error] {e}. Is Docker Model Runner running? See ../00-local-model-setup/README.md")
            sys.exit(1)

        print(f"Assistant: {answer}\n")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
```

Add the two missing imports and standard entry-point guard at the top of the file:

```python
import argparse
import json
import os
import re
import sys
from difflib import get_close_matches
```

(`argparse` and `sys` are the only genuinely new imports here — everything else was already added in earlier steps.)

Two things worth pointing out even though the shape is familiar from Step 1's series:

- `messages.append({"role": "user", ...})` happens once per turn, here in `main()` — `run_turn()` appends its *own* assistant/observation messages internally when a tool fires, but the user's own message and the final assistant answer are appended at this outer level. That split is why `run_turn()`'s signature takes and mutates the same `messages` list rather than returning a new one.
- The `try`/`except APIError` sits around `run_turn(...)`, not around the whole loop — a connection failure inside any of `run_turn`'s calls to `call_model()` surfaces here and exits the app, rather than being silently retried as if it were a malformed tool call (those are a different failure mode, already handled inside `run_turn` itself).

## Try it

```bash
uv run chat_with_tools.py
```

```text
Chatting with docker.io/ai/gemma4:E4B (web search enabled) — type 'exit' to leave.

You: What's 12 * 7?
Assistant: 84.
You: What's the latest stable version of Python?
Assistant: Python 3.13 is the latest stable release as of now — I confirmed this via a web search.
You: exit
Bye!
```

The first question needs no external information, so the model answers directly. The second triggers `ACTION: web_search {"query": "latest stable Python version"}` behind the scenes — exactly the raw text you saw printed directly in Step 3, now fully wired into a real conversation.

## Checkpoint

<details>
<summary>Full <code>chat_with_tools.py</code></summary>

```python
#!/usr/bin/env python3
"""Step 2 — a chat app augmented with one tool (web search) the model may choose to use.

Run:
    uv run chat_with_tools.py
"""
import argparse
import json
import os
import re
import sys
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
    parser = argparse.ArgumentParser(description="Chat app with an optional web-search tool.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

    # Docker Model Runner doesn't check the API key, but the client requires one.
    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Chatting with {args.model} (web search enabled) — type 'exit' to leave.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            answer = run_turn(client, args.model, messages)
        except APIError as e:
            print(f"[error] {e}. Is Docker Model Runner running? See ../00-local-model-setup/README.md")
            sys.exit(1)

        print(f"Assistant: {answer}\n")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
```

</details>

This matches [../02-chat-with-web-search/chat_with_tools.py](../02-chat-with-web-search/chat_with_tools.py) exactly.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `[error] Could not reach model` immediately on start | Docker Model Runner not running / model not pulled | See [00-local-model-setup](../00-local-model-setup/README.md) |
| Model prints the `ACTION:` line to the user instead of "just" calling it | The model didn't follow the protocol exactly | Expected occasionally with small local models — the retry loop or a stricter system prompt usually resolves it |
| Conversation seems to "forget" tool results were ever used, or grows very slow after many turns | `messages` has no bounded-window trimming here (unlike `chat.py` in Step 1's app) | Known simplification for this step — see the exercises in the recap for adding it back |

Next: **[Recap and Exercises](10-recap-and-exercises.md)**.

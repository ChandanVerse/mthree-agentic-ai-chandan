# Step 7 — CLI Flags and Error Handling

> [Back to index](README.md) · Previous: [Bounded Memory](07-bounded-memory.md) · Next: [Recap and Exercises](09-recap-and-exercises.md)

## Goal

Make the model and endpoint overridable from the command line — matching what we already did in `single_prompt.py` — and replace an unhandled crash with a friendly, actionable error message when the model is unreachable.

## Why this matters

Hardcoded `DEFAULT_MODEL`/`DEFAULT_BASE_URL` constants are fine for a single trainee on one machine, but the moment someone wants to try a different Gemma tag, or point at a teammate's machine, editing source code is the wrong way to do it. And right now, an unreachable model crashes with a raw `openai` stack trace — correct, but not helpful to someone who just needs to know "go check Docker Model Runner."

## 1. Add `argparse`

At the top of the file:

```python
import argparse
import sys

from openai import APIError, OpenAI
```

(Note the import changes from `OpenAI` alone to `APIError, OpenAI` — we need `APIError` for the `except` block below.)

At the top of `main()`, before creating the client:

```python
    parser = argparse.ArgumentParser(description="Basic chat app talking to a local LLM via Docker Model Runner.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()
```

This is the same pattern used in `single_prompt.py` — worth pointing out to trainees that they've already written this exact block once.

## 2. Replace every `DEFAULT_MODEL`/`DEFAULT_BASE_URL` usage with `args.*`

Three places change:

```python
    client = OpenAI(base_url=args.base_url, api_key="not-needed")
```

```python
    print(f"Chatting with {args.model} via {args.base_url} — type 'exit' or 'quit' to leave.\n")
```

```python
        stream = client.chat.completions.create(model=args.model, messages=messages, stream=True)
```

The `DEFAULT_MODEL`/`DEFAULT_BASE_URL` constants stay in the file — they're still the *defaults* the flags fall back to — but nothing in `main()` reads them directly anymore.

## 3. Handle connection failures gracefully

Wrap the streaming call in a `try`/`except`:

```python
        try:
            stream = client.chat.completions.create(model=args.model, messages=messages, stream=True)
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    print(token, end="", flush=True)
                    reply_chunks.append(token)
        except APIError as e:
            print(f"\n[error] Could not reach model '{args.model}': {e}")
            print("Is Docker Model Runner running and has the model been pulled? See ../00-local-model-setup/README.md")
            sys.exit(1)
```

Two design choices worth discussing with trainees:

- We catch `openai.APIError`, the base class for API-level failures, rather than a bare `except Exception`. The same reasoning as in `single_prompt.py`: a broken connection is a known, actionable failure with a specific fix, so it earns a specific message. Anything else should still surface normally.
- The error message doesn't just say *what* failed — it says *where to look* (`00-local-model-setup/README.md`). A good error message for a training environment tells the trainee their next action, not just the exception text.

## Try it

Run the fully-configured app:

```bash
uv run chat.py
```

Try the flags:

```bash
uv run chat.py --model docker.io/ai/gemma4:E4B --base-url http://localhost:12434/v1
```

Then break it deliberately to see the new error path:

```bash
uv run chat.py --base-url http://localhost:9999/v1
```

You should see a clean two-line `[error]` message and a clean process exit — no raw traceback.

## Checkpoint

<details>
<summary>Full <code>chat.py</code> (final)</summary>

```python
#!/usr/bin/env python3
"""Step 1 — a basic chat app: one LLM call per turn, no tools, no loop.

Run:
    uv run chat.py
"""
import argparse
import os
import sys

from openai import APIError, OpenAI

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer directly; "
    "ask a clarifying question only if the request is genuinely ambiguous."
)

# An ever-growing transcript eventually crowds out the system prompt or
# exceeds the context window. Keep a fixed-size window instead of resending
# the entire history forever.
MAX_TURNS_KEPT = 12


def trim_history(messages: list[dict]) -> list[dict]:
    """Keep the system prompt plus only the most recent MAX_TURNS_KEPT turns.

    `messages[0]` is always the system prompt; everything after it alternates
    user/assistant, so slicing the last `MAX_TURNS_KEPT * 2` entries keeps
    that many complete turns.
    """
    system, *turns = messages
    windowed = turns[-MAX_TURNS_KEPT * 2 :]
    return [system, *windowed]


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic chat app talking to a local LLM via Docker Model Runner.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

    # Docker Model Runner doesn't check the API key, but the client requires one.
    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Chatting with {args.model} via {args.base_url} — type 'exit' or 'quit' to leave.\n")

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
        messages = trim_history(messages)

        print("Assistant: ", end="", flush=True)
        reply_chunks: list[str] = []
        try:
            stream = client.chat.completions.create(model=args.model, messages=messages, stream=True)
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    print(token, end="", flush=True)
                    reply_chunks.append(token)
        except APIError as e:
            print(f"\n[error] Could not reach model '{args.model}': {e}")
            print("Is Docker Model Runner running and has the model been pulled? See ../00-local-model-setup/README.md")
            sys.exit(1)

        print()  # newline after the streamed reply
        messages.append({"role": "assistant", "content": "".join(reply_chunks)})


if __name__ == "__main__":
    main()
```

</details>

This matches [../01-basic-chat-app/chat.py](../01-basic-chat-app/chat.py) exactly. If yours differs, diff the two files and reconcile before moving on.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `NameError: name 'args' is not defined` outside `main()` | Tried to use `args.model` before `args = parser.parse_args()` ran | Confirm parsing happens first in `main()` |
| Error message prints but the program doesn't exit | Missing `sys.exit(1)`, or it's outside the `except` block's indentation | Check the `except APIError` block ends with `sys.exit(1)` at the right indent |
| Flags don't seem to take effect | Old `DEFAULT_MODEL`/`DEFAULT_BASE_URL` reference left somewhere in `main()` | Search the file for any remaining `DEFAULT_MODEL`/`DEFAULT_BASE_URL` use inside `main()` and swap to `args.*` |

Next: **[Recap and Exercises](09-recap-and-exercises.md)**.

# Step 2 — The `single_prompt.py` Script

> [Back to index](README.md) · Previous: [Environment Setup](02-environment-setup.md) · Next: [Chat Loop Skeleton](04-chat-loop-skeleton.md)

## Goal

Before building a whole loop, prove the simplest possible thing works: one prompt in, one reply out, program exits. If this doesn't work, nothing later will — so we isolate it first.

## Why this matters

This is the smallest unit of "talking to an LLM": a single HTTP request with one user message, no history, no loop. Every other app in this series is this same request, just wrapped in progressively more machinery. Get comfortable with it in isolation before adding anything else.

Create the file:

```bash
touch single_prompt.py
```

## 1. Parse the prompt from the command line

Start with just argument handling — no API call yet.

```python
#!/usr/bin/env python3
"""Single-shot LLM call: one prompt in, one reply out. No loop, no memory.

Run:
    uv run single_prompt.py "What is an agent?"
"""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one prompt to a local LLM and print the reply.")
    parser.add_argument("prompt", nargs="?", help="Prompt text. If omitted, you'll be asked for it.")
    args = parser.parse_args()

    prompt = args.prompt or input("Prompt: ").strip()
    if not prompt:
        sys.exit("No prompt given.")

    print(f"(stub) would send prompt: {prompt!r}")


if __name__ == "__main__":
    main()
```

`nargs="?"` makes the positional argument optional — if it's missing, we fall back to an interactive `input()` prompt rather than erroring immediately. Try both paths:

```bash
uv run single_prompt.py "hello"
uv run single_prompt.py
```

## 2. Add configuration for the model and endpoint

We don't want to hardcode the model tag or URL — both should be overridable without touching code, since different trainees (or later apps) may point at different models.

```python
import os

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")
```

Add the corresponding flags to the parser, right after the `prompt` argument:

```python
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL})")
```

Reading defaults from environment variables first, then letting flags override *those*, gives three ways to configure the same value — env var for "set it once for my machine," flag for "override just this run," hardcoded default for "just work out of the box." That layering shows up throughout the series.

## 3. Replace the stub with a real API call

Remove the `print("(stub) ...")` line and replace it with an actual client and request:

```python
from openai import APIConnectionError, OpenAI
```

```python
    # Docker Model Runner doesn't check the API key, but the client requires one.
    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
        )
    except APIConnectionError as e:
        sys.exit(f"[error] Could not reach '{args.base_url}': {e}")

    print(response.choices[0].message.content)
```

Two things worth pointing out to trainees here:

- `messages` is a **list of dicts**, even for a single message. Every LLM SDK uses this shape — get used to it now, because `chat.py` will build on exactly this structure.
- We catch `APIConnectionError` specifically, not a bare `except Exception`. A connection failure ("Docker Model Runner isn't running") is a predictable, actionable failure mode we can give a helpful message for. Anything else — a genuine bug — should surface with its real traceback, not get swallowed.

## Try it

```bash
uv run single_prompt.py "What is an agent, in one sentence?"
```

Expected output: a one- or two-line answer from the model, then the program exits. Then break it deliberately to see the error path — stop Docker Model Runner (or point `--base-url` at a wrong port) and rerun:

```bash
uv run single_prompt.py "test" --base-url http://localhost:9999/v1
```

You should get a clean `[error] Could not reach ...` message, not a raw traceback.

## Checkpoint

<details>
<summary>Full <code>single_prompt.py</code></summary>

```python
#!/usr/bin/env python3
"""Single-shot LLM call: one prompt in, one reply out. No loop, no memory.

Run:
    uv run single_prompt.py "What is an agent?"
"""
import argparse
import os
import sys

from openai import APIConnectionError, OpenAI

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one prompt to a local LLM and print the reply.")
    parser.add_argument("prompt", nargs="?", help="Prompt text. If omitted, you'll be asked for it.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

    prompt = args.prompt or input("Prompt: ").strip()
    if not prompt:
        sys.exit("No prompt given.")

    # Docker Model Runner doesn't check the API key, but the client requires one.
    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
        )
    except APIConnectionError as e:
        sys.exit(f"[error] Could not reach '{args.base_url}': {e}")

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
```

</details>

This matches [../01-basic-chat-app/single_prompt.py](../01-basic-chat-app/single_prompt.py) exactly.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'openai'` | Ran with `python` instead of `uv run`, or forgot `uv add openai` | Use `uv run single_prompt.py ...`; confirm `openai` is in `pyproject.toml` |
| Hangs with no output | Model still loading on first call | Normal — wait, it completes once loaded |
| `[error] Could not reach ...` even though Docker Model Runner is running | Wrong port, or route is `/engines/v1` on your Docker Desktop version | Try `--base-url http://localhost:12434/engines/v1` |

Next: **[Chat Loop Skeleton](04-chat-loop-skeleton.md)** — now that we know the connection works, let's build something interactive.

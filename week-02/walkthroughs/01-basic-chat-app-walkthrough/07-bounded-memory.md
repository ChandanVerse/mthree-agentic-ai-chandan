# Step 6 — Bounded Memory

> [Back to index](README.md) · Previous: [Streaming Responses](06-streaming-responses.md) · Next: [CLI Flags and Error Handling](08-cli-flags-and-error-handling.md)

## Goal

Stop `messages` from growing without limit, by keeping only the most recent N turns plus the system prompt.

## Why this matters

Right now, every turn appends two entries to `messages` and never removes any. In a long-running conversation, that list eventually causes one of two failures: it crowds out the system prompt's influence relative to everything else in the context, or it exceeds the model's context window outright and the request fails. This is a real, common gotcha in agent and chatbot systems generally, not a toy problem specific to this app — unbounded short-term memory silently breaks things. A fixed-size window is the simplest fix, and it's the right default even for production systems that later add smarter summarization on top.

## 1. Add the constant and the trim function

Near the top of the file, after `SYSTEM_PROMPT`:

```python
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
```

Walk through the slicing logic on a whiteboard or in the REPL — it trips people up on first read:

- `system, *turns = messages` unpacks the list into the first element (`messages[0]`, always the system prompt — this is the invariant from Step 4) and everything after it as a list called `turns`.
- Because every turn adds exactly one `user` entry and one `assistant` entry, `turns` alternates `[user, assistant, user, assistant, ...]`. One "turn" is 2 list entries.
- `turns[-MAX_TURNS_KEPT * 2 :]` takes the last `MAX_TURNS_KEPT * 2` entries — i.e., the last `MAX_TURNS_KEPT` complete turns. Slicing beyond the start of a list in Python is safe (it just returns everything available), so this works fine even in short conversations.
- Rebuild the list with the system prompt back in front: `[system, *windowed]`.

## 2. Call it after appending the user message

```python
        messages.append({"role": "user", "content": user_input})
        messages = trim_history(messages)
```

Trim right after adding the new user message, before the API call — that way the request itself never exceeds the window, not just the value stored for next time.

## Try it

The default window (12 turns) is too generous to demonstrate quickly. For the demo, temporarily lower it:

```python
MAX_TURNS_KEPT = 2
```

Then run a conversation that plants a fact and asks about it five turns later:

```text
You: Remember this number: 42.
Assistant: Got it — 42.
You: What's 2+2?
Assistant: 4.
You: What's 3+3?
Assistant: 6.
You: What's 10+10?
Assistant: 20.
You: What number did I ask you to remember?
Assistant: I'm not sure — could you remind me?
```

With `MAX_TURNS_KEPT = 2`, the "remember 42" turn has scrolled out of the window by the time the question comes back around. This is the trimming working as designed, not a bug — it's a good moment to discuss the tradeoff explicitly with trainees: window size trades recall depth against context-window safety, and a real system might use summarization instead of a hard cutoff. Set `MAX_TURNS_KEPT` back to `12` before moving on.

## Checkpoint

<details>
<summary>Full <code>chat.py</code> (bounded memory stage)</summary>

```python
#!/usr/bin/env python3
"""Step 1 — a basic chat app: one LLM call per turn, no tools, no loop.

Run:
    uv run chat.py
"""
import os

from openai import OpenAI

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
    # Docker Model Runner doesn't check the API key, but the client requires one.
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key="not-needed")
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Chatting with {DEFAULT_MODEL} via {DEFAULT_BASE_URL} — type 'exit' or 'quit' to leave.\n")

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
        stream = client.chat.completions.create(model=DEFAULT_MODEL, messages=messages, stream=True)
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                print(token, end="", flush=True)
                reply_chunks.append(token)

        print()  # newline after the streamed reply
        messages.append({"role": "assistant", "content": "".join(reply_chunks)})


if __name__ == "__main__":
    main()
```

</details>

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ValueError: not enough values to unpack` on `trim_history` | Called it before `messages` has the system prompt in it | Confirm `messages` is initialized with the system prompt before the loop starts |
| Trimming seems to do nothing | `MAX_TURNS_KEPT` set too high to notice in a short demo | Temporarily lower it, as shown above, then restore it |
| Assistant message missing after trimming | Trimmed *before* appending the assistant's reply instead of after | Trim only once, right after the user message is appended |

Next: **[CLI Flags and Error Handling](08-cli-flags-and-error-handling.md)** — make the model and endpoint configurable, and fail gracefully when the connection drops.

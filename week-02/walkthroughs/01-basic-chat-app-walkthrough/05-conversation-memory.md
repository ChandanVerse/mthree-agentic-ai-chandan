# Step 4 — Conversation Memory

> [Back to index](README.md) · Previous: [Chat Loop Skeleton](04-chat-loop-skeleton.md) · Next: [Streaming Responses](06-streaming-responses.md)

## Goal

Replace the fake reply with a real model call, and give the loop actual state: a running list of messages so the model has context from earlier in the conversation.

## Why this matters

Docker Model Runner's API — like every LLM HTTP API — is **stateless**. The server remembers nothing between requests. If you want the model to "remember" what was said three turns ago, *you* have to resend the entire conversation, every single time. This is the single most important thing to understand about how LLM memory actually works, and it's worth saying explicitly to trainees: there is no session on the server side. The "memory" lives entirely in a Python list on your machine.

## 1. Set up the client and constants

At the top of the file, add:

```python
import os

from openai import OpenAI

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer directly; "
    "ask a clarifying question only if the request is genuinely ambiguous."
)
```

The `SYSTEM_PROMPT` sets behavior once, up front — but note it's just another entry in the same `messages` list structure we used in `single_prompt.py`, with `role: "system"` instead of `role: "user"`.

## 2. Create the client and the message list

At the top of `main()`, before the `print(...)` banner:

```python
    # Docker Model Runner doesn't check the API key, but the client requires one.
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key="not-needed")
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
```

`messages[0]` is always the system prompt from this point forward — every later step in this walkthrough relies on that invariant, so call it out explicitly.

## 3. Replace the fake reply with a real call

Replace this line:

```python
        print(f"Assistant: (pretend reply to: {user_input!r})")
```

with:

```python
        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(model=DEFAULT_MODEL, messages=messages)
        reply = response.choices[0].message.content
        print(f"Assistant: {reply}")

        messages.append({"role": "assistant", "content": reply})
```

Three appends and one call — walk through the order carefully:

1. Append the user's message to `messages` **before** calling the API, so it's included in this request.
2. Send the *whole list* — system prompt, every prior turn, and the new user message.
3. Append the assistant's reply **after** the call, so the *next* turn includes it too.

If a trainee asks "why not just send the latest message?" — that's exactly what `single_prompt.py` does, and exactly why it has no memory. The difference between the two scripts is this list and nothing else.

## Try it

```bash
uv run chat.py
```

Have a short multi-turn exchange that requires memory to answer correctly, for example:

```text
You: My name is Priya.
Assistant: Nice to meet you, Priya!
You: What's my name?
Assistant: Your name is Priya.
```

If the second answer is right, memory is working — the model didn't "remember" anything itself, your `messages` list did the work.

## Checkpoint

<details>
<summary>Full <code>chat.py</code> (memory stage)</summary>

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

        response = client.chat.completions.create(model=DEFAULT_MODEL, messages=messages)
        reply = response.choices[0].message.content
        print(f"Assistant: {reply}")

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
```

</details>

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| Model "forgets" the very first message immediately | Appending user message *after* the API call instead of before | Append user message first, always |
| `AttributeError` on `response.choices[0].message.content` | Copy-paste error, or the request failed silently | Print `response` to inspect the raw object while debugging |
| Replies feel like they ignore the system prompt | `SYSTEM_PROMPT` message missing from `messages[0]`, or accidentally overwritten | Confirm `messages` is initialized once, before the loop, not inside it |

Next: **[Streaming Responses](06-streaming-responses.md)** — right now the app sits silently until the whole reply is ready. Let's fix that.

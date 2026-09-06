# Step 5 — Streaming Responses

> [Back to index](README.md) · Previous: [Conversation Memory](05-conversation-memory.md) · Next: [Bounded Memory](07-bounded-memory.md)

## Goal

Make the assistant's reply appear token by token as it's generated, instead of the terminal sitting silently until the entire response is ready.

## Why this matters

This is purely a UX change — streaming doesn't change what the model produces, only how quickly you see it appear. It matters more here than it might against a hosted API, because a local model on a laptop is noticeably slower; without streaming, a multi-second pause with zero feedback reads as "is this frozen?" With streaming, the same wait reads as "it's thinking, and typing."

## 1. Turn on `stream=True`

Replace the single-call block:

```python
        response = client.chat.completions.create(model=DEFAULT_MODEL, messages=messages)
        reply = response.choices[0].message.content
        print(f"Assistant: {reply}")

        messages.append({"role": "assistant", "content": reply})
```

with:

```python
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
```

Point out the shape change to trainees: with `stream=True`, the client returns an **iterator** of small chunk objects instead of one response object. Each chunk carries a `delta` — an incremental piece of the message — rather than the full `message` you got before. `chunk.choices[0].delta.content` can be `None` (for example, on the very first or last chunk), which is why we check `if token:` before printing or collecting it.

`end="", flush=True"` on `print` matters twice here: `end=""` stops each token from starting a new line, and `flush=True` forces the token to appear immediately instead of sitting in an output buffer — without it, tokens can appear in bursts rather than smoothly.

We still collect every token into `reply_chunks` and join them at the end, because `messages` needs the **complete** reply as a single string for the next turn — history doesn't care how the text arrived, only what it says.

## Try it

```bash
uv run chat.py
```

Ask something that produces a longer answer (e.g., "Explain the difference between a list and a tuple in Python, in three sentences.") and watch the reply print incrementally rather than appearing all at once.

## Checkpoint

<details>
<summary>Full <code>chat.py</code> (streaming stage)</summary>

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
| Output has no newline before the next `You:` prompt | Forgot the trailing `print()` after the `for` loop | Add it back — it exists solely to end the streamed line |
| `messages` ends up with `None` mixed into the assistant content | Appending `token` directly instead of filtering with `if token:` first | Only append/print non-`None` tokens |
| Nothing prints until the whole reply is done anyway | Missing `flush=True`, or terminal buffering output | Confirm both `end=""` and `flush=True` are present |

Next: **[Bounded Memory](07-bounded-memory.md)** — the `messages` list currently grows forever. Let's see why that's a problem and fix it.

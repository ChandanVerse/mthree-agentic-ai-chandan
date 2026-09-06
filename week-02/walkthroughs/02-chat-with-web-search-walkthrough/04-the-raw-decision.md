# Step 3 — The Raw Decision

> [Back to index](README.md) · Previous: [Defining the Tool Contract](03-defining-the-tool-contract.md) · Next: [Parsing the Decision](05-parsing-the-decision.md)

## Goal

Wire the tool description into a real system prompt, make a real (non-streaming) model call, and look at the model's raw text reply before writing a single line of parsing code.

## Why this matters

It's easy to jump straight to writing a regex and never actually confirm what you're regexing against. Before `try_parse_action()` exists, you should see with your own eyes that a small local model, taught the `ACTION:` protocol in its system prompt, actually produces that literal text for a question that needs the tool — and produces an ordinary answer, with no `ACTION:` line at all, for a question that doesn't. If that's not true for your model tag, no amount of downstream parsing logic will fix it; you'd need a different tag or the retry loop built in a later step to do more work.

This step also introduces `call_model()` as a **non-streaming** call — a deliberate difference from [01-basic-chat-app](../01-basic-chat-app-walkthrough/06-streaming-responses.md)'s token-by-token streaming. Streaming prints output to the user as it arrives, but this app needs the complete reply text *before* deciding whether it's a tool call or a direct answer — you can't un-print an `ACTION:` line the user already saw stream past. Getting the whole reply back in one piece is what makes that decision possible.

Create the file:

```bash
touch chat_with_tools.py
```

## 1. Imports and configuration

```python
#!/usr/bin/env python3
"""Step 2 — a chat app augmented with one tool (web search) the model may
choose to use.

Run:
    uv run chat_with_tools.py
"""
import os

from openai import OpenAI

from tools import tools_prompt_block

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")
```

Same `DEFAULT_MODEL`/`DEFAULT_BASE_URL`/environment-variable pattern as every other app in this series — nothing new here.

## 2. The system prompt

```python
SYSTEM_PROMPT = f"""You are a helpful assistant with access to one tool.

{tools_prompt_block()}

To call a tool, reply with ONLY this line (no other text):
ACTION: web_search {{"query": "your search query"}}

If you don't need the tool, just answer the user's question directly in plain text.
Only call the tool when you genuinely need current or external information —
not for things you already know.
"""
```

Three things this prompt is doing, worth calling out to trainees:

- `tools_prompt_block()` from Step 2 is embedded directly — the model's view of "what tools exist" and the code's view (`TOOLS`) come from the exact same source, so they can't drift apart.
- The example `ACTION:` line shows the model the literal syntax to reproduce, rather than describing the format in prose and hoping it infers the punctuation correctly.
- "Only call the tool when you genuinely need current or external information" exists because small models tend to over-trigger on anything tool-shaped — see this step's "Common mistakes" table.

## 3. A minimal, non-streaming model call

```python
def call_model(client: OpenAI, model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content
```

No `stream=True`, no chunk iteration — just the complete message content, returned in one piece.

## 4. Scratch code: inspect two raw decisions

Add a temporary `main()` at the bottom — this is throwaway code to observe behavior, not the finished app:

```python
def main() -> None:
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key="not-needed")

    for question in ["What's 12 * 7?", "What's the latest stable version of Python?"]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        reply = call_model(client, DEFAULT_MODEL, messages)
        print(f"Q: {question}")
        print(f"raw reply: {reply!r}\n")


if __name__ == "__main__":
    main()
```

## Try it

```bash
uv run chat_with_tools.py
```

Expected output (your model's exact wording may differ — what matters is the shape):

```text
Q: What's 12 * 7?
raw reply: '12 * 7 is 84.'

Q: What's the latest stable version of Python?
raw reply: 'ACTION: web_search {"query": "latest stable version of Python"}'
```

Point out to trainees exactly what's different between the two replies: the first is a plain-text answer, full stop. The second is *only* the `ACTION:` line — no other prose around it, because the system prompt explicitly asked for that. The next step turns that raw string into a `(tool_name, args)` pair your code can act on.

## Checkpoint

<details>
<summary>Full <code>chat_with_tools.py</code> (raw decision inspection — temporary <code>main()</code>)</summary>

```python
#!/usr/bin/env python3
"""Step 2 — a chat app augmented with one tool (web search) the model may
choose to use.

Run:
    uv run chat_with_tools.py
"""
import os

from openai import OpenAI

from tools import tools_prompt_block

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")

SYSTEM_PROMPT = f"""You are a helpful assistant with access to one tool.

{tools_prompt_block()}

To call a tool, reply with ONLY this line (no other text):
ACTION: web_search {{"query": "your search query"}}

If you don't need the tool, just answer the user's question directly in plain text.
Only call the tool when you genuinely need current or external information —
not for things you already know.
"""


def call_model(client: OpenAI, model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def main() -> None:
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key="not-needed")

    for question in ["What's 12 * 7?", "What's the latest stable version of Python?"]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        reply = call_model(client, DEFAULT_MODEL, messages)
        print(f"Q: {question}")
        print(f"raw reply: {reply!r}\n")


if __name__ == "__main__":
    main()
```

</details>

This is scaffolding for observation, not final code — the temporary `main()` above is fully replaced in [Step 8](09-assembling-the-cli-and-loop.md). `call_model()` and `SYSTEM_PROMPT`, though, are final as written here.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| Model always emits `ACTION:`, even for "What's 12 * 7?" | Small models over-trigger on anything tool-shaped | Tighten the "only call when you genuinely need it" instruction further, or accept this as the reliability gap the retry/validation steps exist for |
| Model prints prose *around* the `ACTION:` line instead of only that line | The model didn't follow the strict-format instruction exactly | Expected occasionally with small local models — Step 4's parser is deliberately built to still find the line even with surrounding text |
| `openai.APIConnectionError` | Docker Model Runner isn't running, or the model wasn't pulled | See [00-local-model-setup](../00-local-model-setup/README.md) |

Next: **[Parsing the Decision](05-parsing-the-decision.md)** — turn that raw reply text into a `(tool_name, args)` pair your code can act on.

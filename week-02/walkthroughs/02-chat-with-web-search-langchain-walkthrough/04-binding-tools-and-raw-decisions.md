# Step 3 — Binding Tools and Raw Decisions

> [Back to index](README.md) · Previous: [Defining the Tool](03-defining-the-tool.md) · Next: [Executing the Tool Call](05-executing-the-tool-call.md)

## Goal

Wire the tool into a real model call with `bind_tools()`, and look directly at what the model's decision looks like on the wire — before writing any code that acts on it.

## Why this matters

This is the step the hand-rolled version can't show you as cleanly, because in that app the "decision" is buried inside a raw text reply that then has to be regexed apart from everything else the model said. Here, the decision is already a structured field: `response.tool_calls`. Isolating this step — decide, then look, before you ever execute anything — matters because it's the fastest way to answer the question this whole variant exists to test: *does your model tag actually honor structured tool-calling?* If `tool_calls` comes back populated exactly when it should (and empty exactly when it shouldn't), the rest of this walkthrough is just plumbing. If it doesn't, no amount of downstream code will fix that — you'd need the hand-rolled fallback instead.

## 1. Add the new imports

At the top of the file, alongside the existing imports:

```python
from ddgs import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
```

## 2. Bind the tool and inspect two decisions

Add a temporary `main()` at the bottom of the file — this is scratch code to observe behavior, not the finished loop:

```python
def main() -> None:
    llm = ChatOpenAI(model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, api_key="not-needed", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    for question in ["What's 12 * 7?", "Who won the most recent F1 world championship?"]:
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
        response = llm_with_tools.invoke(messages)
        print(f"Q: {question}")
        print(f"tool_calls: {response.tool_calls}")
        print(f"content: {response.content!r}\n")


if __name__ == "__main__":
    main()
```

`temperature=0` is set for reproducibility while you're comparing behavior — it doesn't change anything structural about `bind_tools()`. Also note the same "Docker Model Runner doesn't check the API key, but the client requires one" workaround from every other app in this series: `api_key="not-needed"`.

## Try it

```bash
uv run chat_with_tools.py
```

Expected output (your model's exact wording and the call's `id` will differ — what matters is the shape):

```text
Q: What's 12 * 7?
tool_calls: []
content: "12 * 7 is 84."

Q: Who won the most recent F1 world championship?
tool_calls: [{'name': 'web_search', 'args': {'query': 'most recent F1 world championship winner'}, 'id': 'eLuQIsMU2C7sQh00AgEyxin1UbhbZfUA', 'type': 'tool_call'}]
content: ''
```

Point out to trainees what each field means: `name` is which tool, matched against `TOOLS_BY_NAME` in the next step; `args` is already a parsed dict — no JSON string to `json.loads()`; `id` is what ties a later `ToolMessage` back to this specific call. When `tool_calls` is empty, `content` carries the model's direct answer instead — that's the "no tool needed" path, and it's the same field the finished `run_turn()` returns for.

## Checkpoint

<details>
<summary>Full <code>chat_with_tools.py</code> (raw decision inspection — temporary <code>main()</code>)</summary>

```python
#!/usr/bin/env python3
"""Step 2, LangChain variant — the same one-tool chat app as
../02-chat-with-web-search, rebuilt on LangChain's native tool-calling
instead of a hand-rolled text protocol.

The original app teaches the model a strict text format in the system
prompt ("ACTION: web_search {...}") and then regexes that back out of the
reply, because Gemma via Docker Model Runner wasn't assumed to support
provider-native tool-calling reliably. This version tests that assumption:
`ChatOpenAI(...).bind_tools([web_search])` asks the model to return a
structured tool call, and LangChain parses it for you — no ACTION_NAME_PATTERN,
no JSON_OBJECT_PATTERN, no try_parse_action.

Same web_search tool (same DDGS backend, same bounded has_more/total return
shape) as the hand-rolled version, so the only variable being compared is
the tool-calling mechanism itself. Whether this is actually more reliable
than the text-protocol version depends on how well your specific Gemma tag
handles tool-calling through Docker Model Runner's OpenAI-compatible
endpoint — small local models are not guaranteed to honor it every turn.
Run both apps on the same prompts and compare.

Run:
    uv run chat_with_tools.py
"""
import os

from ddgs import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a web_search tool. "
    "Only call it when you genuinely need current or external information — "
    "not for things you already know."
)


@tool
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the live web and return up to 5 results (title, url, snippet).

    Use this for anything you can't already answer confidently from general
    knowledge — current events, prices, recent facts.
    """
    with DDGS() as ddgs:
        # Ask for one extra result so we can tell whether more exist,
        # without a second round-trip — same trick as the hand-rolled version.
        hits = list(ddgs.text(query, max_results=max_results + 1))

    has_more = len(hits) > max_results
    hits = hits[:max_results]
    return {
        "results": [{"title": h.get("title"), "url": h.get("href"), "snippet": h.get("body")} for h in hits],
        "has_more": has_more,
        "total": len(hits) + (1 if has_more else 0),
    }


TOOLS = [web_search]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def main() -> None:
    llm = ChatOpenAI(model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, api_key="not-needed", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    for question in ["What's 12 * 7?", "Who won the most recent F1 world championship?"]:
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
        response = llm_with_tools.invoke(messages)
        print(f"Q: {question}")
        print(f"tool_calls: {response.tool_calls}")
        print(f"content: {response.content!r}\n")


if __name__ == "__main__":
    main()
```

</details>

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `tool_calls` is always `[]`, even for questions that need search | Model tag doesn't reliably support structured tool-calling through this endpoint | Try a different Gemma tag, or fall back to the hand-rolled text-protocol app |
| `openai.APIConnectionError` | Docker Model Runner isn't running, or the model wasn't pulled | See [00-local-model-setup](../00-local-model-setup/README.md) |
| Forgot to bind tools, called `llm.invoke(...)` instead of `llm_with_tools.invoke(...)` | Easy copy-paste slip | `response.tool_calls` will always be empty on the unbound `llm` — check which variable you called |

Next: **[Executing the Tool Call](05-executing-the-tool-call.md)** — turn this observation into an actual round trip: run the tool, and feed its result back to the model.

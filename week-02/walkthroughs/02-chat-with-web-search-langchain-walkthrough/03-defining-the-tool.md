# Step 2 — Defining the Tool

> [Back to index](README.md) · Previous: [Environment Setup](02-environment-setup.md) · Next: [Binding Tools and Raw Decisions](04-binding-tools-and-raw-decisions.md)

## Goal

Write the `web_search` tool using LangChain's `@tool` decorator, and compare it directly against the hand-rolled version's [tools.py](../02-chat-with-web-search/tools.py).

## Why this matters

The hand-rolled version needs two things to describe a tool to the model: a Pydantic schema (`WebSearchArgs`) so a malformed call can be rejected before it runs, and `tools_prompt_block()` to render that schema as text inside the system prompt, because the model has no other way to learn what's available. `@tool` collapses both into one thing: it reads the function's type hints to build the schema, and its docstring to build the description — then `bind_tools()` (next step) sends that as a real API field instead of prompt text. You're not writing less *capability*, you're writing less *plumbing* — the tool still needs a name, a typed signature, and a description; you just don't hand-maintain a second copy of that information for the prompt.

Create the file:

```bash
touch chat_with_tools.py
```

## 1. Write the module docstring and imports

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
from langchain_core.tools import tool

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a web_search tool. "
    "Only call it when you genuinely need current or external information — "
    "not for things you already know."
)
```

Notice `SYSTEM_PROMPT` no longer contains any tool description, ACTION-line instructions, or a `{tools_prompt_block()}` interpolation — just a plain behavioral instruction. That's the whole diff against the hand-rolled version's prompt: the *mechanism* for calling a tool used to have to be taught in prose; now it's an API-level concern `bind_tools()` handles in Step 3.

## 2. Define the tool

```python
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
```

Compare this to [tools.py](../02-chat-with-web-search/tools.py): the function body — the actual `DDGS` call and the `has_more`/`total` shape — is identical. What's gone is `WebSearchArgs` (the type hints on `query`/`max_results` are the schema now) and the `TOOLS` dict's `"description"` key (the docstring is the description now). `TOOLS_BY_NAME` plays the same role the hand-rolled `TOOLS` dict does — a lookup from the name the model uses back to something callable — but keyed off `t.name`, which `@tool` derived from the function name for you.

## Try it

The `@tool` decorator turns `web_search` into a `StructuredTool` object, not a plain function, so you can't call it directly — invoke it the way `run_turn()` will later, via `.invoke(...)`:

```bash
uv run python -c "from chat_with_tools import web_search; print(web_search.invoke({'query': 'python programming language'}))"
```

You should see a dict shaped like:

```python
{'results': [{'title': '...', 'url': '...', 'snippet': '...'}, ...], 'has_more': True, 'total': 6}
```

The exact results vary (it's a live search), but the shape — `results`/`has_more`/`total` — never does, same as the hand-rolled version.

## Checkpoint

<details>
<summary>Full <code>chat_with_tools.py</code> (tool defined, no model wiring yet)</summary>

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
from langchain_core.tools import tool

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
```

</details>

This is deliberately incomplete — there's no `main()` yet, so the file isn't runnable as an app until Step 6. Everything written here survives unchanged into the finished file.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `TypeError: 'StructuredTool' object is not callable` | Called `web_search(query=...)` directly instead of `.invoke(...)` | Use `web_search.invoke({"query": ...})` |
| Tool's description looks empty or wrong to the model later | Docstring missing, or moved below the type-hinted signature | Keep the docstring as the first statement in the function body — that's what `@tool` reads |
| `max_results` doesn't behave as expected | Forgot the type hint / default (`max_results: int = 5`) | `@tool` derives the schema from the signature — an untyped or missing default changes what it infers |

Next: **[Binding Tools and Raw Decisions](04-binding-tools-and-raw-decisions.md)** — wire this tool into a model call and see what the model's decision actually looks like before anything gets executed.

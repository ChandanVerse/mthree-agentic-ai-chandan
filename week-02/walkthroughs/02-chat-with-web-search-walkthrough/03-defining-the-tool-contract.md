# Step 2 — Defining the Tool Contract

> [Back to index](README.md) · Previous: [Environment Setup](02-environment-setup.md) · Next: [The Raw Decision](04-the-raw-decision.md)

## Goal

Build `tools.py`: a `web_search` tool with a typed argument schema, a real (network-calling) implementation, and a small registry that describes it — all before any model is involved.

## Why this matters

A tool is a contract, not just a function. If you skip the schema and just let the model's text flow straight into a Python function call, a misspelled field or an extra key doesn't fail loudly — it either crashes deep inside your code or, worse, gets silently ignored. Defining `WebSearchArgs` as a Pydantic model means a malformed call gets rejected with a specific, readable error *before* it ever touches the network — and that error is exactly what gets fed back to the model in a later step, so it can correct itself.

The return shape matters just as much as the input shape. A tool that returns a bare list of five results invites both the model and your own code to treat "the five things I got back" as "everything there is." Returning `{"results": [...], "has_more": bool, "total": n}` instead makes that distinction explicit — nobody downstream can mistake a truncated view for a complete one.

Create the file:

```bash
touch tools.py
```

## 1. Module docstring and imports

```python
"""Tool definitions for the tool-augmented chat app.

A tool is a contract: a name, a typed argument schema, and a predictable
return shape — not just "a Python function the model happens to call". We
use Pydantic for the schema so a malformed call can be rejected *before*
it runs, and we design the return shape to never lie about being complete.
"""
from pydantic import BaseModel, Field
from ddgs import DDGS
```

## 2. The argument schema

```python
class WebSearchArgs(BaseModel):
    """Arguments for the web_search tool. Extra/misspelled fields from the
    model are rejected by Pydantic rather than silently ignored."""

    query: str = Field(..., description="The search query, e.g. 'current weather in Tokyo'")
```

A single required field, `query`. `Field(..., description=...)` isn't decorative — that description string is what gets rendered into the system prompt in the next step, so the model sees the same explanation of what `query` means that a developer reading this class would.

## 3. The implementation

```python
def web_search(query: str, max_results: int = 5) -> dict:
    """Run a live web search and return a small, bounded set of results.

    Returns an explicit `has_more`/`total` shape instead of a bare list —
    the caller (and the model) should never mistake "top 5 shown" for
    "the complete picture" the way a bare list implies.
    """
    with DDGS() as ddgs:
        # Ask for one extra result so we can tell whether more exist,
        # without a second round-trip.
        hits = list(ddgs.text(query, max_results=max_results + 1))

    has_more = len(hits) > max_results
    hits = hits[:max_results]
    return {
        "results": [{"title": h.get("title"), "url": h.get("href"), "snippet": h.get("body")} for h in hits],
        "has_more": has_more,
        "total": len(hits) + (1 if has_more else 0),
    }
```

The `max_results + 1` trick is worth pointing out explicitly: asking for one extra result and then slicing it back off is how the function learns whether more results *exist* without a second network call. Without it, you'd either always report `has_more: False` (a lie) or need to search twice (wasteful).

## 4. The tool registry and its prompt rendering

```python
# The tool registry: everything the dispatcher needs to validate, describe,
# and execute a call, keyed by the name the model uses in its ACTION line.
TOOLS = {
    "web_search": {
        "schema": WebSearchArgs,
        "execute": web_search,
        "description": (
            "Search the live web and return up to 5 results (title, url, snippet). "
            "Use it for anything you can't already answer confidently from general "
            "knowledge — current events, prices, recent facts."
        ),
    }
}


def tools_prompt_block() -> str:
    """Render the tool contract as text for the system prompt.

    Small local models served via Docker Model Runner (Gemma included) don't
    reliably support the provider-native "tools" API parameter the way
    hosted Claude/GPT calls do. So instead of relying on that, we teach the
    model a strict text protocol by hand — the same underlying idea (name +
    typed args script), just carried in the prompt instead of an API field.
    """
    lines = ["Available tools:"]
    for name, spec in TOOLS.items():
        field_names = ", ".join(spec["schema"].model_fields)
        lines.append(f"- {name}({field_names}): {spec['description']}")
    return "\n".join(lines)
```

`TOOLS` is keyed by the exact name the model will use in its `ACTION:` line — that key is what a later dispatch step looks up against. `tools_prompt_block()` walks the registry and turns it into the plain-text block the model reads in its system prompt; `spec["schema"].model_fields` pulls the field names straight from the Pydantic model, so the prompt and the validator can never drift out of sync with each other.

## Try it

```bash
uv run python -c "from tools import tools_prompt_block, web_search; print(tools_prompt_block()); print(web_search('current weather in Tokyo', max_results=2))"
```

Expect the rendered tool description followed by a real search result:

```text
Available tools:
- web_search(query): Search the live web and return up to 5 results (title, url, snippet). Use it for anything you can't already answer confidently from general knowledge — current events, prices, recent facts.
{'results': [{'title': '...', 'url': '...', 'snippet': '...'}, {'title': '...', 'url': '...', 'snippet': '...'}], 'has_more': True, 'total': 3}
```

## Checkpoint

<details>
<summary>Full <code>tools.py</code></summary>

```python
"""Tool definitions for the tool-augmented chat app.

A tool is a contract: a name, a typed argument schema, and a predictable
return shape — not just "a Python function the model happens to call". We
use Pydantic for the schema so a malformed call can be rejected *before*
it runs, and we design the return shape to never lie about being complete.
"""
from pydantic import BaseModel, Field
from ddgs import DDGS


class WebSearchArgs(BaseModel):
    """Arguments for the web_search tool. Extra/misspelled fields from the
    model are rejected by Pydantic rather than silently ignored."""

    query: str = Field(..., description="The search query, e.g. 'current weather in Tokyo'")


def web_search(query: str, max_results: int = 5) -> dict:
    """Run a live web search and return a small, bounded set of results.

    Returns an explicit `has_more`/`total` shape instead of a bare list —
    the caller (and the model) should never mistake "top 5 shown" for
    "the complete picture" the way a bare list implies.
    """
    with DDGS() as ddgs:
        # Ask for one extra result so we can tell whether more exist,
        # without a second round-trip.
        hits = list(ddgs.text(query, max_results=max_results + 1))

    has_more = len(hits) > max_results
    hits = hits[:max_results]
    return {
        "results": [{"title": h.get("title"), "url": h.get("href"), "snippet": h.get("body")} for h in hits],
        "has_more": has_more,
        "total": len(hits) + (1 if has_more else 0),
    }


# The tool registry: everything the dispatcher needs to validate, describe,
# and execute a call, keyed by the name the model uses in its ACTION line.
TOOLS = {
    "web_search": {
        "schema": WebSearchArgs,
        "execute": web_search,
        "description": (
            "Search the live web and return up to 5 results (title, url, snippet). "
            "Use it for anything you can't already answer confidently from general "
            "knowledge — current events, prices, recent facts."
        ),
    }
}


def tools_prompt_block() -> str:
    """Render the tool contract as text for the system prompt.

    Small local models served via Docker Model Runner (Gemma included) don't
    reliably support the provider-native "tools" API parameter the way
    hosted Claude/GPT calls do. So instead of relying on that, we teach the
    model a strict text protocol by hand — the same underlying idea (name +
    typed args script), just carried in the prompt instead of an API field.
    """
    lines = ["Available tools:"]
    for name, spec in TOOLS.items():
        field_names = ", ".join(spec["schema"].model_fields)
        lines.append(f"- {name}({field_names}): {spec['description']}")
    return "\n".join(lines)
```

</details>

This matches [../02-chat-with-web-search/tools.py](../02-chat-with-web-search/tools.py) exactly.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `pydantic.ValidationError` while just testing `WebSearchArgs` directly | Forgot the required `query` field, e.g. `WebSearchArgs()` | Always pass `query=...` |
| `web_search(...)` hangs or raises a network error | No internet access, or a transient search-backend issue | Confirm the Step 1 `ddgs` check still works on its own |
| `tools_prompt_block()` output doesn't mention a field you expected | `field_names` comes from `spec["schema"].model_fields` — it reflects the Pydantic model, not the `description` string | Add/rename fields on `WebSearchArgs`, not in the description text |

Next: **[The Raw Decision](04-the-raw-decision.md)** — wire this tool into a real model call, and look at the model's decision as raw, unparsed text.

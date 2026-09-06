# Step 3 — Web Search, Save Note, and the Registry

> [Back to index](README.md) · Previous: [The Calculator Tool](03-the-calculator-tool.md) · Next: [The Profile and Reply Format](05-the-profile-and-reply-format.md)

## Goal

Add `web_search` and `save_note`, then finish `tools.py` with the `TOOLS` registry and the `IDEMPOTENT_TOOLS`/`SIDE_EFFECTING_TOOLS` classification that a later dispatcher step depends on.

## Why this matters

`web_search` is the same tool from [02-chat-with-web-search](../02-chat-with-web-search-walkthrough/03-defining-the-tool-contract.md) — read-only, bounded, nothing new. `save_note` is genuinely new to this series: it's the first tool in any of these apps that **mutates state outside the conversation**. Writing to disk means a second, identical call after an ambiguous failure isn't free — if the first write actually landed but the app couldn't confirm it, blindly retrying could produce two notes instead of one.

That's the reason `IDEMPOTENT_TOOLS` and `SIDE_EFFECTING_TOOLS` exist as two explicit sets rather than one flag on a per-call basis: the classification is a property of the *tool*, decided once, here, rather than something the dispatcher has to guess about a specific failure later. Get the classification right at the source, and every later retry decision follows from a single set-membership check.

## 1. Update the imports and add `NOTES_DIR`

Replace the imports block at the top of `tools.py` with:

```python
import ast
import operator
import os
from pathlib import Path

from pydantic import BaseModel, Field
from ddgs import DDGS

NOTES_DIR = Path(__file__).parent / "agent_notes"
```

`os` and `Path` are for `save_note`'s file writing; `DDGS` is the no-API-key DuckDuckGo client `web_search` uses. `NOTES_DIR` is defined once, near the top, even though only `save_note` (added below) uses it.

## 2. `web_search`

Add this after the `calculator` section:

```python
# --------------------------------------------------------------------------
# web_search — idempotent
# --------------------------------------------------------------------------


class WebSearchArgs(BaseModel):
    query: str = Field(..., description="The search query, e.g. 'population of France'")


def web_search(query: str, max_results: int = 5) -> dict:
    """Live web search with an explicit has_more/total shape so a partial
    result set is never mistaken for the complete picture."""
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results + 1))
    has_more = len(hits) > max_results
    hits = hits[:max_results]
    return {
        "results": [{"title": h.get("title"), "url": h.get("href"), "snippet": h.get("body")} for h in hits],
        "has_more": has_more,
        "total": len(hits) + (1 if has_more else 0),
    }
```

Identical in shape to [02-chat-with-web-search's `web_search`](../02-chat-with-web-search-walkthrough/03-defining-the-tool-contract.md#3-the-implementation) — the `max_results + 1` trick, the `has_more`/`total` bounded return. If you've already built that app, this is a straight copy.

## 3. `save_note`

```python
# --------------------------------------------------------------------------
# save_note — side-effecting
# --------------------------------------------------------------------------


class SaveNoteArgs(BaseModel):
    title: str = Field(..., description="Short filename-safe title for the note")
    content: str = Field(..., description="The note's body text")


def save_note(title: str, content: str) -> dict:
    """Write a markdown note to ./agent_notes/. This is the one side-effecting
    tool in this app — it mutates external state (the filesystem), so it
    must be classified and handled differently from the two read-only tools
    above (see SIDE_EFFECTING_TOOLS in agent.py)."""
    NOTES_DIR.mkdir(exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in title.strip()) or "untitled"
    path = NOTES_DIR / f"{safe_name}.md"
    path.write_text(f"# {title}\n\n{content}\n")
    return {"saved_to": os.path.relpath(path)}
```

Two things worth pointing out: `safe_name` strips anything that isn't alphanumeric, `-`, or `_` out of the title before it becomes part of a filesystem path — the model's `title` string is untrusted input, and a raw title like `"../../etc/passwd"` must never reach `Path` unsanitized. And the return value reports `saved_to` as a relative path — a small but deliberate piece of the tool's bounded return shape, so the model (and a human reading the trace) can see exactly where the file landed.

## 4. The registry and classification

```python
# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

TOOLS = {
    "calculator": {
        "schema": CalculatorArgs,
        "execute": calculator,
        "description": "Evaluate a numeric arithmetic expression and return a float. Never compute totals mentally — always use this.",
    },
    "web_search": {
        "schema": WebSearchArgs,
        "execute": web_search,
        "description": "Search the live web and return up to 5 results (title, url, snippet).",
    },
    "save_note": {
        "schema": SaveNoteArgs,
        "execute": save_note,
        "description": "Save a short note to disk for later reference. This WRITES A FILE — a side effect, not a lookup.",
    },
}

# Classify every tool as safe-to-auto-retry or never-auto-retry. Read-only
# lookups are idempotent; anything that mutates state is not.
IDEMPOTENT_TOOLS = {"calculator", "web_search"}
SIDE_EFFECTING_TOOLS = {"save_note"}


def tools_prompt_block() -> str:
    """Render the tool contract as text for the system prompt — see the
    matching function in ../02-chat-with-web-search/tools.py for why this is
    a hand-rolled text protocol rather than a native "tools" API parameter."""
    lines = ["Available tools:"]
    for name, spec in TOOLS.items():
        field_names = ", ".join(spec["schema"].model_fields)
        lines.append(f"- {name}({field_names}): {spec['description']}")
    return "\n".join(lines)
```

`save_note`'s description is written in capital letters on purpose ("WRITES A FILE") — a small nudge in the prompt text itself, on top of the structural classification, since the system prompt later adds an explicit rule that `save_note` should only be called when the user actually asks for it. `IDEMPOTENT_TOOLS` and `SIDE_EFFECTING_TOOLS` are plain Python sets keyed by the same tool names as `TOOLS` — nothing enforces that every tool appears in exactly one of them, so double-check that by hand whenever you add a fourth tool later (Exercise 1 in the recap does exactly this).

## Try it

```bash
uv run python -c "
from tools import tools_prompt_block, TOOLS, IDEMPOTENT_TOOLS, SIDE_EFFECTING_TOOLS, save_note
print(tools_prompt_block())
print()
print(save_note('Test Note', 'Hello from the small agent.'))
print('idempotent:', IDEMPOTENT_TOOLS, ' side-effecting:', SIDE_EFFECTING_TOOLS)
"
```

Expected output:

```text
Available tools:
- calculator(expression): Evaluate a numeric arithmetic expression and return a float. Never compute totals mentally — always use this.
- web_search(query): Search the live web and return up to 5 results (title, url, snippet).
- save_note(title, content): Save a short note to disk for later reference. This WRITES A FILE — a side effect, not a lookup.

{'saved_to': 'agent_notes/Test_Note.md'}
idempotent: {'calculator', 'web_search'}  side-effecting: {'save_note'}
```

A real `agent_notes/Test_Note.md` file should now exist in your project folder — delete it freely, it's gitignored.

## Checkpoint

<details>
<summary>Full <code>tools.py</code></summary>

```python
"""Tool definitions for the small agent.

Three tools, deliberately chosen to cover both halves of the idempotent
vs. side-effecting distinction:

- `calculator` and `web_search` are read-only / idempotent — safe to
  auto-retry on a transient failure.
- `save_note` writes a file — a side-effecting action that must NOT be
  blindly retried (a retried "save" after an ambiguous failure could write
  a duplicate note).

Each tool is a contract: a name, a typed Pydantic schema, and a
predictable, bounded return shape — not just a bare Python function.
"""
import ast
import operator
import os
from pathlib import Path

from pydantic import BaseModel, Field
from ddgs import DDGS

NOTES_DIR = Path(__file__).parent / "agent_notes"

# --------------------------------------------------------------------------
# calculator — idempotent
# --------------------------------------------------------------------------


class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="A numeric arithmetic expression, e.g. '17 * 12.99'")


# Only these AST node types are allowed — this is what makes `calculator`
# safe to expose to a model at all. A bare `eval()` on model-generated text
# would let it run arbitrary Python (e.g. "__import__('os').system(...)");
# walking a restricted AST instead means the tool can only ever compute
# arithmetic, no matter what string the model sends.
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def calculator(expression: str) -> float:
    """Evaluate a numeric expression using a restricted AST walk (safe subset
    of Python arithmetic — no names, calls, attribute access, or imports)."""
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as e:
        # A predictable, typed failure — the dispatcher feeds this back to
        # the model as a correctable argument error.
        raise ValueError(f"could not evaluate '{expression}': {e}") from e


# --------------------------------------------------------------------------
# web_search — idempotent
# --------------------------------------------------------------------------


class WebSearchArgs(BaseModel):
    query: str = Field(..., description="The search query, e.g. 'population of France'")


def web_search(query: str, max_results: int = 5) -> dict:
    """Live web search with an explicit has_more/total shape so a partial
    result set is never mistaken for the complete picture."""
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results + 1))
    has_more = len(hits) > max_results
    hits = hits[:max_results]
    return {
        "results": [{"title": h.get("title"), "url": h.get("href"), "snippet": h.get("body")} for h in hits],
        "has_more": has_more,
        "total": len(hits) + (1 if has_more else 0),
    }


# --------------------------------------------------------------------------
# save_note — side-effecting
# --------------------------------------------------------------------------


class SaveNoteArgs(BaseModel):
    title: str = Field(..., description="Short filename-safe title for the note")
    content: str = Field(..., description="The note's body text")


def save_note(title: str, content: str) -> dict:
    """Write a markdown note to ./agent_notes/. This is the one side-effecting
    tool in this app — it mutates external state (the filesystem), so it
    must be classified and handled differently from the two read-only tools
    above (see SIDE_EFFECTING_TOOLS in agent.py)."""
    NOTES_DIR.mkdir(exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in title.strip()) or "untitled"
    path = NOTES_DIR / f"{safe_name}.md"
    path.write_text(f"# {title}\n\n{content}\n")
    return {"saved_to": os.path.relpath(path)}


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

TOOLS = {
    "calculator": {
        "schema": CalculatorArgs,
        "execute": calculator,
        "description": "Evaluate a numeric arithmetic expression and return a float. Never compute totals mentally — always use this.",
    },
    "web_search": {
        "schema": WebSearchArgs,
        "execute": web_search,
        "description": "Search the live web and return up to 5 results (title, url, snippet).",
    },
    "save_note": {
        "schema": SaveNoteArgs,
        "execute": save_note,
        "description": "Save a short note to disk for later reference. This WRITES A FILE — a side effect, not a lookup.",
    },
}

# Classify every tool as safe-to-auto-retry or never-auto-retry. Read-only
# lookups are idempotent; anything that mutates state is not.
IDEMPOTENT_TOOLS = {"calculator", "web_search"}
SIDE_EFFECTING_TOOLS = {"save_note"}


def tools_prompt_block() -> str:
    """Render the tool contract as text for the system prompt — see the
    matching function in ../02-chat-with-web-search/tools.py for why this is
    a hand-rolled text protocol rather than a native "tools" API parameter."""
    lines = ["Available tools:"]
    for name, spec in TOOLS.items():
        field_names = ", ".join(spec["schema"].model_fields)
        lines.append(f"- {name}({field_names}): {spec['description']}")
    return "\n".join(lines)
```

</details>

This matches [../03-small-agent/tools.py](../03-small-agent/tools.py) exactly.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `save_note` raises `FileNotFoundError` | `NOTES_DIR.mkdir(exist_ok=True)` missing or called with the wrong path | Confirm `NOTES_DIR = Path(__file__).parent / "agent_notes"` and the `mkdir` call are both present |
| A note's filename doesn't match its title at all | `safe_name`'s comprehension replaces every non-alphanumeric character with `_`, including spaces | Expected behavior — titles with spaces or punctuation become underscored filenames |
| `tools_prompt_block()` doesn't mention `save_note` | `TOOLS` dict wasn't updated, or `save_note` was defined but never registered | Every tool needs an entry in `TOOLS`, not just a function definition |
| A tool works but never gets retried after a transient failure, even though it should | Tool name missing from `IDEMPOTENT_TOOLS` | Add it — the dispatcher in Step 7 checks this set, not the tool's actual behavior |

Next: **[The Profile and Reply Format](05-the-profile-and-reply-format.md)** — start `agent.py`, wire the tool registry into a system prompt, and look at the model's raw multi-block decision.

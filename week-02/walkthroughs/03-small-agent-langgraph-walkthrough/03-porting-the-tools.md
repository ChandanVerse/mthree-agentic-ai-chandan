# Step 2 — Porting the Tools

> [Back to index](README.md) · Previous: [Environment Setup](02-environment-setup.md) · Next: [The save_note Closure](04-the-save-note-closure.md)

## Goal

Port `calculator` and `web_search` from [tools.py](../03-small-agent/tools.py) to LangChain's `@tool` decorator, keeping their actual behavior byte-for-byte identical.

## Why this matters

The hand-rolled version needs three things to describe a tool to the model: a Pydantic schema class (`CalculatorArgs`, `WebSearchArgs`) so a malformed call can be rejected before it runs, an entry in the `TOOLS` registry dict pairing that schema with the function and a description string, and `tools_prompt_block()` to render all of that as text inside the system prompt — because in the hand-rolled version, the model has no other way to learn what's available. `@tool` collapses all three into one: it reads the function's type hints to build the schema and its docstring to build the description, and `create_agent` (Step 4) sends that as a real API field instead of prompt text. This is the same trade [02-chat-with-web-search-langchain](../02-chat-with-web-search-langchain-walkthrough/03-defining-the-tool.md) made for a single tool — here it applies twice over, and a third time, differently, in the next step.

What's worth calling out explicitly: `calculator`'s actual safety property — a restricted AST walk instead of a bare `eval()` — isn't part of what's being compared at all. It carries over completely unchanged. The orchestration layer is what this whole app is testing; the tools themselves are deliberately left alone.

## 1. Write the module docstring and imports

Create the file and start it with:

```python
#!/usr/bin/env python3
"""Step 3, LangGraph variant — the same ReAct agent as ../03-small-agent,
rebuilt on LangGraph's prebuilt `create_react_agent` instead of a hand-rolled
Thought/Action/Observation loop.

The original agent.py hand-writes ~270 lines: a strict text protocol the
model must follow, regexes to pull Thought/Action/Action Input/Final Answer
out of the reply, a Pydantic-validating dispatcher, and a step-count
guardrail. This version replaces essentially all of that with:

    create_agent(model, tools)

`create_agent` (langchain>=1.0, in `langchain.agents`) is the current home
for what used to be `langgraph.prebuilt.create_react_agent` — that older
import still works in this langgraph version but prints a deprecation
warning pointing here. It's still a LangGraph state graph under the hood
(the object returned is a `CompiledStateGraph`, same `.invoke()`/recursion
mechanics), just re-homed into the `langchain` package as the two libraries
converge on one prebuilt-agent API. It implements exactly the ReAct loop
above: call the model, run whatever tool calls it asked for, feed the
results back, repeat until the model replies with no more tool calls. Like
../02-chat-with-web-search-langchain, this relies on the model's native
tool-calling ability through Docker Model Runner's OpenAI-compatible API
rather than a hand-taught text format.

Same three tools as the original (calculator, web_search, save_note), same
sandboxed AST-walking calculator — the orchestration layer is what's being
compared here, not the tools themselves. See `make_save_note_tool()` below
for how the original's "never blind-retry a side-effecting tool" guardrail
survives without the original's custom dispatcher — it doesn't survive
unchanged, and that gap is called out there and in the README.

Run:
    uv run agent.py "What is 17 * 12.99, and is that under 220?"
    uv run agent.py            # interactive: one goal per line
"""
import ast
import operator
import os
from pathlib import Path

from ddgs import DDGS
from langchain_core.tools import tool

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")
DEFAULT_MAX_STEPS = 6  # same guardrail role as ../03-small-agent, mapped to recursion_limit below

NOTES_DIR = Path(__file__).parent / "agent_notes"

SYSTEM_PROMPT = """You are a careful problem-solving agent. Work step by step and use tools instead of guessing.

Rules:
- Never compute arithmetic yourself — always call calculator.
- Only call save_note when the user explicitly asks you to save or remember something.
- If a tool call fails, read the error and correct your input on the next try.
- Give a final answer as soon as you genuinely have enough information — don't call tools you don't need.
"""
```

Notice `SYSTEM_PROMPT` no longer contains a rendered tool list, an output format, or a worked example — just plain behavioral rules. Compare this to the hand-rolled `SYSTEM_PROMPT`, which had to interpolate `tools_prompt_block()` and spell out an entire `Thought:`/`Action:`/`Action Input:` grammar with a worked example, because a small local model can't be trusted to infer a strict text format from a description alone. Native tool-calling doesn't need any of that — the tool list and each tool's schema travel as a structured API field, not prompt text.

## 2. Port the calculator tool

The sandboxed AST walk is copied verbatim from [../03-small-agent/tools.py](../03-small-agent/tools.py) — only the wrapper around it changes:

```python
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


@tool
def calculator(expression: str) -> float:
    """Evaluate a numeric arithmetic expression and return a float. Never compute totals mentally — always use this."""
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as e:
        raise ValueError(f"could not evaluate '{expression}': {e}") from e
```

Compare this to the hand-rolled version: `CalculatorArgs` (a whole Pydantic class) is gone — the single `expression: str` type hint is the schema now. The docstring is the description that used to live as a separate string in the `TOOLS` dict. Raising `ValueError` still works exactly the same way it did before: it becomes the tool's error message, which `create_agent`'s loop feeds back to the model as a `ToolMessage` for it to react to — you'll see that round trip directly once `build_agent()` exists in Step 4.

## 3. Port the web_search tool

```python
@tool
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the live web and return up to 5 results (title, url, snippet)."""
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

Same `DDGS` backend, same `has_more`/`total` shape as [tools.py](../03-small-agent/tools.py)'s version — the only diff is that `WebSearchArgs` and the `TOOLS` dict entry are gone, for the same reason as `calculator`.

## Try it

`@tool` turns each function into a `StructuredTool`, not a plain callable, so invoke it via `.invoke(...)`:

```bash
uv run python -c "from agent import calculator; print(calculator.invoke({'expression': '17 * 12.99'}))"
```

```text
220.83
```

```bash
uv run python -c "from agent import calculator; print(calculator.invoke({'expression': '1 / 0'}))"
```

This should raise, surfacing the same `ValueError` message the hand-rolled tool raises — nothing about the sandboxing changed.

## Checkpoint

<details>
<summary>Full <code>agent.py</code> (tools defined, no model wiring yet)</summary>

```python
#!/usr/bin/env python3
"""Step 3, LangGraph variant — the same ReAct agent as ../03-small-agent,
rebuilt on LangGraph's prebuilt `create_react_agent` instead of a hand-rolled
Thought/Action/Observation loop.

The original agent.py hand-writes ~270 lines: a strict text protocol the
model must follow, regexes to pull Thought/Action/Action Input/Final Answer
out of the reply, a Pydantic-validating dispatcher, and a step-count
guardrail. This version replaces essentially all of that with:

    create_agent(model, tools)

`create_agent` (langchain>=1.0, in `langchain.agents`) is the current home
for what used to be `langgraph.prebuilt.create_react_agent` — that older
import still works in this langgraph version but prints a deprecation
warning pointing here. It's still a LangGraph state graph under the hood
(the object returned is a `CompiledStateGraph`, same `.invoke()`/recursion
mechanics), just re-homed into the `langchain` package as the two libraries
converge on one prebuilt-agent API. It implements exactly the ReAct loop
above: call the model, run whatever tool calls it asked for, feed the
results back, repeat until the model replies with no more tool calls. Like
../02-chat-with-web-search-langchain, this relies on the model's native
tool-calling ability through Docker Model Runner's OpenAI-compatible API
rather than a hand-taught text format.

Same three tools as the original (calculator, web_search, save_note), same
sandboxed AST-walking calculator — the orchestration layer is what's being
compared here, not the tools themselves. See `make_save_note_tool()` below
for how the original's "never blind-retry a side-effecting tool" guardrail
survives without the original's custom dispatcher — it doesn't survive
unchanged, and that gap is called out there and in the README.

Run:
    uv run agent.py "What is 17 * 12.99, and is that under 220?"
    uv run agent.py            # interactive: one goal per line
"""
import ast
import operator
import os
from pathlib import Path

from ddgs import DDGS
from langchain_core.tools import tool

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")
DEFAULT_MAX_STEPS = 6  # same guardrail role as ../03-small-agent, mapped to recursion_limit below

NOTES_DIR = Path(__file__).parent / "agent_notes"

SYSTEM_PROMPT = """You are a careful problem-solving agent. Work step by step and use tools instead of guessing.

Rules:
- Never compute arithmetic yourself — always call calculator.
- Only call save_note when the user explicitly asks you to save or remember something.
- If a tool call fails, read the error and correct your input on the next try.
- Give a final answer as soon as you genuinely have enough information — don't call tools you don't need.
"""

# --------------------------------------------------------------------------
# calculator — identical sandboxed AST walk to ../03-small-agent/tools.py.
# This isn't part of what's being compared (that's the orchestration layer),
# so it's left untouched rather than swapped for e.g. simpleeval.
# --------------------------------------------------------------------------
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


@tool
def calculator(expression: str) -> float:
    """Evaluate a numeric arithmetic expression and return a float. Never compute totals mentally — always use this."""
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as e:
        raise ValueError(f"could not evaluate '{expression}': {e}") from e


@tool
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the live web and return up to 5 results (title, url, snippet)."""
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

</details>

This is deliberately incomplete — no `save_note`, no model wiring, no `main()` yet. Everything written here survives unchanged into the finished file.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `TypeError: 'StructuredTool' object is not callable` | Called `calculator(expression=...)` directly instead of `.invoke(...)` | Use `calculator.invoke({"expression": ...})` |
| Tool's description looks empty or wrong to the model later | Docstring missing, or written above the `@tool` decorator instead of inside the function body | Keep the docstring as the first statement inside the function |
| `max_results` doesn't behave as expected | Forgot the type hint / default (`max_results: int = 5`) | `@tool` derives the schema from the signature — an untyped or missing default changes what it infers |

Next: **[The save_note Closure](04-the-save-note-closure.md)** — the one tool where porting to `@tool` isn't a straightforward simplification.

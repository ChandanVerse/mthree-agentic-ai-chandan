# Step 3 — The save_note Closure

> [Back to index](README.md) · Previous: [Porting the Tools](03-porting-the-tools.md) · Next: [Assembling the Agent](05-assembling-the-agent.md)

## Goal

Port `save_note` — the one tool where the framework doesn't just remove code, it forces a different, weaker way of expressing a rule the hand-rolled dispatcher had a first-class home for.

## Why this matters

In [03-small-agent](../03-small-agent/), `save_note` is the app's one side-effecting tool — it writes a file, so it's classified in `SIDE_EFFECTING_TOOLS` and handled by a specific rule in `dispatch_tool_call()`: if it has already failed once this task, refuse to retry it, and stop the *entire agent run* by returning `retryable_by_model=False`. That rule exists because a retried "save" after an ambiguous failure could double-write a note if the first attempt partially landed — see [03-small-agent's dispatcher-retry-policy step](../03-small-agent-walkthrough/08-the-dispatcher-retry-policy.md) if you want the full reasoning. The rule works there because the controller owns a `tool_failure_counts: dict[str, int]`, threaded through every call, that it can check before dispatching and act on decisively — stop, don't retry.

`create_agent`'s loop has no such hook. There's no controller-level dict you get to inspect from outside the tool, and no first-class way to say "this specific tool call already failed once, refuse to run the rest of the graph." The state has to live *somewhere*, and the only place left to put it is inside the tool itself — which is exactly what a Python closure is for: a nested function that captures a variable from its enclosing scope and can mutate it across calls via `nonlocal`.

This gets you back *part* of the original rule: a second `save_note` call within the same run is refused rather than blindly retried, so the file-write side effect is still protected. But notice what it can't get back — refusing the second write doesn't halt anything else the model might do. In the hand-rolled version, a failed `save_note` stops the whole agent run, full stop. Here, the model could still call `calculator` next, or give a Final Answer, right after `save_note` refuses. If you need the original's "stop the whole run" behavior exactly, that requires a custom LangGraph node or conditional edge — which gives up the one-liner simplicity `create_agent` is supposed to buy you. Naming this gap precisely (not just "it's a bit different") is the actual point of this step.

## 1. Write the factory function

```python
def make_save_note_tool():
    """Build a save_note tool with its own per-run failure memory closed over
    it, so a second attempt within the same run is refused rather than
    blindly retried.

    Note 7 §6 / ../03-small-agent/agent.py's SIDE_EFFECTING_TOOLS rule: a
    side-effecting tool that already failed once must not be blindly
    retried — a retried "save" after an ambiguous failure could double-write
    a note if the first attempt partially landed. `create_react_agent`'s
    loop has no first-class concept of "this tool is unsafe to retry" the
    way the original's `dispatch_tool_call()` did, and there's no
    per-invocation state to hang that policy on outside the tool itself. So
    the policy moves into a closure: this object remembers whether *this
    run's* save_note has already failed once, and refuses to actually write
    again if so.

    This is weaker than the original in one real way: the original's
    dispatcher set `retryable_by_model=False` and stopped the *entire agent
    run* on that failure. Here, refusing the second write doesn't halt
    anything else the model might do — the model could still call a
    different tool or give a Final Answer. If you need "hard stop the whole
    run," that requires a custom conditional edge, not the prebuilt agent.
    """
    failed_once = False

    @tool
    def save_note(title: str, content: str) -> dict:
        """Save a short note to disk for later reference. This WRITES A FILE — a side effect, not a lookup."""
        nonlocal failed_once
        if failed_once:
            raise ValueError("'save_note' already failed once this task and is not safe to auto-retry.")
        try:
            NOTES_DIR.mkdir(exist_ok=True)
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in title.strip()) or "untitled"
            path = NOTES_DIR / f"{safe_name}.md"
            path.write_text(f"# {title}\n\n{content}\n")
            return {"saved_to": os.path.relpath(path)}
        except OSError as e:
            failed_once = True
            raise ValueError(f"could not save note: {e}") from e

    return save_note
```

Walk through why this is a *factory function* returning a tool, not just a tool: `failed_once` has to start fresh for every new agent run (a note failing in a previous goal shouldn't block saving in the next one — the hand-rolled version resets `tool_failure_counts = {}` at the top of every `run_agent()` call for the same reason). Calling `make_save_note_tool()` once per run and handing the fresh closure it returns to `create_agent` is what gives each run its own independent memory — you'll wire that up in Step 4's `build_agent()`.

Trace the two failure paths:

- **First failure** (`except OSError`): `failed_once` flips to `True`, and the error becomes a `ValueError` the model sees as a normal tool failure — it can still try again, once.
- **Second attempt** (`if failed_once`): refused immediately, before `NOTES_DIR.mkdir(...)` is even attempted — this is the no-blind-retry rule, now living entirely inside the tool rather than in a controller-level dispatcher.

## Try it

There's no easy way to force a live model into triggering a real `OSError` on demand, so demonstrate the closure directly instead — occupy `save_note`'s target path with a plain file first, so `NOTES_DIR.mkdir(exist_ok=True)` fails both times you'd expect it to behave differently:

```bash
uv run python -c "
import agent
open('agent_notes', 'w').close()  # a file, not a directory, blocks mkdir()
tool = agent.make_save_note_tool()
try:
    tool.invoke({'title': 't', 'content': 'c'})
except Exception as e:
    print('first call:', e)
try:
    tool.invoke({'title': 't2', 'content': 'c2'})
except Exception as e:
    print('second call:', e)
"
rm agent_notes
```

Expected output:

```text
first call: could not save note: [Errno 17] File exists: 'agent_notes'
second call: 'save_note' already failed once this task and is not safe to auto-retry.
```

The first call's message comes from inside the `except OSError` branch; the second comes from the `if failed_once` guard — proof the closure remembers across calls without any external dict. Point out to trainees that a **fresh** `make_save_note_tool()` call would reset that memory, which is exactly what happens once per agent run in Step 4.

## Checkpoint

<details>
<summary>Full <code>agent.py</code> (all three tools defined, no model wiring yet)</summary>

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


def make_save_note_tool():
    """Build a save_note tool with its own per-run failure memory closed over
    it, so a second attempt within the same run is refused rather than
    blindly retried.

    Note 7 §6 / ../03-small-agent/agent.py's SIDE_EFFECTING_TOOLS rule: a
    side-effecting tool that already failed once must not be blindly
    retried — a retried "save" after an ambiguous failure could double-write
    a note if the first attempt partially landed. `create_react_agent`'s
    loop has no first-class concept of "this tool is unsafe to retry" the
    way the original's `dispatch_tool_call()` did, and there's no
    per-invocation state to hang that policy on outside the tool itself. So
    the policy moves into a closure: this object remembers whether *this
    run's* save_note has already failed once, and refuses to actually write
    again if so.

    This is weaker than the original in one real way: the original's
    dispatcher set `retryable_by_model=False` and stopped the *entire agent
    run* on that failure. Here, refusing the second write doesn't halt
    anything else the model might do — the model could still call a
    different tool or give a Final Answer. If you need "hard stop the whole
    run," that requires a custom conditional edge, not the prebuilt agent.
    """
    failed_once = False

    @tool
    def save_note(title: str, content: str) -> dict:
        """Save a short note to disk for later reference. This WRITES A FILE — a side effect, not a lookup."""
        nonlocal failed_once
        if failed_once:
            raise ValueError("'save_note' already failed once this task and is not safe to auto-retry.")
        try:
            NOTES_DIR.mkdir(exist_ok=True)
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in title.strip()) or "untitled"
            path = NOTES_DIR / f"{safe_name}.md"
            path.write_text(f"# {title}\n\n{content}\n")
            return {"saved_to": os.path.relpath(path)}
        except OSError as e:
            failed_once = True
            raise ValueError(f"could not save note: {e}") from e

    return save_note
```

</details>

Still deliberately incomplete — no model wiring, no `main()`. Everything written here survives unchanged into the finished file.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `failed_once` never actually blocks a second call | Reassigned `failed_once = True` inside the `try` without `nonlocal`, creating a new local variable instead of mutating the enclosing one | `nonlocal failed_once` must be the first line inside `save_note` |
| Every new agent run still refuses to save | Called `make_save_note_tool()` once at import time and reused the same closure across runs | Call it fresh inside `build_agent()` (Step 4) so each run gets its own `failed_once` |
| Expecting the model to be stopped entirely after one `save_note` failure | Assuming this closure behaves exactly like the hand-rolled dispatcher's `retryable_by_model=False` | It doesn't — re-read the "Why this matters" section above; the model can still act after the refusal |

Next: **[Assembling the Agent](05-assembling-the-agent.md)** — wire all three tools and the model together with `create_agent`.

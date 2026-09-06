# Step 4 — Assembling the Agent

> [Back to index](README.md) · Previous: [The save_note Closure](04-the-save-note-closure.md) · Next: [Running the Graph and Guardrails](06-running-the-graph-and-guardrails.md)

## Goal

Wire the model and all three tools together with `create_agent`, and watch the entire hand-rolled loop — `call_model`, `parse_reply`, `dispatch_tool_call`, and `run_agent`'s `for` loop — disappear into one function call.

## Why this matters

This is the step the whole app exists to demonstrate. In [03-small-agent](../03-small-agent/), driving one ReAct step takes four cooperating pieces: `call_model()` to make the API call, `parse_reply()` to regex apart whatever came back, `dispatch_tool_call()` to validate and execute a call, and `run_agent()`'s `for step_num in range(1, max_steps + 1)` to repeat all of that until a `Final Answer` shows up. `create_agent(llm, tools, system_prompt=...)` *is* that loop, already built — the `CompiledStateGraph` it returns knows how to call the model, notice `tool_calls` on the reply, run them, append the results, and loop again, entirely on its own. You get a `.invoke()` method back, not a loop you drive by hand.

Naming note, worth calling out to trainees directly: older LangGraph tutorials point at `langgraph.prebuilt.create_react_agent`. In the versions this app is pinned to (`langgraph` 1.2.11 / `langchain` 1.4.0), that import still works but prints a deprecation warning — the prebuilt agent has moved to `langchain.agents.create_agent`. It's still a LangGraph `CompiledStateGraph` under the hood; only the import path and the `system_prompt=` keyword changed. If you're following older material, expect this drift.

## 1. Add the new imports

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
```

## 2. Write `build_agent()`

```python
def build_agent(model: str, base_url: str):
    """Fresh LLM + fresh tools (including a fresh save_note failure-memory
    closure) per call — mirrors the original's `tool_failure_counts = {}`
    being reset inside every `run_agent()` call, i.e. no memory carries
    across goals.
    """
    llm = ChatOpenAI(model=model, base_url=base_url, api_key="not-needed", temperature=0)
    tools = [calculator, web_search, make_save_note_tool()]
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
```

Two details worth pointing out, both direct echoes of decisions already made earlier in the series:

- `temperature=0` and `api_key="not-needed"` are the same Docker Model Runner workarounds every app in this series uses — the client library requires an API key even though the local endpoint doesn't check it.
- `make_save_note_tool()` is called **inside** `build_agent()`, not at module level. That's what gives every call to `build_agent()` — and therefore every fresh goal — its own independent `failed_once` closure, exactly mirroring the hand-rolled version's `tool_failure_counts = {}` being reset at the top of every `run_agent()` call. Get this wrong (call it once at import time and reuse the tool) and a `save_note` failure on one goal would incorrectly block saving on the *next* goal too.

## Try it

There's no CLI yet, so invoke the compiled graph directly to prove the round trip works — this mirrors how [02-chat-with-web-search-langchain](../02-chat-with-web-search-langchain-walkthrough/04-binding-tools-and-raw-decisions.md) inspected a raw decision before wiring in the rest of the app:

```bash
uv run python -c "
from langchain_core.messages import HumanMessage
from agent import build_agent, DEFAULT_MODEL, DEFAULT_BASE_URL

graph = build_agent(DEFAULT_MODEL, DEFAULT_BASE_URL)
result = graph.invoke({'messages': [HumanMessage(content='What is 12 * 7?')]})
for m in result['messages']:
    print(type(m).__name__, '->', getattr(m, 'tool_calls', None) or m.content)
"
```

Expected shape (your model's exact wording will differ):

```text
HumanMessage -> What is 12 * 7?
AIMessage -> [{'name': 'calculator', 'args': {'expression': '12 * 7'}, 'id': '...', 'type': 'tool_call'}]
ToolMessage -> 84.0
AIMessage -> 12 * 7 is 84.
```

Four messages, no code of yours in between them — `create_agent` called the model, saw the `tool_calls`, ran `calculator.invoke(...)`, appended a `ToolMessage`, and called the model again, all inside `.invoke()`. That entire round trip is what `dispatch_tool_call` plus the hand-rolled `run_agent` loop used to do by hand.

## Checkpoint

<details>
<summary>Full <code>agent.py</code> (agent assembled, no run loop/CLI yet)</summary>

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
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

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


def build_agent(model: str, base_url: str):
    """Fresh LLM + fresh tools (including a fresh save_note failure-memory
    closure) per call — mirrors the original's `tool_failure_counts = {}`
    being reset inside every `run_agent()` call, i.e. no memory carries
    across goals.
    """
    llm = ChatOpenAI(model=model, base_url=base_url, api_key="not-needed", temperature=0)
    tools = [calculator, web_search, make_save_note_tool()]
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
```

</details>

Still deliberately incomplete — no `run_agent()`, no guardrail, no `main()` yet.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `save_note` stays refused across unrelated goals | `make_save_note_tool()` called once at module level and reused | Call it fresh inside `build_agent()`, so each call gets an independent closure |
| `LangGraphDeprecatedSinceV10` warning on import | Copied `from langgraph.prebuilt import create_react_agent` from older docs instead of `from langchain.agents import create_agent` | Use the import shown above — same object, current home |
| `openai.AuthenticationError` | Forgot `api_key="not-needed"` on `ChatOpenAI(...)` | Docker Model Runner doesn't check the key, but the client still requires the argument |

Next: **[Running the Graph and Guardrails](06-running-the-graph-and-guardrails.md)** — wrap `.invoke()` in a step-limit guardrail and a readable trace.

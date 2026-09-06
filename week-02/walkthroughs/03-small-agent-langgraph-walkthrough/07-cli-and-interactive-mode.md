# Step 6 — CLI and Interactive Mode

> [Back to index](README.md) · Previous: [Running the Graph and Guardrails](06-running-the-graph-and-guardrails.md) · Next: [Recap and Exercises](08-recap-and-exercises.md)

## Goal

Replace the scratch `python -c` snippets with the real `argparse` flags, one-shot and interactive modes, and connection-error handling — finishing the file.

## Why this matters

There's genuinely nothing new to teach here — this is the payoff of having built the agent mechanics first. `argparse`, the `try`/`except (EOFError, KeyboardInterrupt)` input guard, and the `except APIError` connection handling are exactly the same patterns as the hand-rolled [03-small-agent](../03-small-agent/)'s `main()`, which itself reused the pattern from [01-basic-chat-app](../01-basic-chat-app-walkthrough/README.md) and [02-chat-with-web-search](../02-chat-with-web-search-walkthrough/README.md). The only thing worth pausing on: `run_one()` calls `build_agent()` **fresh on every goal**, not once outside the loop — that's what gives every goal its own independent `save_note` closure, the same "no memory carries across goals" contract the hand-rolled version keeps by resetting `tool_failure_counts = {}` inside every `run_agent()` call.

## 1. Add the remaining imports

```python
import argparse
import sys

from openai import APIError
```

`APIError` comes from the `openai` package directly (a transitive dependency of `langchain-openai`) — LangChain doesn't wrap it in its own exception type, so the `except` clause below looks identical to the hand-rolled version's.

## 2. Write `main()`

```python
def main() -> None:
    parser = argparse.ArgumentParser(
        description="A small ReAct agent (LangGraph variant) with calculator, web_search, and save_note tools."
    )
    parser.add_argument("goal", nargs="?", help="The task to hand the agent. Omit for interactive mode.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help=f"Step-limit guardrail (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument("--quiet", action="store_true", help="Only print the final answer, not the step trace")
    args = parser.parse_args()

    def run_one(goal: str) -> None:
        graph = build_agent(args.model, args.base_url)
        try:
            answer = run_agent(graph, goal, max_steps=args.max_steps, verbose=not args.quiet)
        except APIError as e:
            print(f"[error] {e}. Is Docker Model Runner running? See ../00-local-model-setup/README.md")
            sys.exit(1)
        print(f"\nFinal Answer: {answer}")

    if args.goal:
        run_one(args.goal)
        return

    print(f"Small agent ready ({args.model}, LangGraph). Each line is a fresh goal — no memory carries over between them. Type 'exit' to quit.\n")
    while True:
        try:
            goal = input("Goal: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if goal.lower() in {"exit", "quit"}:
            break
        if not goal:
            continue
        run_one(goal)
        print()


if __name__ == "__main__":
    main()
```

Notice `build_agent(args.model, args.base_url)` is called **inside** `run_one()`, which itself is called once per goal — both in one-shot mode (once) and in interactive mode (once per line typed). That's the concrete mechanism behind "no memory carries over between goals": every call gets a brand-new `ChatOpenAI` instance and a brand-new `make_save_note_tool()` closure, so a `save_note` failure on one goal can never block a save on the next one.

## Try it

```bash
uv run agent.py "A store sells 17 units at \$12.99 each with 8% discount and 6% tax on the discounted amount. Is the total under \$216?"
```

```text
[action] calculator({'expression': '17 * 12.99'})
[observation] 220.83
[action] calculator({'expression': '220.83 * 0.92'})
[observation] 203.16360000000003
[action] calculator({'expression': '203.16360000000003 * 1.06'})
[observation] 215.35341600000004

Final Answer: The total cost is approximately $215.35.

Since $215.35 is less than $216, the total is under $216.
```

Then try interactive mode, the quiet flag, and the error path — same three checks as every other app in this series:

```bash
uv run agent.py                                       # interactive: one goal per line
uv run agent.py --quiet "What's 2^10?"                # only the final answer
uv run agent.py --base-url http://localhost:9999/v1 "test"   # should fail fast, not traceback
```

The last command should print a clean `[error]` line pointing at [00-local-model-setup](../00-local-model-setup/README.md), not a raw stack trace.

## Checkpoint

<details>
<summary>Full <code>agent.py</code> (final)</summary>

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
import argparse
import ast
import operator
import os
import sys
from pathlib import Path

from ddgs import DDGS
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from openai import APIError

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


def print_trace(messages: list) -> None:
    """Print whatever the model actually produced en route to a Final
    Answer. Unlike the original, which *forced* an explicit "Thought: ..."
    narration every step (a deliberate auditability choice), native
    tool-calling doesn't require the model to narrate anything — you get
    free text only if the model chose to include some alongside a tool
    call, which can be terse or empty.
    """
    for message in messages[1:-1]:  # skip the initial goal and the final answer
        if isinstance(message, AIMessage):
            if message.content:
                print(f"[assistant] {message.content}")
            for call in message.tool_calls:
                print(f"[action] {call['name']}({call['args']})")
        elif isinstance(message, ToolMessage):
            print(f"[observation] {message.content}")


def run_agent(graph, goal: str, max_steps: int = DEFAULT_MAX_STEPS, verbose: bool = True) -> str:
    # LangGraph's recursion_limit counts graph super-steps (roughly: one for
    # the agent node, one for the tool node, per tool-calling round), not
    # "ReAct steps" the way the original's max_steps counted Thought/Action
    # cycles — this is an approximation of the same guardrail, not an exact
    # equivalent.
    config = {"recursion_limit": max_steps * 2 + 1}
    try:
        result = graph.invoke({"messages": [HumanMessage(content=goal)]}, config=config)
    except GraphRecursionError:
        return f"Stopped: exceeded max_steps ({max_steps}) without reaching a final answer."

    messages = result["messages"]
    if verbose:
        print_trace(messages)
    return messages[-1].content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A small ReAct agent (LangGraph variant) with calculator, web_search, and save_note tools."
    )
    parser.add_argument("goal", nargs="?", help="The task to hand the agent. Omit for interactive mode.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help=f"Step-limit guardrail (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument("--quiet", action="store_true", help="Only print the final answer, not the step trace")
    args = parser.parse_args()

    def run_one(goal: str) -> None:
        graph = build_agent(args.model, args.base_url)
        try:
            answer = run_agent(graph, goal, max_steps=args.max_steps, verbose=not args.quiet)
        except APIError as e:
            print(f"[error] {e}. Is Docker Model Runner running? See ../00-local-model-setup/README.md")
            sys.exit(1)
        print(f"\nFinal Answer: {answer}")

    if args.goal:
        run_one(args.goal)
        return

    print(f"Small agent ready ({args.model}, LangGraph). Each line is a fresh goal — no memory carries over between them. Type 'exit' to quit.\n")
    while True:
        try:
            goal = input("Goal: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if goal.lower() in {"exit", "quit"}:
            break
        if not goal:
            continue
        run_one(goal)
        print()


if __name__ == "__main__":
    main()
```

</details>

This matches [../03-small-agent-langgraph/agent.py](../03-small-agent-langgraph/agent.py) exactly.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `save_note` failing on one goal blocks it on later goals in the same interactive session | `build_agent()` (or `make_save_note_tool()` alone) hoisted outside `run_one()`/the `while True` loop | `build_agent(args.model, args.base_url)` must be called inside `run_one()`, once per goal |
| `NameError: name 'args' is not defined` inside `run_one` | Tried to reference `args` somewhere it isn't in scope | `run_one` is a nested function defined inside `main()` specifically so it can close over `args` — keep it nested |
| `pydantic`/`langchain` import errors | Mixed environments — a stray `VIRTUAL_ENV` pointing at a different project's `.venv` | Always launch via `uv run agent.py` from inside this project's directory |

Next: **[Recap and Exercises](08-recap-and-exercises.md)**.

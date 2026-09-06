# Step 5 — Running the Graph and Guardrails

> [Back to index](README.md) · Previous: [Assembling the Agent](05-assembling-the-agent.md) · Next: [CLI and Interactive Mode](07-cli-and-interactive-mode.md)

## Goal

Wrap `graph.invoke()` in a readable trace printer and a step-limit guardrail, so a model that never converges stops cleanly instead of running forever.

## Why this matters

Step 4's scratch test worked because the model reached a `Final Answer` in one tool hop. Nothing so far stops it from *not* doing that — a model can get stuck calling tools in a cycle, or keep asking for slightly different variations of the same calculation without ever concluding. The hand-rolled version's `run_agent()` guards against exactly this with `max_steps`, checked directly inside its `for step_num in range(1, max_steps + 1)` loop: the moment the counter runs out, it returns a clean "Stopped: exceeded max_steps" message instead of hanging.

`create_agent`'s loop is inside the compiled graph, so you can't check a Python `for` loop's counter from outside it — but LangGraph has its own version of the same idea: `recursion_limit`, a `config` key bounding how many graph super-steps `.invoke()` will run before raising `GraphRecursionError`. The mapping is not exact, though, and getting this wrong is the single easiest way to end up debugging a guardrail that fires at the wrong time. `recursion_limit` counts graph super-steps — roughly one for the agent node's model call, one for the tool-execution node, per round of tool-calling — not "ReAct steps" the way `max_steps` counted Thought/Action cycles directly. `run_agent()` below passes `max_steps * 2 + 1` as an approximation of the same guardrail, not a translation of it. If you needed the cap to trigger at a precise number of tool calls, you'd have to count `tool_calls` across the returned message list yourself rather than trust `recursion_limit` to mean the same thing.

## 1. Add the new imports

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
```

## 2. Write `print_trace()`

```python
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
```

Compare this to the hand-rolled version's trace: that app *forced* an explicit `Thought: ...` line every step because the whole point of the text-protocol format was auditability — the model had no choice but to narrate its reasoning before acting. Native tool-calling doesn't require that. `print_trace()` prints whatever free text the model happened to attach alongside a `tool_calls` list, which the model might leave empty. You still get real tool calls and real observations in the trace; you just don't get a guaranteed reasoning line before each one.

## 3. Write `run_agent()`

```python
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
```

`messages[-1].content` plays the same role as the hand-rolled version's `parsed.final_answer` — the last message in a graph run that exited normally (as opposed to raising `GraphRecursionError`) is always the model's final `AIMessage` with no more `tool_calls`, since that's the graph's own exit condition.

## Try it

```bash
uv run python -c "
from agent import build_agent, run_agent, DEFAULT_MODEL, DEFAULT_BASE_URL

graph = build_agent(DEFAULT_MODEL, DEFAULT_BASE_URL)
answer = run_agent(graph, 'What is 17 * 12.99, and is that under 220?')
print(f'\nFinal Answer: {answer}')
"
```

Expected shape (exact wording varies by model):

```text
[action] calculator({'expression': '17 * 12.99'})
[observation] 220.83

Final Answer: 17 * 12.99 is 220.83, which is just over 220, not under it.
```

Then see the guardrail actually fire by setting an unreasonably low cap:

```bash
uv run python -c "
from agent import build_agent, run_agent, DEFAULT_MODEL, DEFAULT_BASE_URL

graph = build_agent(DEFAULT_MODEL, DEFAULT_BASE_URL)
answer = run_agent(graph, 'Search for three different facts about Mars, one at a time.', max_steps=1)
print(answer)
"
```

```text
Stopped: exceeded max_steps (1) without reaching a final answer.
```

With `max_steps=1`, `recursion_limit` is `1 * 2 + 1 = 3` — barely enough for one model call and one tool call, not enough to reach a `Final Answer` on a multi-fact goal, so `GraphRecursionError` fires and the fallback message returns exactly like the hand-rolled version's.

## Checkpoint

<details>
<summary>Full <code>agent.py</code> (guardrail in place, no CLI yet)</summary>

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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

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
```

</details>

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| Guardrail fires much sooner than expected | Passed `max_steps` straight through as `recursion_limit` instead of `max_steps * 2 + 1` | Remember `recursion_limit` counts graph super-steps, not tool calls — roughly double |
| `print_trace` shows nothing for a step that clearly called a tool | Iterated `messages` instead of `messages[1:-1]`, printing the final answer's own (empty) tool_calls, or missed that `message.content` can legitimately be empty | Empty `content` alongside a populated `tool_calls` list is normal — the model isn't required to narrate |
| `NameError: name 'GraphRecursionError' is not defined` | Imported it from `langgraph.prebuilt` or forgot the import entirely | It lives in `langgraph.errors` |

Next: **[CLI and Interactive Mode](07-cli-and-interactive-mode.md)** — wrap this in the same `argparse`/REPL pattern from the rest of the series.

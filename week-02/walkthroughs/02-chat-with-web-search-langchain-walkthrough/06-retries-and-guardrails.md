# Step 5 — Retries and Guardrails

> [Back to index](README.md) · Previous: [Executing the Tool Call](05-executing-the-tool-call.md) · Next: [Assembling the CLI and Loop](07-assembling-the-cli-and-loop.md)

## Goal

Wrap `run_turn()` in a bounded retry loop, and guard against a tool name the model got wrong or a tool call that raises — matching the hand-rolled version's self-correction behavior.

## Why this matters

This is the step that pushes back on the framing "LangChain removes the hand-rolled logic." It removes the *parsing* half — `bind_tools()` validates argument shape before the call ever reaches your code, so a genuinely malformed call is now rare. It does **not** remove the need for bounded retries: a model can still say a tool name that isn't registered (`bind_tools()` constrains what the model *should* call, not what a confused model actually emits), and a tool call with syntactically valid arguments can still fail at runtime — a network error inside `web_search`, for instance. Both are still your problem, same as they were in the hand-rolled version's `resolve_tool_call`, and for the same reason: an API-level schema can validate *shape*, not *correctness* or *success*.

## 1. Add the retry constant

Insert this between `SYSTEM_PROMPT` and the `web_search` tool definition:

```python
# Note 7 §7's dispatcher still applies conceptually — we still cap retries
# so a model that keeps mis-calling the tool doesn't loop forever — but the
# *parsing* half of the dispatcher (malformed JSON, hallucinated tool names)
# is now LangChain's problem: bind_tools() validates the call against the
# tool's schema before it ever reaches us.
MAX_TOOL_RETRIES = 2
```

## 2. Rewrite `run_turn()` with the retry loop

Replace the Step 4 version with:

```python
def run_turn(llm_with_tools, messages: list) -> str:
    """Handle one user turn: at most one tool hop, then a final answer.

    This is the entire replacement for the hand-rolled version's
    try_parse_action/resolve_tool_call/run_turn trio: `response.tool_calls`
    arrives already parsed and validated, so there's nothing left to regex.
    """
    for _ in range(MAX_TOOL_RETRIES + 1):
        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            return response.content  # model chose to answer directly

        messages.append(response)
        retry_needed = False
        for call in response.tool_calls:
            tool_fn = TOOLS_BY_NAME.get(call["name"])
            if tool_fn is None:
                # Still worth guarding — bind_tools() constrains what the
                # model *should* call, not what a confused model actually
                # emits.
                content = f"Unknown tool '{call['name']}'. Available: {list(TOOLS_BY_NAME)}."
                retry_needed = True
            else:
                try:
                    content = str(tool_fn.invoke(call["args"]))
                except Exception as e:  # tool-side failure, not a parsing failure
                    content = f"Tool '{call['name']}' failed: {e}"
                    retry_needed = True
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

        if not retry_needed:
            return llm_with_tools.invoke(messages).content

    return "I couldn't complete that tool call correctly after a few tries — could you rephrase?"
```

What changed from Step 4, and why each change exists:

- `TOOLS_BY_NAME.get(call["name"])` instead of `TOOLS_BY_NAME[call["name"]]` — a missing key now becomes a `ToolMessage` telling the model what went wrong, not a Python `KeyError` crashing the app. The message is fed back to the model as an error, exactly like the hand-rolled `resolve_tool_call`'s "Unknown tool" message — just without the `difflib` fuzzy-match suggestion, since a confused model choosing from an explicit tool list is a rarer failure mode to begin with.
- `try/except Exception` around `tool_fn.invoke(...)` — a tool can fail for reasons that have nothing to do with what the model asked for (a network blip inside `web_search`, say). That failure becomes tool output the model can react to, not an unhandled exception.
- `retry_needed` — tracked per turn, across *all* calls the model made this turn, not per call. If any call in the batch had a problem, the whole turn loops back to `llm_with_tools.invoke(messages)` again, now with the error `ToolMessage`(s) in context, giving the model a chance to self-correct.
- The outer `for _ in range(MAX_TOOL_RETRIES + 1)` — bounds how many times that self-correction can happen before giving up with the same fallback message the hand-rolled version uses.

## Try it

There's no easy way to force a live model into calling an unknown tool or a failing one on demand — this is best demonstrated by reading the code path with trainees rather than triggering it live. If you want to see it fire, temporarily rename `TOOLS_BY_NAME`'s key after construction (e.g. add a line `TOOLS_BY_NAME["web_search_typo"] = TOOLS_BY_NAME.pop("web_search")`) and ask a question that needs search — the model will call `web_search` (the name it was told about via `bind_tools()`), `TOOLS_BY_NAME.get("web_search")` will now return `None`, and you'll see the "Unknown tool" retry path fire before the final fallback message. Remove that line afterward.

## Checkpoint

<details>
<summary>Full <code>chat_with_tools.py</code> (guardrails in place, temporary <code>main()</code>)</summary>

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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a web_search tool. "
    "Only call it when you genuinely need current or external information — "
    "not for things you already know."
)

# Note 7 §7's dispatcher still applies conceptually — we still cap retries
# so a model that keeps mis-calling the tool doesn't loop forever — but the
# *parsing* half of the dispatcher (malformed JSON, hallucinated tool names)
# is now LangChain's problem: bind_tools() validates the call against the
# tool's schema before it ever reaches us.
MAX_TOOL_RETRIES = 2


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


def run_turn(llm_with_tools, messages: list) -> str:
    """Handle one user turn: at most one tool hop, then a final answer.

    This is the entire replacement for the hand-rolled version's
    try_parse_action/resolve_tool_call/run_turn trio: `response.tool_calls`
    arrives already parsed and validated, so there's nothing left to regex.
    """
    for _ in range(MAX_TOOL_RETRIES + 1):
        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            return response.content  # model chose to answer directly

        messages.append(response)
        retry_needed = False
        for call in response.tool_calls:
            tool_fn = TOOLS_BY_NAME.get(call["name"])
            if tool_fn is None:
                # Still worth guarding — bind_tools() constrains what the
                # model *should* call, not what a confused model actually
                # emits.
                content = f"Unknown tool '{call['name']}'. Available: {list(TOOLS_BY_NAME)}."
                retry_needed = True
            else:
                try:
                    content = str(tool_fn.invoke(call["args"]))
                except Exception as e:  # tool-side failure, not a parsing failure
                    content = f"Tool '{call['name']}' failed: {e}"
                    retry_needed = True
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

        if not retry_needed:
            return llm_with_tools.invoke(messages).content

    return "I couldn't complete that tool call correctly after a few tries — could you rephrase?"


def main() -> None:
    llm = ChatOpenAI(model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, api_key="not-needed", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    messages: list = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="Who won the most recent F1 world championship?"),
    ]
    answer = run_turn(llm_with_tools, messages)
    print(f"Assistant: {answer}")


if __name__ == "__main__":
    main()
```

</details>

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `retry_needed` reset inside the `for call in response.tool_calls` loop | Placed the `retry_needed = False` inside the inner loop instead of once per outer iteration | It must be set once before the inner `for`, so it reflects the whole turn's calls, not just the last one |
| Loop never terminates on a persistently broken tool | `range(MAX_TOOL_RETRIES + 1)` changed to an unbounded `while True` | Keep the bounded `range(...)` — this is exactly the guardrail the hand-rolled version also relies on |
| Error `ToolMessage` never reaches the model | `continue`/early return added inside the `for call` loop, skipping the `messages.append(ToolMessage(...))` | Every call, successful or not, must produce exactly one `ToolMessage` |

Next: **[Assembling the CLI and Loop](07-assembling-the-cli-and-loop.md)** — wrap this in the same REPL loop and CLI flags pattern from the rest of the series.

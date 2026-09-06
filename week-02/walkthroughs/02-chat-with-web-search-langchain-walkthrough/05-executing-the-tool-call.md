# Step 4 — Executing the Tool Call

> [Back to index](README.md) · Previous: [Binding Tools and Raw Decisions](04-binding-tools-and-raw-decisions.md) · Next: [Retries and Guardrails](06-retries-and-guardrails.md)

## Goal

Turn the raw decision from Step 3 into an actual round trip: run the tool the model asked for, and feed the result back so the model can use it in a real answer.

## Why this matters

This is the direct replacement for the hand-rolled version's `resolve_tool_call` plus the `Observation: ...`-message half of `run_turn()`. The mechanics are the same idea either way — execute, then tell the model what happened — but the *shape* of "telling the model what happened" is different and matters: the hand-rolled version stuffs the result into a `user`-role message, which works but is a slight lie about who's "speaking." A `ToolMessage` is the structurally correct chat-API representation of "this is what tool call `X` returned," and it's tied to the specific call via `tool_call_id` rather than positional ordering. With only one tool and one call per turn that distinction is invisible; it stops being invisible the moment a model requests more than one tool call in a single turn.

## 1. Add the message types you need

Update the `langchain_core.messages` import:

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
```

(`AIMessage` isn't used directly yet in this step — `response` from `llm_with_tools.invoke()` already *is* an `AIMessage`. It's imported now because `main()` will need it in Step 6.)

## 2. Write `run_turn()`

Add this above `main()`, replacing the inline loop-and-print scratch code from Step 3:

```python
def run_turn(llm_with_tools, messages: list) -> str:
    """Handle one user turn: at most one tool hop, then a final answer."""
    response = llm_with_tools.invoke(messages)

    if not response.tool_calls:
        return response.content  # model chose to answer directly

    messages.append(response)
    for call in response.tool_calls:
        tool_fn = TOOLS_BY_NAME[call["name"]]
        content = str(tool_fn.invoke(call["args"]))
        messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

    return llm_with_tools.invoke(messages).content
```

Walk through the order with trainees, the same way [conversation memory in the basic chat app](../01-basic-chat-app-walkthrough/05-conversation-memory.md) called out append order:

1. `messages.append(response)` — the model's own turn, tool calls and all, goes into the transcript *before* any tool result. The chat API expects an assistant message with `tool_calls` to precede the `tool` messages that answer them.
2. `tool_fn.invoke(call["args"])` — `call["args"]` is already a parsed dict (Step 3), so this is a direct call, no `json.loads()` in sight.
3. `str(...)` — tool results here are Python dicts; `ToolMessage.content` expects text, so it's stringified before wrapping.
4. `ToolMessage(content=content, tool_call_id=call["id"])` — the `tool_call_id` is what lets the model (and the API) match this result back to the specific call that asked for it.
5. A second `llm_with_tools.invoke(messages)` — now that the transcript includes the tool's result, ask the model again for its actual answer.

## 3. Use it from `main()`

Replace the Step 3 scratch loop with a single real turn:

```python
def main() -> None:
    llm = ChatOpenAI(model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, api_key="not-needed", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    messages: list = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="Who won the most recent F1 world championship?"),
    ]
    answer = run_turn(llm_with_tools, messages)
    print(f"Assistant: {answer}")
```

Still not the finished CLI/REPL loop — that's Step 6. This is scratch code to prove the round trip works end to end for one hardcoded question.

## Try it

```bash
uv run chat_with_tools.py
```

Expected shape of the output (exact wording depends on your model and a live search, so don't expect it verbatim):

```text
Assistant: The most recent F1 World Drivers' Champion is Max Verstappen.
```

If instead you get an empty or oddly truncated answer, that's the flakiness this variant's README already warns about — small local models aren't guaranteed to use a `ToolMessage` result well every time. It's worth showing trainees this happening live rather than treating it as something broken in the walkthrough.

## Checkpoint

<details>
<summary>Full <code>chat_with_tools.py</code> (single tool hop, temporary <code>main()</code>)</summary>

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
    """Handle one user turn: at most one tool hop, then a final answer."""
    response = llm_with_tools.invoke(messages)

    if not response.tool_calls:
        return response.content  # model chose to answer directly

    messages.append(response)
    for call in response.tool_calls:
        tool_fn = TOOLS_BY_NAME[call["name"]]
        content = str(tool_fn.invoke(call["args"]))
        messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

    return llm_with_tools.invoke(messages).content


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
| `KeyError` on `TOOLS_BY_NAME[call["name"]]` | Model called a tool name that isn't registered | Expected to be possible in principle — Step 5 replaces this with a guarded `.get()` instead of `[]` |
| API error about message ordering / missing tool response | `ToolMessage` appended without the preceding `messages.append(response)` | The `AIMessage` with `tool_calls` must be in the transcript before its `ToolMessage`s |
| Model's final answer ignores the search result entirely | Sometimes just model behavior — not a bug in this code | Try rephrasing, or compare against the hand-rolled version's `Observation:`-based prompt on the same question |

Next: **[Retries and Guardrails](06-retries-and-guardrails.md)** — the one piece of hand-rolled control flow that survives this rebuild.

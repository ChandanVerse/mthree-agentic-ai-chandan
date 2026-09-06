# Step 6 — The Round Trip

> [Back to index](README.md) · Previous: [Validating and Executing](06-validating-and-executing.md) · Next: [Bounded Retries](08-bounded-retries.md)

## Goal

Assemble `try_parse_action` and `resolve_tool_call` into `run_turn()`: the function that handles one full user turn, executing a tool if the model asked for one and getting a final answer either way.

## Why this matters

This is the step where the pieces actually become a round trip instead of a collection of separately-tested functions. The order matters and is worth walking through explicitly: the model's own `ACTION:` reply has to go into the transcript as an `assistant` message *before* the observation is appended, otherwise the conversation history doesn't reflect what actually happened, and the model's follow-up call would be reasoning from an incomplete transcript. Only after both of those are appended does the second `call_model()` happen — asking the model to answer using information that, as far as the transcript shows, it just retrieved itself.

This version doesn't yet retry a failed tool call — that's Step 7. For now, a validation or lookup error just becomes the final answer, so you can see the success path work cleanly before adding the complexity of self-correction on top of it.

## 1. `run_turn()` — success path first

Add this function after `resolve_tool_call()`:

```python
def run_turn(client: OpenAI, model: str, messages: list[dict]) -> str:
    """Handle one user turn: at most one tool hop, then a final answer."""
    reply = call_model(client, model, messages)
    tool_name, raw_args = try_parse_action(reply)

    if tool_name is None:
        return reply  # model chose to answer directly — no tool needed

    result, error = resolve_tool_call(tool_name, raw_args)
    if error:
        return f"Tool call failed: {error}"  # Step 7 replaces this with a retry

    messages.append({"role": "assistant", "content": reply})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Observation from {tool_name}: {json.dumps(result)}\n\n"
                "Now answer the original question using this observation. Do not call another tool."
            ),
        }
    )
    return call_model(client, model, messages)
```

Note what's fed back as the observation: `json.dumps(result)` — the *entire* bounded result dict from Step 2, `has_more`/`total` included, not just the raw list of hits. The model sees the same "this might not be everything" signal a developer reading the dict would.

## 2. Scratch code: run one full turn end to end

Replace the temporary `main()` from Step 3 with:

```python
def main() -> None:
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key="not-needed")
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What's the latest stable version of Python?"},
    ]
    answer = run_turn(client, DEFAULT_MODEL, messages)
    print(f"Assistant: {answer}")


if __name__ == "__main__":
    main()
```

Still not the finished CLI/REPL loop — that's Step 8. This is scratch code to prove the whole round trip works for one hardcoded question that needs the tool.

## Try it

```bash
uv run chat_with_tools.py
```

Expected shape of the output (exact wording depends on your model and a live search — don't expect it verbatim):

```text
Assistant: Python 3.13 is the latest stable release.
```

If the answer looks like it ignored the search result, or if the model repeats another `ACTION:` line instead of answering, that's a real reliability gap in small local models, not a bug in this code — it's exactly what Step 7's retry loop and the "Do not call another tool" instruction in the observation message exist to reduce.

## Checkpoint

<details>
<summary>Full <code>chat_with_tools.py</code> (single tool hop, temporary <code>main()</code>)</summary>

```python
#!/usr/bin/env python3
"""Step 2 — a chat app augmented with one tool (web search) the model may
choose to use.

Run:
    uv run chat_with_tools.py
"""
import json
import os
import re
from difflib import get_close_matches

from openai import APIError, OpenAI
from pydantic import ValidationError

from tools import TOOLS, tools_prompt_block

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")

# The text protocol a model must follow to call a tool: a line reading
# "ACTION: <tool_name> {json args}". Matching the tool name and the JSON
# blob as two separate steps (below) lets us tell "no tool call at all"
# apart from "tried to call a tool but the JSON is truncated/broken".
ACTION_NAME_PATTERN = re.compile(r"ACTION:\s*(\w+)", re.DOTALL)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = f"""You are a helpful assistant with access to one tool.

{tools_prompt_block()}

To call a tool, reply with ONLY this line (no other text):
ACTION: web_search {{"query": "your search query"}}

If you don't need the tool, just answer the user's question directly in plain text.
Only call the tool when you genuinely need current or external information —
not for things you already know.
"""


def call_model(client: OpenAI, model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def try_parse_action(text: str):
    """Extract a tool call from a model reply.

    Returns one of:
    - (None, None)               the model didn't try to call a tool
    - (tool_name, args_dict)     a syntactically valid call
    - (tool_name, raw_str)       the model *tried* to call a tool, but no
                                  parseable JSON object followed — covers
                                  both a truncated call (no closing brace)
                                  and one with broken JSON syntax — distinct
                                  from "no call" so the caller can react.
    """
    name_match = ACTION_NAME_PATTERN.search(text)
    if not name_match:
        return None, None

    tool_name = name_match.group(1)
    remainder = text[name_match.end() :]
    json_match = JSON_OBJECT_PATTERN.search(remainder)
    if not json_match:
        return tool_name, remainder.strip()

    try:
        return tool_name, json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return tool_name, json_match.group(0)


def resolve_tool_call(tool_name: str, raw_args) -> tuple[dict | None, str | None]:
    """Validate and execute one tool call.

    Returns (result, error_message) — exactly one is set. An error is
    handed back to the model as the *exact* validation/lookup problem,
    rather than crashing the app or silently guessing.
    """
    if tool_name not in TOOLS:
        # Suggest, never auto-substitute — the model must confirm a fuzzy
        # match itself rather than the dispatcher silently rerouting it.
        suggestion = get_close_matches(tool_name, TOOLS.keys(), n=1, cutoff=0.6)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        return None, f"Unknown tool '{tool_name}'. Available tools: {list(TOOLS)}.{hint}"

    spec = TOOLS[tool_name]
    if isinstance(raw_args, str):  # JSON failed to parse in try_parse_action
        return None, f"Arguments for '{tool_name}' were not valid JSON: {raw_args}"

    try:
        validated = spec["schema"].model_validate(raw_args)
    except ValidationError as e:
        return None, f"Invalid arguments for '{tool_name}': {e}"

    result = spec["execute"](**validated.model_dump())
    return result, None


def run_turn(client: OpenAI, model: str, messages: list[dict]) -> str:
    """Handle one user turn: at most one tool hop, then a final answer."""
    reply = call_model(client, model, messages)
    tool_name, raw_args = try_parse_action(reply)

    if tool_name is None:
        return reply  # model chose to answer directly — no tool needed

    result, error = resolve_tool_call(tool_name, raw_args)
    if error:
        return f"Tool call failed: {error}"  # Step 7 replaces this with a retry

    messages.append({"role": "assistant", "content": reply})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Observation from {tool_name}: {json.dumps(result)}\n\n"
                "Now answer the original question using this observation. Do not call another tool."
            ),
        }
    )
    return call_model(client, model, messages)


def main() -> None:
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key="not-needed")
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What's the latest stable version of Python?"},
    ]
    answer = run_turn(client, DEFAULT_MODEL, messages)
    print(f"Assistant: {answer}")


if __name__ == "__main__":
    main()
```

</details>

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| Model's final answer completely ignores the search result | The "Do not call another tool" instruction was dropped from the observation message, and the model tries to search again instead of answering | Keep that sentence — it closes off the one behavior this single-hop design can't support |
| `NameError: json is not defined` | Forgot the `import json` added back in Step 4 | Confirm the imports block matches the checkpoint above |
| Appended messages in the wrong order (observation before the assistant reply) | Easy copy-paste slip | The assistant's `ACTION:` reply must be appended first — the API expects a request that produced a tool call to appear before that call's result |

Next: **[Bounded Retries](08-bounded-retries.md)** — replace the placeholder error return with a real self-correction loop, capped at a fixed number of attempts.

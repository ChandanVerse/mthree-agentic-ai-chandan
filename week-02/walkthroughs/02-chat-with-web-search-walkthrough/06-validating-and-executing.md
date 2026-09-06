# Step 5 — Validating and Executing

> [Back to index](README.md) · Previous: [Parsing the Decision](05-parsing-the-decision.md) · Next: [The Round Trip](07-the-round-trip.md)

## Goal

Turn a parsed `(tool_name, raw_args)` pair into either a real tool result or a specific, actionable error message — never a crash, and never a silent guess.

## Why this matters

Two things can go wrong between "the model named a tool" and "the tool ran": the name might not exist (hallucinated or misspelled), or the arguments might not match the schema (extra field, wrong type, missing required field). Both are worth handling explicitly rather than letting a `KeyError` or `ValidationError` propagate up and crash the app.

The hallucinated-name case has a specific design decision worth calling out: when the model's tool name is *close* to a real one (a typo, a plural, a near-miss), the dispatcher **suggests** the likely match but does not silently substitute it. Auto-correcting a wrong tool name on the model's behalf would mean the model never learns its call was wrong — it would just get an answer from a tool it didn't ask for, with no error to notice or correct from. Surfacing "did you mean X?" as an error instead keeps the model in the loop and gives it something to explicitly confirm on its next attempt.

## 1. New imports

Confirm these are already present from the previous step (added there, used here):

```python
from difflib import get_close_matches
...
from pydantic import ValidationError
from tools import TOOLS, tools_prompt_block
```

## 2. `resolve_tool_call()`

Add this function after `try_parse_action()`:

```python
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
```

Four checks, in order, each guarding against one specific failure:

1. **Unknown tool name** — looked up against `TOOLS`, with a fuzzy-match suggestion via `difflib.get_close_matches` when nothing exact hits.
2. **Broken JSON** — recognized cheaply, because Step 4's `try_parse_action` already marked this case by returning a `str` instead of a `dict` for `raw_args`. No re-parsing needed here — just a type check.
3. **Schema mismatch** — `spec["schema"].model_validate(raw_args)` runs the `WebSearchArgs` validation from Step 2; a `ValidationError`'s message is passed straight through, since Pydantic's own message already names the exact field and problem.
4. **Execution** — only reached once the previous three checks pass. `validated.model_dump()` converts the validated Pydantic model back into plain keyword arguments for `spec["execute"]`.

The function's return convention — `(result, error)`, exactly one populated — means the caller never has to catch an exception to know whether the call succeeded; it just checks which slot is `None`.

## Try it

Test all three error paths plus the success path directly, without a model involved:

```bash
uv run python -c "
from chat_with_tools import resolve_tool_call
print(resolve_tool_call('web_serach', {'query': 'test'}))
print(resolve_tool_call('web_search', '{\"query\":'))
print(resolve_tool_call('web_search', {'not_a_field': 'test'}))
result, error = resolve_tool_call('web_search', {'query': 'current weather in Tokyo'})
print(error, '->', result['has_more'] if result else None)
"
```

Expected shape (search result content will vary):

```text
(None, "Unknown tool 'web_serach'. Available tools: ['web_search']. Did you mean 'web_search'?")
(None, 'Arguments for \'web_search\' were not valid JSON: {"query":')
(None, "Invalid arguments for 'web_search': 1 validation error for WebSearchArgs\nquery\n  Field required ...")
None -> True
```

## Checkpoint

<details>
<summary>Full <code>chat_with_tools.py</code> (through <code>resolve_tool_call</code>)</summary>

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
```

</details>

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| Fuzzy-match hint never appears, even for an obvious typo | `cutoff=0.6` is stricter than the typo's similarity score | Lower the cutoff, but not so far it starts suggesting unrelated tools |
| `resolve_tool_call` raises instead of returning an error | `raw_args` was neither a `dict` nor a `str` (shouldn't happen if called with `try_parse_action`'s output, but easy to break when testing by hand) | Only call it with values shaped like `try_parse_action` produces |
| Validation passes but `execute()` still throws | The schema doesn't cover everything the function needs to succeed (e.g. a network error) | This is a genuinely different failure mode (execution, not validation) — Step 6's retry loop treats it the same as any other tool-call failure, since `web_search` is read-only and safe to retry |

Next: **[The Round Trip](07-the-round-trip.md)** — assemble `try_parse_action` and `resolve_tool_call` into the actual per-turn logic: call the model, execute a tool if it asked for one, and get a final answer.

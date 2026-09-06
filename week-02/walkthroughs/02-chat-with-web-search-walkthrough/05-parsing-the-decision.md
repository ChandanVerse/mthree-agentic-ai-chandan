# Step 4 — Parsing the Decision

> [Back to index](README.md) · Previous: [The Raw Decision](04-the-raw-decision.md) · Next: [Validating and Executing](06-validating-and-executing.md)

## Goal

Turn a raw reply string into one of three outcomes: no tool call, a valid tool call, or an attempted-but-broken tool call — and keep those three cases distinguishable from each other.

## Why this matters

There's a failure mode that's easy to miss if you only test the happy path: a model that *tries* to call a tool but produces truncated or malformed JSON (cut off mid-generation, a missing closing brace, a stray comma). If your parser only recognizes "valid call" and treats everything else as "no call," that broken attempt silently becomes a plain-text answer — the model's clearly-signaled intent to search gets thrown away instead of being reported back as an error it can fix. Keeping "didn't try" and "tried but failed" as two distinct outcomes (instead of collapsing them into one) is what makes the retry loop in a later step possible at all — you can't retry a failure your code never noticed.

## 1. The two regex patterns

Near the top of the file, after the existing imports, add:

```python
import json
import re
from difflib import get_close_matches

from openai import APIError, OpenAI
from pydantic import ValidationError

from tools import TOOLS, tools_prompt_block
```

Replace the earlier `from tools import tools_prompt_block` import with the line above — it now also pulls in `TOOLS` for the next step, plus everything this step and Step 5 need.

Then, below `DEFAULT_BASE_URL`, add:

```python
# The text protocol a model must follow to call a tool: a line reading
# "ACTION: <tool_name> {json args}". Matching the tool name and the JSON
# blob as two separate steps (below) lets us tell "no tool call at all"
# apart from "tried to call a tool but the JSON is truncated/broken".
ACTION_NAME_PATTERN = re.compile(r"ACTION:\s*(\w+)", re.DOTALL)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
```

Two separate patterns, matched in two separate steps, is the whole design here: first look for *any* `ACTION: <name>`, independent of whether what follows is valid JSON. Only once that's found do you go looking for a JSON object after it. That ordering is what lets the function tell "no attempt" apart from "attempted, but the JSON half broke."

## 2. `try_parse_action()`

```python
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
```

Walk through the three return paths with trainees in order:

- No `ACTION:` anywhere in the text → `(None, None)`. The caller's job here is simple: just pass the reply straight through as the answer.
- `ACTION: <name>` found, and everything after it parses as JSON → `(tool_name, args_dict)`, a clean call ready to validate.
- `ACTION: <name>` found, but nothing after it looks like a JSON object at all, *or* it looks like one but fails `json.loads` — both land in the third case, `(tool_name, raw_str)`, carrying the broken text itself rather than a dict. The caller can tell this apart from a clean call because the second element is a `str` instead of a `dict`.

## Try it

Before wiring this into the model call, test it directly against three hand-written strings that cover all three outcomes:

```bash
uv run python -c "
from chat_with_tools import try_parse_action
print(try_parse_action('12 * 7 is 84.'))
print(try_parse_action('ACTION: web_search {\"query\": \"latest stable Python version\"}'))
print(try_parse_action('ACTION: web_search {\"query\": '))
"
```

Expected output — one tuple per line, matching the three cases above:

```text
(None, None)
('web_search', {'query': 'latest stable Python version'})
('web_search', '{"query":')
```

## Checkpoint

<details>
<summary>Full <code>chat_with_tools.py</code> (through <code>try_parse_action</code>)</summary>

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
```

</details>

The temporary `main()` from Step 3 is gone from this checkpoint — it was scratch code, and the next few steps build the real dispatch logic in its place.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `try_parse_action` returns `(None, None)` even though the reply contains `ACTION:` | The model wrote something like `Action:` (different case) — the pattern is case-sensitive | Either loosen the regex with `re.IGNORECASE` or tighten the system prompt's example casing |
| Third case (`raw_str`) triggers even on a well-formed call | Extra text *inside* the JSON braces confuses `JSON_OBJECT_PATTERN`'s greedy match | Expected with a very chatty model; this is exactly the case the next step's error-feedback loop is for |
| `ImportError` for `TOOLS` | Forgot to update the `from tools import ...` line | Import both `TOOLS` and `tools_prompt_block` from `tools` |

Next: **[Validating and Executing](06-validating-and-executing.md)** — turn a parsed `(tool_name, args)` pair into either a real result or a specific, actionable error.

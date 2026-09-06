# Step 5 — Parsing the Reply

> [Back to index](README.md) · Previous: [The Profile and Reply Format](05-the-profile-and-reply-format.md) · Next: [The Dispatcher — Validating Calls](07-the-dispatcher-validating-calls.md)

## Goal

Turn a raw reply string into a `ParsedReply`: a structured value carrying the model's `Thought`, and either an `Action`/`Action Input` pair, a `Final Answer`, or neither.

## Why this matters

Step 2's `try_parse_action()` only had to distinguish three outcomes (no call, valid call, broken call) from one line of text. This app's replies are richer — every turn carries a `Thought`, and then *either* an `Action`/`Action Input` pair *or* a `Final Answer` — so `parse_reply()` needs to extract more structure without conflating "the model is still working" with "the model is done."

The design choice worth calling out before writing any regex: this uses **four separate, narrowly-scoped patterns** instead of one large pattern trying to capture the whole block at once. A single mega-regex covering `Thought`, `Action`, `Action Input`, and `Final Answer` all at once would need to encode "these are mutually exclusive, and I don't know which one is present" inside the regex itself — fragile, and unreadable if a small model's output has any noise or reordering. Four small patterns, checked in a clear order (final answer first, then action, then give up), is what keeps the logic in Python control flow instead of buried in regex alternation.

## 1. New imports

Update the top of `agent.py`:

```python
import json
import os
import re
from dataclasses import dataclass

from openai import OpenAI

from tools import tools_prompt_block
```

## 2. The four patterns

Add this below `DEFAULT_MAX_STEPS`:

```python
# --------------------------------------------------------------------------
# Parsing the model's Thought / Action / Action Input / Final Answer reply.
# Split into separate regexes (rather than one big pattern) so we can tell
# apart "no action at all", "action with no parseable JSON" (truncated or
# malformed), and a clean call.
# --------------------------------------------------------------------------
THOUGHT_PATTERN = re.compile(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer):|\Z)", re.DOTALL)
FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)
ACTION_NAME_PATTERN = re.compile(r"Action:\s*(\w+)", re.DOTALL)
ACTION_INPUT_LABEL_PATTERN = re.compile(r"Action Input:\s*(.*)", re.DOTALL)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
```

`THOUGHT_PATTERN` is the one worth reading carefully: `(.*?)` is a *non-greedy* capture, bounded by a lookahead `(?=\n(?:Action|Final Answer):|\Z)` that stops it exactly at the next `Action:`/`Final Answer:` label, or at the end of the string (`\Z`) if there is none. Without that lookahead, a greedy `.*` would swallow the `Action:`/`Final Answer:` lines that follow, since `re.DOTALL` lets `.` match newlines too.

## 3. `ParsedReply`

```python
@dataclass
class ParsedReply:
    thought: str
    action: str | None = None
    action_input: dict | str | None = None  # str means "malformed/truncated"
    final_answer: str | None = None
```

A plain dataclass, not a Pydantic model — this is an internal parsing result, never validated against external input directly (that's what the dispatcher is for, next). `action_input`'s type hint carries the same "str means broken" convention Step 2's `try_parse_action` used for its second return value, just promoted to a named field instead of a tuple position.

## 4. `parse_reply()`

```python
def parse_reply(text: str) -> ParsedReply:
    thought_match = THOUGHT_PATTERN.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    final_match = FINAL_ANSWER_PATTERN.search(text)
    if final_match:
        return ParsedReply(thought=thought, final_answer=final_match.group(1).strip())

    action_match = ACTION_NAME_PATTERN.search(text)
    if not action_match:
        # The model didn't follow the format at all. Rather than looping on
        # an unparseable turn, the caller treats the raw reply as the answer.
        return ParsedReply(thought=thought)

    action_name = action_match.group(1)
    input_match = ACTION_INPUT_LABEL_PATTERN.search(text)
    search_space = input_match.group(1) if input_match else text[action_match.end() :]
    json_match = JSON_OBJECT_PATTERN.search(search_space)

    if not json_match:
        return ParsedReply(thought=thought, action=action_name, action_input=search_space.strip())

    try:
        action_input = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        action_input = json_match.group(0)

    return ParsedReply(thought=thought, action=action_name, action_input=action_input)
```

Walk the branches in the order the function checks them:

1. **`Final Answer` present** → return immediately with just `thought` and `final_answer` set. Checked first, since a reply that includes both a stray `Action:`-shaped fragment and a genuine `Final Answer` should still be treated as done.
2. **No `Action:` label found either** → the model didn't follow the format at all. `ParsedReply(thought=thought)` leaves `action`, `action_input`, and `final_answer` all at their defaults (`None`) — the caller (built in Step 8) reads this as "give up parsing, return the raw reply."
3. **`Action:` found** → look for `Action Input:` specifically; fall back to searching everything after the action name if that label is missing (a model that drops the label but still writes JSON nearby).
4. **No JSON object found in that search space** → `action_input` becomes the raw stripped text, a `str` — the "attempted but broken" case, same convention as Step 2's parser.
5. **JSON found but `json.loads` fails** → same "broken" outcome, `action_input` stays the raw matched text.
6. **Clean parse** → `action_input` is a real `dict`, ready for the dispatcher.

## Try it

Test all four outcomes directly, without any model call:

```bash
uv run python -c "
from agent import parse_reply
print(parse_reply('Thought: I need to compute 17 * 12.99 before I can answer.\nAction: calculator\nAction Input: {\"expression\": \"17 * 12.99\"}'))
print(parse_reply('Thought: I now know the answer.\nFinal Answer: Yes, it is under 220.'))
print(parse_reply('Thought: Let me try the calc tool.\nAction: calc\nAction Input: {\"expression\": \"1+1\"'))
print(parse_reply('The answer is 84.'))
"
```

Expected output:

```text
ParsedReply(thought='I need to compute 17 * 12.99 before I can answer.', action='calculator', action_input={'expression': '17 * 12.99'}, final_answer=None)
ParsedReply(thought='I now know the answer.', action=None, action_input=None, final_answer='Yes, it is under 220.')
ParsedReply(thought='Let me try the calc tool.', action='calc', action_input='{"expression": "1+1"', final_answer=None)
ParsedReply(thought='', action=None, action_input=None, final_answer=None)
```

Four lines, four distinct outcomes: a clean action, a final answer, an action with truncated JSON (note `action_input` is a `str`, not a `dict`), and a reply that ignored the format entirely (everything falls back to its default).

## Checkpoint

<details>
<summary>Full <code>agent.py</code> (through <code>parse_reply</code>)</summary>

```python
#!/usr/bin/env python3
"""Step 3 — a small ReAct agent with a real dispatcher and guardrails.

This generalizes the single-tool-hop chat app from Step 2 of this series
into the full ReAct loop: `Thought -> Action -> Observation`, repeated
until the model itself decides it has enough to answer, capped by a
`max_steps` guardrail. The dispatcher below handles four failure modes:
hallucinated tool names, malformed/truncated arguments, execution errors,
and (new in this app) never blind-retrying a side-effecting tool.

Run:
    uv run agent.py "What is 17 * 12.99, and is that under 220?"
    uv run agent.py            # interactive: one goal per line
"""
import json
import os
import re
from dataclasses import dataclass

from openai import OpenAI

from tools import tools_prompt_block

DEFAULT_MODEL = os.environ.get("DMR_MODEL", "docker.io/ai/gemma4:E4B")
DEFAULT_BASE_URL = os.environ.get("DMR_BASE_URL", "http://localhost:12434/v1")
DEFAULT_MAX_STEPS = 6  # a step cap is a guardrail, not decoration

# --------------------------------------------------------------------------
# Profile — the fixed identity/constraints shaping every decision. Includes
# a worked example so a small local model has something concrete to
# imitate, since it can't be trusted to infer a strict output format from a
# description alone.
# --------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are a careful problem-solving agent. Work step by step and use tools instead of guessing.

{tools_prompt_block()}

Respond using EXACTLY this format, one block per turn:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: {{"key": "value"}}

After you receive an Observation, continue with another Thought, and either
another Action, or finish with:

Thought: <your reasoning for why you're done>
Final Answer: <your answer to the user's request>

Example:

Thought: I need to compute 17 * 12.99 before I can answer.
Action: calculator
Action Input: {{"expression": "17 * 12.99"}}

Rules:
- Never compute arithmetic yourself — always call calculator.
- Only call save_note when the user explicitly asks you to save or remember something.
- If a tool call fails, read the error and correct your Action Input on the next turn.
- Give a Final Answer as soon as you genuinely have enough information — don't call tools you don't need.
"""

# --------------------------------------------------------------------------
# Parsing the model's Thought / Action / Action Input / Final Answer reply.
# Split into separate regexes (rather than one big pattern) so we can tell
# apart "no action at all", "action with no parseable JSON" (truncated or
# malformed), and a clean call.
# --------------------------------------------------------------------------
THOUGHT_PATTERN = re.compile(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer):|\Z)", re.DOTALL)
FINAL_ANSWER_PATTERN = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)
ACTION_NAME_PATTERN = re.compile(r"Action:\s*(\w+)", re.DOTALL)
ACTION_INPUT_LABEL_PATTERN = re.compile(r"Action Input:\s*(.*)", re.DOTALL)
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ParsedReply:
    thought: str
    action: str | None = None
    action_input: dict | str | None = None  # str means "malformed/truncated"
    final_answer: str | None = None


def parse_reply(text: str) -> ParsedReply:
    thought_match = THOUGHT_PATTERN.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    final_match = FINAL_ANSWER_PATTERN.search(text)
    if final_match:
        return ParsedReply(thought=thought, final_answer=final_match.group(1).strip())

    action_match = ACTION_NAME_PATTERN.search(text)
    if not action_match:
        # The model didn't follow the format at all. Rather than looping on
        # an unparseable turn, the caller treats the raw reply as the answer.
        return ParsedReply(thought=thought)

    action_name = action_match.group(1)
    input_match = ACTION_INPUT_LABEL_PATTERN.search(text)
    search_space = input_match.group(1) if input_match else text[action_match.end() :]
    json_match = JSON_OBJECT_PATTERN.search(search_space)

    if not json_match:
        return ParsedReply(thought=thought, action=action_name, action_input=search_space.strip())

    try:
        action_input = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        action_input = json_match.group(0)

    return ParsedReply(thought=thought, action=action_name, action_input=action_input)


def call_model(client: OpenAI, model: str, messages: list[dict]) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def main() -> None:
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key="not-needed")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is 17 * 12.99, and is that under 220?"},
    ]
    print(call_model(client, DEFAULT_MODEL, messages))


if __name__ == "__main__":
    main()
```

</details>

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `thought` includes the `Action:` line as part of its text | `THOUGHT_PATTERN`'s lookahead was written with a greedy `.*` instead of non-greedy `.*?` | Use `(.*?)` — greedy would still stop at the lookahead, but non-greedy makes the intent explicit and avoids edge cases with repeated labels |
| `parse_reply` returns `final_answer=None` even though the text contains `Final Answer:` | Text says `final answer:` or similar in different case | The pattern is case-sensitive by design, matching the exact casing in `SYSTEM_PROMPT`'s instructions |
| `action_input` is a `dict` when you expected the "broken" `str` case, or vice versa | Test string's JSON braces don't match what `JSON_OBJECT_PATTERN` (`\{.*\}`, greedy) actually captures | Print `json_match.group(0)` directly while debugging to see exactly what substring was matched |

Next: **[The Dispatcher — Validating Calls](07-the-dispatcher-validating-calls.md)** — turn a parsed action into either a real result or a specific, actionable error.

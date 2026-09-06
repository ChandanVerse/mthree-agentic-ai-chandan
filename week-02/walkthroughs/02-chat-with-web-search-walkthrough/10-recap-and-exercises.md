# Step 9 — Recap and Exercises

> [Back to index](README.md) · Previous: [Assembling the CLI and Loop](09-assembling-the-cli-and-loop.md)

## What you built

Two files, from an empty folder:

- **`tools.py`** — a `web_search` tool defined as a contract: a Pydantic argument schema, a live implementation with a bounded, self-describing return shape, and a registry that renders its own description into a prompt.
- **`chat_with_tools.py`** — a chat loop where the model decides, every turn, whether it needs to search — expressed as a plain-text `ACTION:` line rather than a native API field — and the app parses, validates, executes, and retries that decision up to a fixed bound, resolving in at most one tool hop before answering.

Both should now match [../02-chat-with-web-search/](../02-chat-with-web-search/) exactly.

## Quick reference card

| Concept | Where it lives |
| --- | --- |
| Tool contract (schema + description + execute) | `TOOLS` dict in [tools.py](../02-chat-with-web-search/tools.py) |
| Bounded, self-describing return shape | `web_search()`'s `has_more`/`total` fields |
| Prompt-based tool calling (no native function-calling) | `tools_prompt_block()` + `SYSTEM_PROMPT` |
| Distinguishing "no call" from "broken call" | `try_parse_action()`'s three return shapes |
| Malformed-args mitigation | `resolve_tool_call()`'s `ValidationError` branch |
| Hallucinated-tool-name mitigation (suggest, never auto-substitute) | `resolve_tool_call()`'s `get_close_matches` branch |
| The actual round trip (observation fed back as a message) | `run_turn()`'s success path |
| Bounded self-correction retries | `MAX_TOOL_RETRIES` in `run_turn()` |
| The "at most one tool hop" ceiling | `run_turn()` only ever calls `call_model()` a second time, never a third |

## Gotchas reference

| Symptom | Cause | Fix |
| --- | --- | --- |
| Model always searches, even for trivial questions | Small models over-trigger on tool-shaped prompts | Tighten the system prompt's "only call when you genuinely need it" instruction, or lower temperature |
| Model prints the `ACTION:` line to the user instead of "just" calling it | The model didn't follow the protocol exactly | Expected occasionally with small local models — the retry loop or a stricter prompt usually resolves it |
| `[error] Could not reach model` | Docker Model Runner not running / model not pulled | See [00-local-model-setup](../00-local-model-setup/README.md) |
| Conversation slows down or grows unbounded over a long session | No `trim_history()`-style windowing here (unlike Step 1's `chat.py`) | Left out deliberately for this step — Exercise 4 below adds it back |

## Discussion questions

1. What's the one-sentence test from Step 0 for "is this actually agentic," and where exactly in the code does the model's decision happen?
2. `try_parse_action()` returns three different shapes from one function instead of raising exceptions for the failure cases. What would change in `run_turn()` if it raised instead?
3. Why does `resolve_tool_call()` *suggest* a fuzzy-matched tool name instead of just using it? Construct a scenario where auto-substituting would have produced a wrong answer silently.
4. This app resolves in at most one tool hop. Describe a question where that's a real limitation — one where a model would need to see a first search result before knowing what to search for next.
5. `web_search` is read-only, so blind retries are safe. What would have to change in `resolve_tool_call()` or `run_turn()` if the tool instead sent an email or wrote to a database?

## Exercises

Roughly ordered easiest to hardest:

1. **Add a second tool.** A `calculator` tool (evaluate a simple arithmetic expression) is a good first addition — it exercises the same schema/registry/prompt-rendering path as `web_search` without needing the network.
2. **Show which tool fired.** Print `[used web_search]` (or similar) alongside the assistant's answer whenever `run_turn()` took the tool path, so trainees can see the decision without reading logs.
3. **Loosen or tighten the trigger.** Adjust the "only call the tool when you genuinely need it" sentence in `SYSTEM_PROMPT` and observe how it changes the model's behavior on borderline questions ("What year did the Berlin Wall fall?" vs. "What's the weather right now in Tokyo?").
4. **Bring back bounded history.** Port `trim_history()` from [01-basic-chat-app/chat.py](../01-basic-chat-app/chat.py) into this app, keeping the system prompt plus the last N turns — being careful that a trim never splits an `ACTION:`/observation exchange in half.
5. **Make `MAX_TOOL_RETRIES` visible.** Add a `--max-tool-retries` flag, and log each retry attempt to stderr so a trainee can watch the self-correction loop happen turn by turn.
6. **Rebuild `chat_with_tools.py` from a blank file, unaided.** The best test of whether the concepts stuck: close this walkthrough and rewrite it from memory, checking against [../02-chat-with-web-search/chat_with_tools.py](../02-chat-with-web-search/chat_with_tools.py) only at the end.

## What's next

**[../03-small-agent/README.md](../03-small-agent/README.md)** — generalize this single tool hop into a full ReAct loop (`Thought → Action → Observation`, repeated) with multiple tools, a `max_steps` guardrail, and an idempotent-vs-side-effecting tool distinction that this app's single read-only tool didn't force you to confront.

**[../02-chat-with-web-search-langchain-walkthrough/README.md](../02-chat-with-web-search-langchain-walkthrough/README.md)** — a side detour, not a next step: rebuild this same app on LangChain's native tool-calling, and see exactly how much of `try_parse_action`/`resolve_tool_call` a framework actually removes versus what survives unchanged (hint: the retry loop).

# Step 7 — Recap and Exercises

> [Back to index](README.md) · Previous: [CLI and Interactive Mode](07-cli-and-interactive-mode.md)

## What you built

A rebuild of [03-small-agent](../03-small-agent/)'s `agent.py` on `create_agent`: the same three tools, the same ReAct behavior, the same step-limit guardrail concept and side-effecting no-blind-retry rule — but with `create_agent(llm, tools, system_prompt=...)` doing the work that `call_model()`/`parse_reply()`/`dispatch_tool_call()`/`run_agent()`'s manual loop did by hand. It should now match [../03-small-agent-langgraph/agent.py](../03-small-agent-langgraph/agent.py) exactly.

## Quick reference card

| Concept | Where it lives in this app |
| --- | --- |
| The whole ReAct loop | `create_agent(llm, tools, system_prompt=...)` in `build_agent()` |
| Native tool-calling | Each `@tool`-decorated function; no text format taught in `SYSTEM_PROMPT` |
| Step-limit guardrail (approximate) | `recursion_limit` in `run_agent()`'s `config`, set to `max_steps * 2 + 1` |
| Side-effecting no-blind-retry rule (weakened) | The `failed_once` closure inside `make_save_note_tool()` |
| Sandboxed arithmetic (no `eval`) | `_eval_node()` — copied verbatim from [../03-small-agent/tools.py](../03-small-agent/tools.py) |
| Fresh state per goal | `build_agent()` called inside `run_one()`, once per goal |
| Step trace | `print_trace()` — prints whatever the model actually produced, not a forced `Thought:` format |

## Tradeoffs: what the framework buys you, and what it costs

| | Hand-rolled ([03-small-agent](../03-small-agent/)) | This app |
| --- | --- | --- |
| Lines spent on the loop itself | `run_agent()`'s `for step_num in range(1, max_steps + 1)`, manually calling, parsing, dispatching, appending | Zero — the loop lives inside `create_agent`'s compiled graph |
| Lines spent on parsing the model's decision | `parse_reply()` — five regexes | Zero — `tool_calls` arrives already parsed |
| Lines spent on validating a call's arguments | Pydantic schema classes + `dispatch_tool_call()`'s `model_validate` | Zero separate schema — inferred from each function's type hints |
| Hallucinated tool name handling | `difflib.get_close_matches` suggests the nearest real tool | Handled before the call reaches your code — the model chooses from an explicit tool list via the API's schema field |
| Step-limit guardrail precision | Exact — counts Thought/Action cycles directly | Approximate — `recursion_limit` counts graph super-steps, mapped via `max_steps * 2 + 1` |
| Side-effecting tool retry safety | A dispatcher-level rule (`retryable_by_model=False`) that **stops the whole agent run** | A tool-level closure that refuses a second write but **cannot stop anything else the model does next** |
| Visibility into *why* a call failed to parse | Full — you control the regex and see every intermediate string | None needed for parsing (the API owns it) — but also none available if tool-calling itself misbehaves |
| What still has to be hand-rolled either way | The tools' own implementation, the system prompt's behavioral rules | Identical — see `calculator`, `web_search`, `make_save_note_tool()` |
| Dependency footprint | `openai` + `pydantic` | `langchain`, `langchain-openai`, `langgraph`, and their transitive dependencies |
| Reliability on small/local models | Deliberately robust to models that can't do structured tool-calling at all | Depends entirely on whether your model tag honors `tools`/`tool_calls` |

The net trade here is much larger than the single-tool-hop comparison in [02-chat-with-web-search-langchain](../02-chat-with-web-search-langchain-walkthrough/README.md): an entire loop, parser, and most of a dispatcher disappear into one function call. The one place the trade isn't free is retry-safety for a side-effecting tool — the hand-rolled dispatcher's global "stop everything" behavior has no equivalent hook in the prebuilt graph, and the closure workaround is a real, named step down in what it can guarantee.

## Gotchas reference

| Symptom | Cause | Fix |
| --- | --- | --- |
| `LangGraphDeprecatedSinceV10` warning | Copied a snippet from older LangGraph docs using `create_react_agent` | Not applicable to this app's own code — only relevant if you copy from older material; use `langchain.agents.create_agent` |
| Model never calls a tool, even when it should | Some Gemma tags/quantizations don't reliably emit tool calls through Docker Model Runner's OpenAI-compat layer | Fall back to [03-small-agent](../03-small-agent/)'s text-protocol approach, or try a different model tag |
| Agent stops with "exceeded max_steps" sooner than expected | `recursion_limit` counts graph steps, not tool calls | Raise `--max-steps`, remembering the real limit passed to LangGraph is roughly double |
| `save_note` writes into `agent_notes/` unexpectedly | That's the intended, sandboxed side effect of the tool | `agent_notes/` is gitignored; delete it freely |
| `[error] ... Is Docker Model Runner running?` | Docker Model Runner not running / model not pulled | See [00-local-model-setup](../00-local-model-setup/README.md) |

## Discussion questions

1. Walk through exactly which hand-rolled functions from [03-small-agent](../03-small-agent/) disappeared entirely, which shrank, and which (arguably) got harder to reason about because the mechanism moved inside a library.
2. `make_save_note_tool()` is a factory function, not a plain `@tool`-decorated function like the other two tools. What specifically about `save_note`'s requirements forces that difference?
3. Name the exact behavior the hand-rolled dispatcher's `retryable_by_model=False` guaranteed that the closure in this app cannot guarantee. What would it take to get that exact behavior back inside `create_agent`'s graph?
4. `recursion_limit` is set to `max_steps * 2 + 1`, not `max_steps`. What's the `+ 1` for, and what would you observe differently if it were left out?
5. If your model tag turns out not to support tool-calling reliably, how would this app fail, and would it fail differently from the hand-rolled version in the same situation?

## Exercises

Ordered easiest to hardest:

1. **Print the raw `AIMessage` before `print_trace` formats it.** Add a line inside `run_agent()` right after `result = graph.invoke(...)` that prints `result["messages"]` raw, before `print_trace()` runs — a good way to see the exact message objects `create_agent` produced, the same way Step 4's scratch test did in isolation.
2. **Add a fourth tool.** Pick something simple (a `get_time` tool with no arguments, say). Confirm it only takes a `@tool`-decorated function added to the `tools` list in `build_agent()` — no changes anywhere else — and that this is a much smaller diff than adding a tool to the hand-rolled `TOOLS` registry would be.
3. **Force the save_note refusal path inside a real run**, not just the isolated Step 3 demo: occupy the `agent_notes` path with a plain file (as in [Step 3's Try it](04-the-save-note-closure.md#try-it)), then give the agent a goal that asks it to save two different notes in one turn, and watch the trace show the second call being refused while the first tool call's error is still fed back to the model.
4. **Give `run_one()` a hard-stop-on-side-effecting-failure behavior**, matching the original dispatcher exactly: after `run_agent()` returns, check whether the trace's last `ToolMessage` before the final answer contains the closure's refusal message, and if so, discard the model's subsequent answer and print the original "already failed once" message instead. This is intentionally awkward — the exercise is to feel *why* it's awkward, compared to how naturally `retryable_by_model=False` expressed the same rule in the hand-rolled dispatcher.
5. **Run both agents on 5-10 identical goals** covering all three tools, and produce a short table: which agent's trace looked more informative, which one's guardrail fired at a more predictable point, and whether either one ever mishandled the side-effecting tool. This is the actual side-by-side comparison this walkthrough's README asks you to run.
6. **Rebuild `agent.py` from memory.** Close this walkthrough, write the whole file from scratch — tools, closure, `build_agent`, `run_agent`, `main()` — then diff it against [../03-small-agent-langgraph/agent.py](../03-small-agent-langgraph/agent.py) only at the end.

## What's next

You've now built both variants of the same agent — compare them directly on the same goals if you haven't already, paying particular attention to the one place this walkthrough's rebuild is strictly weaker (the side-effecting no-retry rule) rather than treating the framework version as a pure win. If you want to close that gap for real, look at LangGraph's lower-level graph-building API (custom nodes and conditional edges) rather than `create_agent` — that's the tool that would let you express "stop the whole run" as a first-class rule again, at the cost of the one-liner simplicity this walkthrough demonstrated.

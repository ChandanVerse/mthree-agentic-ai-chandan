# Step 11 — Recap and Exercises

> [Back to index](README.md) · Previous: [CLI and Interactive Mode](11-cli-and-interactive-mode.md)

## What you built

Two files, from an empty folder:

- **`tools.py`** — three tools as contracts: a sandboxed `calculator` (a restricted AST walk, never `eval()`), a `web_search` with a bounded, self-describing return shape, and a file-writing `save_note` — plus `IDEMPOTENT_TOOLS`/`SIDE_EFFECTING_TOOLS`, the classification every later retry decision is built on.
- **`agent.py`** — a `Thought → Action → Observation` loop where the model decides, every turn, whether it needs a tool or has enough to give a `Final Answer`, wrapped in a dispatcher that validates every call, a retry policy that never blind-retries a side effect, and a `max_steps` guardrail that bounds the whole thing.

Both should now match [../03-small-agent/](../03-small-agent/) exactly.

## Quick reference card

| Concept | Where it lives |
| --- | --- |
| ReAct loop | `run_agent()`'s `for step_num in range(1, max_steps + 1)` in [agent.py](../03-small-agent/agent.py) |
| Step-limit guardrail | `max_steps` parameter, default `DEFAULT_MAX_STEPS = 6` |
| Worked example in the prompt | The `Example:` block inside `SYSTEM_PROMPT` |
| Distinguishing final answer / clean call / broken call / no call | `parse_reply()`'s four return shapes on `ParsedReply` |
| Tool contract | `TOOLS` dict in [tools.py](../03-small-agent/tools.py) |
| Idempotent vs. side-effecting classification | `IDEMPOTENT_TOOLS` / `SIDE_EFFECTING_TOOLS` in [tools.py](../03-small-agent/tools.py) |
| Sandboxed arithmetic (no `eval`) | `_eval_node()` in [tools.py](../03-small-agent/tools.py) |
| Hallucinated-tool-name mitigation (suggest, never auto-substitute) | `dispatch_tool_call()`'s `get_close_matches` branch |
| The `retryable_by_model` load-bearing flag | `ToolError` and every branch of `dispatch_tool_call()` that constructs one |
| Never blind-retrying a side effect | `dispatch_tool_call()`'s `SIDE_EFFECTING_TOOLS` + `tool_failure_counts` check |
| No memory across goals | A fresh `messages` list built inside every `run_agent()` call |

## Gotchas reference

| Symptom | Cause | Fix |
| --- | --- | --- |
| Agent stops with "exceeded max_steps" | Model kept calling tools without converging, or genuinely needed more steps | Raise `--max-steps`, or check the trace for a repeating mistake |
| Agent's reply doesn't follow the Thought/Action format at all | Small local models sometimes ignore formatting instructions | The controller falls back to returning the raw reply as the answer rather than looping forever — see `parse_reply`'s `action is None` branch |
| `save_note` writes into `agent_notes/` unexpectedly | That's the intended, sandboxed side effect of the tool | `agent_notes/` is gitignored; delete it freely |
| `[error] Could not reach model` | Docker Model Runner not running / model not pulled | See [00-local-model-setup](../00-local-model-setup/README.md) |

## Discussion questions

1. Trace exactly which lines changed between Step 8's `while True` loop and Step 9's capped version. Is the step cap a change to *what* the loop can do, or only to *how long* it's allowed to keep doing it?
2. `dispatch_tool_call()` returns `retryable_by_model=False` in exactly one place. Find it, and explain why that's the only branch where the dispatcher refuses a call outright rather than reporting an error and letting the model try again.
3. Why is `IDEMPOTENT_TOOLS`/`SIDE_EFFECTING_TOOLS` a property of the *tool* (a fixed set membership) rather than something computed per failure? Construct a tool where you genuinely couldn't decide idempotence in advance.
4. This app has no memory across goals — each CLI invocation or interactive line starts fresh. Describe a realistic use case where that's the right design, and one where it would actively get in the way.
5. `parse_reply()` treats "no `Action:` found" the same way whether or not a `Thought:` was found. What would change in `run_agent()` if a reply with a `Thought` but no `Action` and no `Final Answer` were instead treated as a retryable parsing error, fed back to the model, rather than returned immediately as the answer?

## Exercises

Roughly ordered easiest to hardest:

1. **Add a fourth tool.** A `current_time` tool (no arguments, returns the current date/time) is a good first addition — it's read-only, so it belongs in `IDEMPOTENT_TOOLS`, and it exercises the same schema/registry/prompt-rendering path as the existing three without needing the network.
2. **Log the trace to a file, not just stdout.** Have `run_agent()` optionally append every `Thought`/`Action`/`Observation` line to a per-run log file, so a long interactive session can be reviewed after the fact.
3. **Make the retry-count guardrail visible.** Print a warning the moment a tool's `tool_failure_counts` entry reaches, say, 2 — even though nothing currently stops the model from retrying an idempotent tool indefinitely within the `max_steps` budget.
4. **Give `save_note` a second chance, safely.** Right now, any `save_note` failure ends the whole run. Design (on paper first) a way to let the model retry `save_note` safely — for example, requiring it to pass back a value proving it saw the confirmation of the first attempt failing before a second attempt is allowed. What has to be true about the tool for this to be genuinely safe?
5. **Add a self-check step, straight from this app's own "What's Next."** After a `Final Answer`, add an evaluator call that re-derives the result independently — a basic Reflexion loop — and re-run the tax-on-subtotal example from Step 10 enough times to see whether it catches a wrong first attempt.
6. **Rebuild `agent.py` from a blank file, unaided.** The best test of whether the concepts stuck: close this walkthrough and rewrite it from memory, checking against [../03-small-agent/agent.py](../03-small-agent/agent.py) only at the end.

## What's next

This app deliberately stops at plain ReAct. Two natural extensions, if you want to keep building beyond the exercises above:

1. **Plan-and-Execute** — add a planning call before the loop that writes out the full step sequence up front, useful for auditing a plan before any tool runs.
2. **Reflexion** — after a `Final Answer`, add an evaluator step that re-derives the result independently and, on mismatch, retries the whole goal with the mistake noted in context (Exercise 5 above is a first pass at this).

**[../03-small-agent-langgraph/README.md](../03-small-agent-langgraph/README.md)** — a side detour, not a next step: the same three tools and the same agent, rebuilt on LangGraph's prebuilt `create_agent`, replacing `parse_reply`, `dispatch_tool_call`, `ToolError`, and `run_agent`'s manual loop with a single function call. Worth reading after this walkthrough specifically to see what that one-liner quietly gives up: an exact `max_steps` guardrail (LangGraph's `recursion_limit` counts graph super-steps, not `Thought`/`Action` cycles directly) and this app's hard stop-the-whole-run behavior on a failed side-effecting tool (the framework version can only refuse the second write, not halt everything else the model might do next).

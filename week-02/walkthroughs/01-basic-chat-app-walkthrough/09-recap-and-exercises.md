# Step 8 — Recap and Exercises

> [Back to index](README.md) · Previous: [CLI Flags and Error Handling](08-cli-flags-and-error-handling.md)

## What you built

Two scripts, from an empty folder:

- **`single_prompt.py`** — one prompt, one reply, no memory. The atomic unit every other app in this series builds on.
- **`chat.py`** — an interactive loop with a system prompt, full conversation memory, token-by-token streaming, a bounded history window, and configurable model/endpoint flags with graceful connection-error handling.

Both should now match [../01-basic-chat-app/](../01-basic-chat-app/) exactly.

## Quick reference card

| Concept | Where it lives |
| --- | --- |
| Plain LLM app | The entire `chat.py` script — one call per turn, no tools |
| Message roles | `{"role": "system"\|"user"\|"assistant", "content": ...}` dicts throughout |
| Short-term memory | The `messages` list, resent in full on every call |
| Statelessness | Why `messages` must be resent — the server remembers nothing |
| Streaming | `stream=True` + iterating `chunk.choices[0].delta.content` |
| Bounded memory | `trim_history()` and `MAX_TURNS_KEPT` |
| Configurable endpoint | `--model`/`--base-url` flags, falling back to `DMR_MODEL`/`DMR_BASE_URL` env vars |
| Graceful failure | `except APIError` around the streaming call, `except APIConnectionError` in `single_prompt.py` |

## Gotchas reference

| Symptom | Cause | Fix |
| --- | --- | --- |
| `[error] Could not reach model` | Docker Model Runner isn't running or the model wasn't pulled | See [Step 1 — Environment Setup](02-environment-setup.md) |
| Replies drift or forget early context after a long chat | `MAX_TURNS_KEPT` window dropped older turns | Expected behavior — see [Step 6](07-bounded-memory.md); raise the constant for a longer window |
| First reply is slow | Local model loading into memory | Normal on the first call after Docker Model Runner (re)starts |
| `404` on `/v1/chat/completions` | Docker Desktop version exposes the route at `/engines/v1` | `--base-url http://localhost:12434/engines/v1` |

## Discussion questions

Good for a short group discussion before moving to Step 2 of the series:

1. What specifically would have to change for this app to become an agent, per the three-questions test in [Step 0](01-overview-and-concepts.md#why-this-isnt-just-a-chatbot-with-extra-steps)?
2. `trim_history()` drops the *oldest* turns. What's a scenario where that's the wrong strategy, and what would you do instead?
3. Why does `single_prompt.py` catch `APIConnectionError` but `chat.py` catches the broader `APIError`? (Hint: look at what `stream=True` can fail on that a single non-streaming call can't.)
4. The system prompt is sent on *every single request* even though it never changes. Is that wasteful? What would you check before deciding it's a problem worth solving?

## Exercises

Roughly ordered easiest to hardest — good as a follow-up lab:

1. **Add a `/reset` command.** Typing `/reset` at the `You:` prompt should clear `messages` back to just the system prompt, without exiting the app.
2. **Show the turn count.** Print how many turns are currently held in memory (e.g., `[3/12 turns]`) somewhere in the banner or prompt, so trimming behavior is visible without reading the code.
3. **Make the system prompt configurable.** Add a `--system-prompt` flag (or read it from a file) instead of hardcoding `SYSTEM_PROMPT`.
4. **Log the conversation.** Write each turn to a local file (e.g., `transcript.jsonl`, one JSON object per line) as it happens, so a session can be replayed or reviewed later.
5. **Swap the trimming strategy.** Instead of dropping the oldest turns, summarize them into a single system-style message before dropping them (a preview of the "long-term memory via summarization" pattern used in more advanced agents).
6. **Rebuild `single_prompt.py` from a blank file, unaided.** The best test of whether the concepts stuck: close this walkthrough and rewrite it from memory, checking against [../01-basic-chat-app/single_prompt.py](../01-basic-chat-app/single_prompt.py) only at the end.

## What's next

**[../02-chat-with-web-search/](../02-chat-with-web-search/README.md)** — the model gets one tool and has to decide, per the chatbot-vs-agent test from Step 0, whether, when, and how to use it. This is where the "plain LLM app" shape you just built starts to bend toward something closer to an agent.

# Step 3 — Chat Loop Skeleton

> [Back to index](README.md) · Previous: [The `single_prompt.py` Script](03-single-prompt-script.md) · Next: [Conversation Memory](05-conversation-memory.md)

## Goal

Build the REPL (read-eval-print loop) mechanics on their own, with a fake reply, before wiring in the real model. Getting the loop's exit conditions and edge cases right is easier without an API call in the way.

## Why this matters

A surprising amount of "chat app" code has nothing to do with the LLM at all — it's ordinary interactive-CLI hygiene: how do you exit cleanly, what happens on an empty line, what happens if the user hits Ctrl-C mid-input. Get this scaffolding solid first, then the next steps only ever touch the middle of the loop.

Create the file:

```bash
touch chat.py
```

## 1. Write the loop shell

```python
#!/usr/bin/env python3
"""Step 1 — a basic chat app: one LLM call per turn, no tools, no loop.

Run:
    uv run chat.py
"""


def main() -> None:
    print("Chatting — type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not user_input:
            continue

        print(f"Assistant: (pretend reply to: {user_input!r})")


if __name__ == "__main__":
    main()
```

Walk through each guard clause with trainees — each one exists because of a specific way a real terminal session misbehaves:

- **`try`/`except (EOFError, KeyboardInterrupt)`** — catches Ctrl-D (end of input, raises `EOFError`) and Ctrl-C (raises `KeyboardInterrupt`) so the program exits with a clean "Bye!" instead of a traceback. This is the single most common rough edge in hand-rolled CLI loops — without it, every interrupted demo ends in a stack trace on screen.
- **`.strip()`** — trims trailing whitespace/newlines so `"exit "` and `"exit"` both match.
- **`if user_input.lower() in {"exit", "quit"}`** — case-insensitive exit, checked *before* we do anything else with the input.
- **`if not user_input: continue`** — an empty line (just pressing Enter) silently loops back for another prompt, rather than sending a blank message anywhere.

## Try it

```bash
uv run chat.py
```

Confirm all three exit paths work: typing `exit`, typing `quit`, and pressing Ctrl-C. Also confirm pressing Enter on an empty line just re-prompts without printing anything.

## Checkpoint

<details>
<summary>Full <code>chat.py</code> (skeleton stage)</summary>

```python
#!/usr/bin/env python3
"""Step 1 — a basic chat app: one LLM call per turn, no tools, no loop.

Run:
    uv run chat.py
"""


def main() -> None:
    print("Chatting — type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not user_input:
            continue

        print(f"Assistant: (pretend reply to: {user_input!r})")


if __name__ == "__main__":
    main()
```

</details>

This is a deliberate scaffold, not final code — the `print(f"Assistant: (pretend reply...")` line disappears in the next step, replaced with a real API call. Nothing here appears verbatim in the finished [chat.py](../01-basic-chat-app/chat.py), but everything here survives inside it.

Next: **[Conversation Memory](05-conversation-memory.md)** — replace the fake reply with a real model call, and give the loop something to remember.

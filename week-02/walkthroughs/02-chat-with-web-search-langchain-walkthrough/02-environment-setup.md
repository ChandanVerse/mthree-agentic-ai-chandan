# Step 1 — Environment Setup

> [Back to index](README.md) · Previous: [Overview and Concepts](01-overview-and-concepts.md) · Next: [Defining the Tool](03-defining-the-tool.md)

## Goal

Get the local model reachable (if it isn't already from an earlier step in the series) and scaffold this app's project with its one new dependency: `langchain-openai`.

## Why this matters

Nothing about the local-model setup is specific to this app — if you've already worked through [00-local-model-setup](../00-local-model-setup/README.md) or built [02-chat-with-web-search](../02-chat-with-web-search/), skip straight to scaffolding below. The one thing worth pausing on is dependency weight: `langchain-openai` alone doesn't just add itself, it pulls in `langchain-core`, `langsmith`, `tiktoken`, and their transitive dependencies. That's a real cost of the simplification this walkthrough is about to demonstrate — worth naming up front rather than discovering by surprise later.

## 1. Confirm the local model is reachable

```bash
curl http://localhost:12434/v1/models
```

If this doesn't return a JSON list of models, work through [00-local-model-setup/README.md](../00-local-model-setup/README.md) first — everything downstream depends on this working.

## 2. Scaffold the project

```bash
uv init chat-with-web-search-langchain
cd chat-with-web-search-langchain
uv add ddgs langchain-openai
rm main.py
```

`ddgs` is the same no-API-key DuckDuckGo client the hand-rolled version uses for `web_search` — the tool's backend doesn't change between the two apps, only how the model is told about it and how its decision to call it gets parsed.

## Checkpoint

Your folder should look like:

```text
chat-with-web-search-langchain/
├── .gitignore
├── .venv/
├── pyproject.toml
└── uv.lock
```

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `curl: (7) Failed to connect` | Docker Model Runner isn't running | Work through [00-local-model-setup](../00-local-model-setup/README.md) |
| `ModuleNotFoundError: No module named 'langchain_openai'` | `uv add langchain-openai` wasn't run, or you're not using `uv run` | Re-run `uv add langchain-openai`; always launch the script via `uv run chat_with_tools.py` |
| Install pulls in far more packages than `openai` alone did in the hand-rolled version | Expected — `langchain-openai` brings `langchain-core`, `langsmith`, `tiktoken` transitively | Not a problem to fix; run `uv tree` if you want to see the full dependency graph |

Next: **[Defining the Tool](03-defining-the-tool.md)**.

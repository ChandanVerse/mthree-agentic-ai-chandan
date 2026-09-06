# Step 1 — Environment Setup

> [Back to index](README.md) · Previous: [Overview and Concepts](01-overview-and-concepts.md) · Next: [Porting the Tools](03-porting-the-tools.md)

## Goal

Get the local model reachable (if it isn't already from an earlier step in the series) and scaffold this app's project with its new dependencies: `langchain`, `langchain-openai`, and `langgraph`.

## Why this matters

Nothing about the local-model setup is specific to this app — if you've already worked through [00-local-model-setup](../00-local-model-setup/README.md) or built [03-small-agent](../03-small-agent/), skip straight to scaffolding below. The one thing worth pausing on is dependency weight: this app pulls in three packages the hand-rolled version didn't need at all (`langchain`, `langchain-openai`, `langgraph`), each with its own transitive dependencies. That's a real cost of the simplification this walkthrough is about to demonstrate — worth naming up front rather than discovering by surprise later.

## 1. Confirm the local model is reachable

```bash
curl http://localhost:12434/v1/models
```

If this doesn't return a JSON list of models, work through [00-local-model-setup/README.md](../00-local-model-setup/README.md) first — everything downstream depends on this working.

## 2. Scaffold the project

```bash
uv init small-agent-langgraph
cd small-agent-langgraph
uv add ddgs langchain langchain-openai langgraph
rm main.py
```

`ddgs` is the same no-API-key DuckDuckGo client the hand-rolled version's `web_search` uses — the tools' actual behavior doesn't change between the two apps, only how the model is told about them and how its decision to call one gets parsed.

## Checkpoint

Your folder should look like:

```text
small-agent-langgraph/
├── .gitignore
├── .venv/
├── pyproject.toml
└── uv.lock
```

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `curl: (7) Failed to connect` | Docker Model Runner isn't running | Work through [00-local-model-setup](../00-local-model-setup/README.md) |
| `ModuleNotFoundError: No module named 'langchain'` or `'langgraph'` | One of the three new dependencies wasn't added, or you're not running via `uv run` | Re-run `uv add langchain langchain-openai langgraph`; always launch with `uv run agent.py` |
| Install pulls in far more packages than [03-small-agent](../03-small-agent/)'s `openai` + `pydantic` alone | Expected — `langchain`/`langgraph` bring their own transitive dependency trees | Not a problem to fix; run `uv tree` if you want to see the full graph |

Next: **[Porting the Tools](03-porting-the-tools.md)**.

# Step 1 — Environment Setup

> [Back to index](README.md) · Previous: [Overview and Concepts](01-overview-and-concepts.md) · Next: [Defining the Tool Contract](03-defining-the-tool-contract.md)

## Goal

Scaffold an empty `uv` project with the three dependencies this app needs, on top of a local model that's already running.

## Why this matters

Nothing about serving the model changes from [01-basic-chat-app](../01-basic-chat-app-walkthrough/02-environment-setup.md) — it's the same Docker Model Runner endpoint, the same Gemma tag. The only new setup is two extra libraries: `pydantic` for the tool's argument schema, and `ddgs` to actually reach the live web.

## 1. Confirm the local model is running

If you haven't already done this earlier in the series:

```bash
curl http://localhost:12434/v1/models
```

This should return a JSON list of models. If it doesn't, stop here and follow [00-local-model-setup/README.md](../00-local-model-setup/README.md) (or [01-basic-chat-app-walkthrough's environment setup](../01-basic-chat-app-walkthrough/02-environment-setup.md) for the full walkthrough of enabling it) before continuing.

## 2. Scaffold the project

```bash
uv init chat-with-web-search
cd chat-with-web-search
uv add ddgs openai pydantic
```

`uv add` installs all three libraries and records them in `pyproject.toml`: `openai` as the HTTP client talking to Docker Model Runner's OpenAI-compatible endpoint, `pydantic` for the tool's argument schema, and `ddgs` — a no-API-key DuckDuckGo search client — as the thing that actually does the searching.

Remove the placeholder file `uv init` created — we'll write our own:

```bash
rm main.py
```

The exact contents `uv init` writes into `pyproject.toml` (build backend, Python version pin) can vary by `uv` version — that's fine, it isn't something this walkthrough depends on. What matters is that `ddgs`, `openai`, and `pydantic` show up under `dependencies`.

## 3. Confirm `ddgs` can actually reach the web

This app needs real internet access — `ddgs` is a no-API-key wrapper, not a mock. Confirm it works before writing any app code around it:

```bash
uv run python -c "from ddgs import DDGS; print([r['title'] for r in DDGS().text('python programming language', max_results=3)])"
```

## Try it

Expect three article titles printed as a Python list, something like:

```text
['Python (programming language)', 'Python (programming language)', 'Welcome to Python.org']
```

Exact titles will vary — what matters is that it returned *something* rather than raising an exception.

## Checkpoint

Your folder should look like:

```text
chat-with-web-search/
├── .gitignore
├── .venv/
├── pyproject.toml
└── uv.lock
```

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `curl: (7) Failed to connect` | Docker Model Runner not running | See [00-local-model-setup](../00-local-model-setup/README.md) |
| `ddgs` call raises a timeout or connection error | No internet access from this machine, or a transient rate limit from the search backend | Retry; confirm general internet access works first |
| `ModuleNotFoundError: No module named 'ddgs'` | Ran `python` instead of `uv run python`, so the project's virtualenv wasn't used | Prefix with `uv run`, or activate `.venv` first |

Next: **[Defining the Tool Contract](03-defining-the-tool-contract.md)**.

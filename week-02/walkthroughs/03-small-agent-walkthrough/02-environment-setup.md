# Step 1 — Environment Setup

> [Back to index](README.md) · Previous: [Overview and Concepts](01-overview-and-concepts.md) · Next: [The Calculator Tool](03-the-calculator-tool.md)

## Goal

Scaffold an empty `uv` project with the three dependencies this app needs, on top of a local model that's already running.

## Why this matters

Nothing about serving the model or the dependency list changes from [02-chat-with-web-search-walkthrough's environment setup](../02-chat-with-web-search-walkthrough/02-environment-setup.md) — same Docker Model Runner endpoint, same Gemma tag, same three libraries. The one thing that's genuinely new here is `save_note`: it writes files to disk, so before you write a line of tool code, get the `.gitignore` entry in place so those files never get accidentally committed.

## 1. Confirm the local model is running

If you haven't already done this earlier in the series:

```bash
curl http://localhost:12434/v1/models
```

This should return a JSON list of models. If it doesn't, stop here and follow [00-local-model-setup/README.md](../00-local-model-setup/README.md) before continuing.

## 2. Scaffold the project

```bash
uv init small-agent
cd small-agent
uv add ddgs openai pydantic
```

`uv add` installs all three libraries and records them in `pyproject.toml`: `openai` as the HTTP client talking to Docker Model Runner's OpenAI-compatible endpoint, `pydantic` for every tool's argument schema, and `ddgs` for the live web search tool.

Remove the placeholder file `uv init` created — we'll write our own:

```bash
rm main.py
```

## 3. Gitignore the notes directory before it exists

`save_note` (built in Step 3) writes markdown files into `./agent_notes/`. Add that folder to `.gitignore` now, before any code can create it:

```bash
echo "agent_notes/" >> .gitignore
```

## Try it

Confirm `ddgs` can actually reach the web — this app needs real internet access, not a mock:

```bash
uv run python -c "from ddgs import DDGS; print([r['title'] for r in DDGS().text('python programming language', max_results=3)])"
```

Expect three article titles printed as a Python list:

```text
['Python (programming language)', 'Python (programming language)', 'Welcome to Python.org']
```

Exact titles will vary — what matters is that it returned *something* rather than raising an exception.

## Checkpoint

Your folder should look like:

```text
small-agent/
├── .gitignore
├── .venv/
├── pyproject.toml
└── uv.lock
```

with `agent_notes/` present in `.gitignore` even though the folder itself doesn't exist yet.

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `curl: (7) Failed to connect` | Docker Model Runner not running | See [00-local-model-setup](../00-local-model-setup/README.md) |
| `ddgs` call raises a timeout or connection error | No internet access from this machine, or a transient rate limit from the search backend | Retry; confirm general internet access works first |
| `agent_notes/` shows up in `git status` later in the walkthrough | The `.gitignore` line was skipped or added after the folder was created and already tracked | Add the line now; if a note file is already tracked, `git rm --cached` it |

Next: **[The Calculator Tool](03-the-calculator-tool.md)** — the first of three tools, and the only one that needs a genuinely new safety idea.

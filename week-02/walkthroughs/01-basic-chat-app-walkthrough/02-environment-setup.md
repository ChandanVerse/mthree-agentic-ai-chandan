# Step 1 — Environment Setup

> [Back to index](README.md) · Previous: [Overview and Concepts](01-overview-and-concepts.md) · Next: [The `single_prompt.py` Script](03-single-prompt-script.md)

## Goal

Get a local model answering `curl` requests, and scaffold an empty `uv` project ready for code. Nothing in this step is specific to this app — it's the same setup used by every app in the series, so you only do it once.

## Why this matters

We're calling a **local** model instead of a hosted API deliberately: no API key, no per-token cost, and it's fully reproducible offline once the model is pulled. The tradeoff — slower first response, needs a reasonably capable laptop — is worth it for a training environment.

## 1. Enable Docker Model Runner

In Docker Desktop: **Settings → Features in development → Beta features**, enable **Docker Model Runner**, and turn on **Enable host-side TCP support** on port `12434`.

Or from the CLI:

```bash
docker desktop enable model-runner --tcp=12434
```

Confirm it's reachable:

```bash
curl http://localhost:12434/v1/models
```

## 2. Pull the model

```bash
docker model pull docker.io/ai/gemma4:E4B
```

Sanity-check it directly, before any Python is involved:

```bash
docker model run docker.io/ai/gemma4:E4B "Say hi in five words."
```

If this doesn't return text, stop here and fix it — every later step depends on this working. See the troubleshooting table below.

## 3. Scaffold the project

```bash
uv init basic-chat-app
cd basic-chat-app
uv add openai
```

`uv init` creates `pyproject.toml`, a `.gitignore`, and a `.venv`. `uv add openai` installs the `openai` Python client and records it as a dependency — we use it purely as an HTTP client for Docker Model Runner's OpenAI-compatible API, not to talk to OpenAI itself.

Remove the placeholder `main.py` uv created — we'll write our own files:

```bash
rm main.py
```

## Checkpoint

Your folder should look like:

```text
basic-chat-app/
├── .gitignore
├── .venv/
├── pyproject.toml
└── uv.lock
```

And this should print a JSON list of available models:

```bash
curl http://localhost:12434/v1/models
```

## Common mistakes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `curl: (7) Failed to connect` | Docker Model Runner not enabled, or host-side TCP support is off | Re-check step 1; run `docker model status` if your version has it |
| `docker model pull` hangs or fails | No internet, or Docker Desktop needs an update | Check Docker Desktop version supports Model Runner |
| App errors with "model not found" later | Model tag was never pulled, or a typo in the tag | `docker model pull docker.io/ai/gemma4:E4B` |
| `404` on `/v1/chat/completions` in a later step | This Docker Desktop version exposes the OpenAI route at `/engines/v1` instead of `/v1` | Use `--base-url http://localhost:12434/engines/v1` when running the scripts |
| `uv: command not found` | `uv` isn't installed | Install from [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |

Next: **[The `single_prompt.py` Script](03-single-prompt-script.md)**.

import argparse
import os
import sys
from openai import OpenAI, APIError

DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b")
API_KEY = "ollama"

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer directly; "
    "ask a clarifying question only if the request is genuinely ambiguous."
)

MAX_TURNS_KEPT = 12

def trim_history(messages: list[dict], max_turns: int = MAX_TURNS_KEPT) -> list[dict]:
    """Keep the system prompt plus only the most recent max_turns turns."""
    if not messages:
        return messages
    system, *turns = messages
    windowed = turns[-max_turns * 2:]
    return [system, *windowed]

def run_chat_app(base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL, api_key: str = API_KEY):
    client = OpenAI(base_url=base_url, api_key=api_key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Chatting with {model} via {base_url} (bounded history) - type 'exit' or 'quit' to stop.\n")

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

        messages.append({"role": "user", "content": user_input})
        messages = trim_history(messages)

        print("Assistant: ", end="", flush=True)
        reply_chunks = []
        try:
            stream = client.chat.completions.create(model=model, messages=messages, stream=True)
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    print(token, end="", flush=True)
                    reply_chunks.append(token)
        except APIError as e:
            print(f"\n[error] Could not reach model '{model}': {e}")
            break

        print("\n")
        messages.append({"role": "assistant", "content": "".join(reply_chunks)})

def main():
    parser = argparse.ArgumentParser(description="Full basic chat application with streaming and bounded history.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

    run_chat_app(base_url=args.base_url, model=args.model)

if __name__ == "__main__":
    main()

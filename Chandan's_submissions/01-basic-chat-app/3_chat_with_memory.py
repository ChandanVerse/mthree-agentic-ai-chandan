import os
from openai import OpenAI, APIError

DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b")
API_KEY = "ollama"

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer directly; "
    "ask a clarifying question only if the request is genuinely ambiguous."
)

def main():
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=API_KEY)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Chatting with {DEFAULT_MODEL} (with memory) - type 'exit' or 'quit' to stop.\n")

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
        try:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
            )
            reply = response.choices[0].message.content
            print(f"Assistant: {reply}\n")
            messages.append({"role": "assistant", "content": reply})
        except APIError as e:
            print(f"[error] {e}")
            break

if __name__ == "__main__":
    main()

import os
from openai import OpenAI, APIConnectionError

DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b")
API_KEY = "ollama"

def main():
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=API_KEY)
    prompt = "What is an AI agent in one sentence?"

    print(f"Sending prompt to {DEFAULT_MODEL} at {DEFAULT_BASE_URL}...\n")
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        print("Response:")
        print(response.choices[0].message.content)
    except APIConnectionError as e:
        print(f"[error] Could not reach endpoint: {e}")

if __name__ == "__main__":
    main()

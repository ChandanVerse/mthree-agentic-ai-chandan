import argparse
import os
import sys
from ddgs import DDGS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import APIError

DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b")
API_KEY = "ollama"

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a web_search tool. "
    "Only call it when you genuinely need current or external information - "
    "not for things you already know."
)

MAX_RESULTS = 5
MAX_TOOL_RETRIES = 2

@tool
def web_search(query: str) -> dict:
    """Search the web for current information about a topic.

    Use this tool only when you need real-time or recent information
    that you don't already know - news, current events, prices, etc.

    Args:
        query: A focused search query (not a question, not a URL).

    Returns:
        A dict with 'results' (list of {title, url, body}) and metadata.
    """
    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=MAX_RESULTS))
    results = [{"title": r["title"], "url": r["href"], "body": r["body"]} for r in raw]
    return {"results": results, "has_more": len(raw) == MAX_RESULTS, "total": len(results)}

TOOLS = [web_search]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

def run_turn(llm_with_tools, messages: list) -> str:
    """Handle one user turn: at most one tool hop, bounded retries, then final answer."""
    for _ in range(MAX_TOOL_RETRIES + 1):
        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            return response.content

        messages.append(response)
        retry_needed = False

        for call in response.tool_calls:
            tool_fn = TOOLS_BY_NAME.get(call["name"])
            if tool_fn is None:
                content = f"Unknown tool {call['name']!r}. Available: {list(TOOLS_BY_NAME)}."
                retry_needed = True
            else:
                try:
                    content = str(tool_fn.invoke(call["args"]))
                except Exception as e:
                    content = f"Tool {call['name']!r} failed: {e}"
                    retry_needed = True

            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

        if not retry_needed:
            return llm_with_tools.invoke(messages).content

    return "I could not complete that after a few tries - please rephrase."

def chat_loop(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, api_key: str = API_KEY) -> None:
    """Interactive REPL loop with LangChain tool calling."""
    llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)
    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]

    print(f"Chatting with {model} via {base_url} (web search enabled)")
    print("Type 'exit' or 'quit' to leave, Ctrl+C to interrupt.\n")

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

        messages.append(HumanMessage(content=user_input))
        try:
            answer = run_turn(llm_with_tools, messages)
        except APIError as e:
            print(f"[error] {e}. Is Ollama running (ollama serve)?")
            break

        print(f"Assistant: {answer}\n")
        # Single place responsible for appending visible reply into history
        messages.append(AIMessage(content=answer))

def main():
    parser = argparse.ArgumentParser(description="Chat app with LangChain tool calling and web search.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

    chat_loop(model=args.model, base_url=args.base_url)

if __name__ == "__main__":
    main()

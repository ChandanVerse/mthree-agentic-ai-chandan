import os
from ddgs import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b")
API_KEY = "ollama"

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a web_search tool. "
    "Only call it when you genuinely need current or external information - "
    "not for things you already know."
)

@tool
def web_search(query: str) -> dict:
    """Search the web for current information about a topic.

    Args:
        query: A focused search query string.

    Returns:
        A dict with 'results' and metadata.
    """
    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=3))
    results = [{"title": r["title"], "url": r["href"], "body": r["body"]} for r in raw]
    return {"results": results, "has_more": len(raw) == 3, "total": len(results)}

TOOLS = [web_search]

def main():
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    test_questions = [
        "What is 12 * 7?",                               # Should NOT trigger tool
        "Who won the most recent F1 world championship?", # SHOULD trigger tool
    ]

    print(f"Testing raw model decisions with {MODEL} via {BASE_URL}...\n")
    for q in test_questions:
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=q)]
        response = llm_with_tools.invoke(messages)

        print(f"Q: {q}")
        print(f"  tool_calls : {response.tool_calls}")
        print(f"  content    : {response.content!r}")
        print()

if __name__ == "__main__":
    main()

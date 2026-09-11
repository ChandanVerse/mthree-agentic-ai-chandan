import os
from ddgs import DDGS
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
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
    """Search the web for current information."""
    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=3))
    results = [{"title": r["title"], "url": r["href"], "body": r["body"]} for r in raw]
    return {"results": results, "has_more": len(raw) == 3, "total": len(results)}

TOOLS = [web_search]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

def run_turn_v1(llm_with_tools, messages: list) -> str:
    response = llm_with_tools.invoke(messages)

    if not response.tool_calls:
        return response.content

    messages.append(response)

    for call in response.tool_calls:
        tool_fn = TOOLS_BY_NAME[call["name"]]
        content = str(tool_fn.invoke(call["args"]))
        messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

    return llm_with_tools.invoke(messages).content

def main():
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    question = "What is the current price of Bitcoin?"
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]

    print(f"Question: {question}")
    answer = run_turn_v1(llm_with_tools, messages)
    print(f"Answer:\n{answer}")

if __name__ == "__main__":
    main()

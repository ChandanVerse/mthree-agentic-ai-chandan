import os
from ddgs import DDGS
from langchain_core.tools import tool

MAX_RESULTS = 5

@tool
def web_search(query: str) -> dict:
    """Search the web for current information about a topic.

    Args:
        query: A focused search query string.

    Returns:
        A dict with 'results' and metadata.
    """
    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=MAX_RESULTS))
    results = [{"title": r["title"], "url": r["href"], "body": r["body"]} for r in raw]
    return {"results": results, "has_more": len(raw) == MAX_RESULTS, "total": len(results)}

TOOLS = [web_search]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

def main():
    print("=== Tool Definition Inspection ===")
    print("Tool name:   ", web_search.name)
    print("Description: ", web_search.description)
    print("Input schema:", web_search.args)
    print()

    print("=== Direct Invocation Test ===")
    query = "Python programming language"
    print(f"Searching for: '{query}'...")
    res = web_search.invoke({"query": query})
    print(f"Results received: {res['total']} items (has_more: {res['has_more']})")
    if res["results"]:
        print(f"First result title: {res['results'][0]['title']}")
        print(f"First result URL:   {res['results'][0]['url']}")

if __name__ == "__main__":
    main()

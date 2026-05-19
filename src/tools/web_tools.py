"""
Web 搜索工具（LangGraph 版）
"""

from langchain_core.tools import tool

_web_searcher = None


def init_web_tools(web_searcher):
    global _web_searcher
    _web_searcher = web_searcher


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the general web (DuckDuckGo) for non-academic sources:
    GitHub repos, blog posts, technical discussions, project pages.
    Use when academic search doesn't have enough info."""
    if not query.strip():
        return "[Error] query is required"
    results = _web_searcher.search(query, max_results=max_results)
    if not results:
        return f"[No results] Web search returned nothing for '{query}'. The search service may be unavailable."
    lines = [f"- **{r.title}**\n  URL: {r.url}\n  Snippet: {r.snippet}" for r in results]
    return f"Found {len(results)} results:\n\n" + "\n\n".join(lines)


@tool
def web_fetch(url: str, max_chars: int = 3000) -> str:
    """Fetch the text content of a specific URL (blog post, GitHub README, etc.).
    Returns plain text with HTML tags stripped. Use after web_search to read a result in detail."""
    if not url.strip():
        return "[Error] url is required"
    text = _web_searcher.fetch(url, max_chars=max_chars)
    if text is None:
        return f"[Error] Failed to fetch {url}"
    return text

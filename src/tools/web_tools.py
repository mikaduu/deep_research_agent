"""
Web 搜索工具：通用网页搜索 + URL 正文抓取

让 Agent 能突破学术数据库的信息墙，搜到：
- GitHub 开源实现
- 技术博客
- 会议讨论 / OpenReview
- Twitter 学术动态
"""

from ..services.web_searcher import WebSearcher
from .tool import Tool, ToolResult


def build_web_search_tool(searcher: WebSearcher) -> Tool:
    def run(args):
        query = (args.get("query") or "").strip()
        max_results = int(args.get("max_results", 5))
        if not query:
            return ToolResult(success=False, error="query is required")
        results = searcher.search(query, max_results=max_results)
        if not results:
            return ToolResult(
                success=True,
                content={"count": 0, "results": [], "note": "No results found or search service unavailable"},
            )
        return ToolResult(
            success=True,
            content={
                "count": len(results),
                "results": [
                    {"title": r.title, "url": r.url, "snippet": r.snippet}
                    for r in results
                ],
            },
        )

    return Tool(
        name="web_search",
        description=(
            "Search the general web (DuckDuckGo) for non-academic sources: "
            "GitHub repos, blog posts, technical discussions, project pages. "
            "Use when academic search (arxiv/s2) doesn't have enough info, "
            "or when you need to check if a method has open-source implementations."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query in English",
                },
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        run=run,
    )


def build_web_fetch_tool(searcher: WebSearcher) -> Tool:
    def run(args):
        url = (args.get("url") or "").strip()
        if not url:
            return ToolResult(success=False, error="url is required")
        text = searcher.fetch(url, max_chars=int(args.get("max_chars", 5000)))
        if text is None:
            return ToolResult(success=False, error=f"Failed to fetch {url}")
        return ToolResult(success=True, content=text)

    return Tool(
        name="web_fetch",
        description=(
            "Fetch the text content of a specific URL (blog post, GitHub README, etc.). "
            "Returns plain text (HTML tags stripped), truncated to max_chars. "
            "Use after web_search to read a promising result in detail."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
                "max_chars": {"type": "integer", "default": 5000},
            },
            "required": ["url"],
        },
        run=run,
    )

"""
搜索工具（LangGraph 版）

用 @tool 装饰器定义，LangGraph 自动生成 schema。
工具函数接受的是具体参数（不是 dict），类型安全。
"""

from typing import List
from langchain_core.tools import tool


# 全局实例（由 create_tools 时注入）
_arxiv_searcher = None
_s2_searcher = None


def init_search_tools(arxiv_searcher, s2_searcher):
    """初始化搜索工具依赖的 service 实例。"""
    global _arxiv_searcher, _s2_searcher
    _arxiv_searcher = arxiv_searcher
    _s2_searcher = s2_searcher


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arXiv for academic papers. Use short English keyword queries (3-8 words).
    Returns paper titles, authors, abstracts and URLs.
    If rate-limited, switch to search_semantic_scholar instead."""
    if not query.strip():
        return "[Error] query is required"
    papers = _arxiv_searcher.search(query, max_results=max_results)
    if not papers:
        return f"[No results] arXiv returned 0 papers for '{query}'. Try a different query or use search_semantic_scholar."
    lines = []
    for p in papers[:max_results]:
        lines.append(f"- **{p.title}**\n  Authors: {', '.join(p.authors[:3])}\n  Abstract: {(p.abstract or '')[:300]}\n  URL: {p.url}\n  ID: {p.paper_id}")
    return f"Found {len(papers)} papers:\n\n" + "\n\n".join(lines)


@tool
def search_semantic_scholar(query: str, max_results: int = 5) -> str:
    """Search Semantic Scholar for academic papers. Broader coverage than arXiv.
    Use as primary search when arXiv is rate-limited, or as a second opinion."""
    if not query.strip():
        return "[Error] query is required"
    papers = _s2_searcher.search(query, max_results=max_results)
    if not papers:
        return f"[No results] Semantic Scholar returned 0 papers for '{query}'."
    lines = []
    for p in papers[:max_results]:
        lines.append(f"- **{p.title}**\n  Authors: {', '.join(p.authors[:3])}\n  Abstract: {(p.abstract or '')[:300]}\n  URL: {p.url}")
    return f"Found {len(papers)} papers:\n\n" + "\n\n".join(lines)

"""
论文工具（LangGraph 版）
"""

from typing import Optional
from langchain_core.tools import tool

_fetcher = None
_analyzer = None
_arxiv_searcher = None
_memory = None


def init_paper_tools(fetcher, analyzer, arxiv_searcher, memory=None):
    global _fetcher, _analyzer, _arxiv_searcher, _memory
    _fetcher = fetcher
    _analyzer = analyzer
    _arxiv_searcher = arxiv_searcher
    _memory = memory


@tool
def fetch_paper_fulltext(arxiv_id: str) -> str:
    """Download an arXiv paper PDF and extract its sections (abstract, introduction,
    method, experiments, conclusion). Use when abstract alone is insufficient.
    Section text is truncated to 1500 chars each."""
    if not arxiv_id.strip():
        return "[Error] arxiv_id is required"
    ft = _fetcher.fetch_fulltext(arxiv_id.strip())
    if ft is None:
        return f"[Error] Unable to fetch or parse PDF for {arxiv_id}"
    sections_preview = {
        k: (v[:1500] + "..." if len(v) > 1500 else v)
        for k, v in ft.sections.items()
    }
    result = f"Paper: {arxiv_id} ({ft.num_pages} pages, {ft.num_chars} chars)\n\n"
    for section, text in sections_preview.items():
        result += f"## {section}\n{text}\n\n"
    return result[:5000]  # 防止过长


@tool
def analyze_paper(arxiv_id: str, focus: Optional[str] = None) -> str:
    """Deeply analyze one paper (downloads full text, does section-level map-reduce).
    Produces structured note with formulas, tables, critical view.
    Also saves note to workspace/paper_notes/ and indexes in vector store.
    Use sparingly on the 2-3 most important papers. Slower but much richer."""
    if not arxiv_id.strip():
        return "[Error] arxiv_id is required"
    metas = _arxiv_searcher.search(f"id:{arxiv_id.strip()}", max_results=1)
    if not metas:
        return f"[Error] Paper metadata not found for {arxiv_id}"

    result = _analyzer.analyze(metas[0], focus=focus, use_fulltext=True)

    # 写入 vector store
    if _memory is not None:
        tldr = result.get("tldr", "")
        contributions = "\n".join(f"- {c}" for c in result.get("contributions", []))
        try:
            _memory.vector.add(
                doc_id=f"paper:{arxiv_id.strip()}",
                content=f"{metas[0].title}\n{tldr}\n\nContributions:\n{contributions}",
                metadata={"type": "paper_note", "arxiv_id": arxiv_id.strip(),
                          "method_name": result.get("_method_name", arxiv_id)},
            )
        except Exception:
            pass

    # 返回摘要给 Agent
    summary = f"**{metas[0].title}**\n\n"
    summary += f"TL;DR: {result.get('tldr', 'N/A')}\n\n"
    summary += f"Problem: {result.get('problem', 'N/A')}\n\n"
    if result.get("contributions"):
        summary += "Contributions:\n" + "\n".join(f"- {c}" for c in result["contributions"]) + "\n\n"
    if result.get("results"):
        summary += f"Results: {result['results']}\n\n"
    if result.get("_note_path"):
        summary += f"[Note saved: {result['_note_path']}]"
    return summary[:3000]

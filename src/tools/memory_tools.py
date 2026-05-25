"""
记忆工具（LangGraph 版）
"""

from langchain_core.tools import tool

_memory = None
_reflection = None


def init_memory_tools(memory_manager, reflection_engine):
    global _memory, _reflection
    _memory = memory_manager
    _reflection = reflection_engine


@tool
def retrieve_memory(query: str) -> str:
    """Search the agent's long-term memory (episodic / skill / vector) for prior
    research related to the query. Call this FIRST at the start of research to
    avoid duplicating prior work. Returns formatted context from all memory layers."""
    if not query.strip():
        return "[Error] query is required"
    try:
        text = _memory.format_context_for_prompt(query)
    except Exception as e:
        return f"[Error] Memory retrieval failed: {str(e)[:200]}"
    if not text:
        return "(No relevant memory found — this appears to be a new topic)"
    return text


@tool
def save_note(doc_id: str, title: str, body: str) -> str:
    """Save an intermediate research note to the vector store.
    Use when you finish investigating a sub-topic and want it searchable later.
    Recommended doc_id format: 'task:<slug>' or 'finding:<slug>'."""
    if not doc_id or not body:
        return "[Error] doc_id and body are required"
    try:
        _memory.save_task_result(doc_id, title or doc_id, body)
    except Exception as e:
        return f"[Error] Failed to save note: {str(e)[:200]}"
    return f"[Saved] Note '{doc_id}' stored in vector memory."


@tool
def save_research_episode(topic: str, final_report: str, quality_hint: float = 0.6) -> str:
    """Save a completed research session and trigger self-learning (extract insights + skills).
    Call this ONCE at the very end, right before finishing.
    quality_hint: your self-assessed quality 0-1."""
    if not topic or not final_report:
        return "[Error] topic and final_report are required"
    try:
        from ..core.models import ResearchResult
        stub = ResearchResult(
            topic=topic, plan=[], task_results=[],
            final_report_markdown=final_report, report_file="", papers=[],
        )
        out = _reflection.reflect(stub)
    except Exception as e:
        return f"[Error] Reflection failed: {str(e)[:200]}"
    return (
        f"[Saved] Episode '{out.get('episode_id')}' stored. "
        f"Quality: {out.get('quality_score', 0):.2f}, "
        f"Skills learned: {out.get('skills_learned', 0)}, "
        f"Tags: {out.get('tags', [])}"
    )

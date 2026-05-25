"""
委派工具（LangGraph 版）

Critic 和 Reviser 作为 subgraph 被调用，但对外暴露为普通 @tool。
每次调用新建独立实例保证角色隔离。
"""

from langchain_core.tools import tool

_settings = None


def init_delegation_tools(settings):
    global _settings
    _settings = settings


@tool
def delegate_to_critic(topic: str, report_md: str) -> str:
    """Send a research report to an independent Critic Agent for peer review.
    The Critic evaluates coverage, evidence quality, coherence, and actionability (each 0-1).
    Call this AFTER you have a complete draft report.
    If score < 0.7, the report needs revision."""
    from ..agents.critic_worker import CriticWorker

    if not topic.strip() or not report_md.strip():
        return "[Error] topic and report_md are required"

    try:
        critic = CriticWorker(_settings)
        review = critic.review(topic, report_md)
    except Exception as e:
        return f"[Error] Critic review failed: {str(e)[:200]}"
    return (
        f"**Critic Review**\n"
        f"- Score: {review.score:.2f} ({'needs revision' if review.needs_revision else 'acceptable'})\n"
        f"- Dimensions: {review.dimension_scores}\n"
        f"- Strengths: {review.strengths}\n"
        f"- Suggestions: {review.suggestions}\n"
        f"- Missing topics: {review.missing_topics}"
    )


@tool
def delegate_to_reviser(topic: str, report_md: str, suggestions: str, missing_topics: str) -> str:
    """Send a report to an independent Reviser Agent who will search for supplementary
    sources and rewrite the report addressing all suggestions.
    Call this when delegate_to_critic returns needs_revision.
    Pass suggestions and missing_topics as comma-separated strings."""
    from ..agents.reviser_worker import ReviserWorker
    from ..core.models import CriticReview

    if not topic.strip() or not report_md.strip():
        return "[Error] topic and report_md are required"

    review = CriticReview(
        score=0.5, needs_revision=True,
        dimension_scores={},
        suggestions=[s.strip() for s in suggestions.split(",") if s.strip()],
        missing_topics=[t.strip() for t in missing_topics.split(",") if t.strip()],
        strengths=[],
    )

    try:
        reviser = ReviserWorker(_settings)
        revised = reviser.revise(topic, report_md, review)
    except Exception as e:
        return f"[Error] Revision failed: {str(e)[:200]}"
    return revised

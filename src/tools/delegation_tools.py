"""
委派工具：把 CriticWorker / ReviserWorker 包装为 Manager 可调用的 Tool。

Worker 内部也是 autonomous loop（能自主验证引用、搜索补充材料），
但对 Manager 来说只是一次 tool call — 委派出去等结果回来。

注意：使用延迟导入避免循环依赖（tools → agents → tools）。
"""

from ..core.config import Settings
from ..core.models import CriticReview
from .tool import Tool, ToolResult


def build_delegate_to_critic_tool(settings: Settings) -> Tool:
    """每次调用都新建 CriticWorker（独立 LLM 实例，角色隔离）。"""
    def run(args):
        from ..agents.critic_worker import CriticWorker  # 延迟导入避免循环

        topic = (args.get("topic") or "").strip()
        report_md = (args.get("report_md") or "").strip()
        if not topic or not report_md:
            return ToolResult(success=False, error="topic and report_md are required")

        critic = CriticWorker(settings)
        review = critic.review(topic, report_md)
        return ToolResult(
            success=True,
            content={
                "score": review.score,
                "needs_revision": review.needs_revision,
                "dimension_scores": review.dimension_scores,
                "strengths": review.strengths,
                "suggestions": review.suggestions,
                "missing_topics": review.missing_topics,
            },
        )

    return Tool(
        name="delegate_to_critic",
        description=(
            "Send a research report to an independent Critic Agent for peer review. "
            "The Critic autonomously verifies citations and evaluates coverage, evidence "
            "quality, coherence, and actionability (each 0-1). Returns structured review. "
            "Call this AFTER you have a complete draft report. "
            "If score < 0.7, the report needs revision."
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The research topic being reviewed",
                },
                "report_md": {
                    "type": "string",
                    "description": "The complete Markdown report to review",
                },
            },
            "required": ["topic", "report_md"],
        },
        run=run,
    )


def build_delegate_to_reviser_tool(settings: Settings) -> Tool:
    """每次调用都新建 ReviserWorker（独立 LLM 实例）。"""
    def run(args):
        from ..agents.reviser_worker import ReviserWorker  # 延迟导入避免循环

        topic = (args.get("topic") or "").strip()
        report_md = (args.get("report_md") or "").strip()
        review_data = args.get("review") or {}
        if not topic or not report_md or not review_data:
            return ToolResult(
                success=False,
                error="topic, report_md, and review are all required",
            )

        review = CriticReview(
            score=float(review_data.get("score", 0.5)),
            needs_revision=True,
            dimension_scores=review_data.get("dimension_scores", {}),
            suggestions=review_data.get("suggestions", []),
            missing_topics=review_data.get("missing_topics", []),
            strengths=review_data.get("strengths", []),
        )

        reviser = ReviserWorker(settings)
        revised_report = reviser.revise(topic, report_md, review)
        return ToolResult(success=True, content=revised_report)

    return Tool(
        name="delegate_to_reviser",
        description=(
            "Send a report + critic review to an independent Reviser Agent. "
            "The Reviser autonomously searches for supplementary sources on missing_topics, "
            "then rewrites the report addressing all suggestions. Returns the revised "
            "Markdown report. Call this when delegate_to_critic returns needs_revision=true."
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "report_md": {
                    "type": "string",
                    "description": "The current report to revise",
                },
                "review": {
                    "type": "object",
                    "description": "The critic review object",
                    "properties": {
                        "score": {"type": "number"},
                        "dimension_scores": {"type": "object"},
                        "suggestions": {"type": "array", "items": {"type": "string"}},
                        "missing_topics": {"type": "array", "items": {"type": "string"}},
                        "strengths": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "required": ["topic", "report_md", "review"],
        },
        run=run,
    )

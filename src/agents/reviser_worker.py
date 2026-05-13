"""
ReviserWorker — 自主修订 Agent

继承 BaseAgent，内部 loop 可以：
- 分析评审意见，识别薄弱点
- 主动搜索补充材料（针对 missing_topics）
- 逐段重写报告
- 最终输出完整修订版报告
"""

from ..core.llm import LLMClient
from ..core.config import Settings
from ..core.models import CriticReview
from ..agents.base_agent import BaseAgent
from ..services.paper_search import ArxivSearcher, SemanticScholarSearcher
from ..tools import Tool, ToolRegistry, ToolResult
from ..tools.search_tools import build_arxiv_search_tool, build_s2_search_tool


REVISER_SYSTEM_PROMPT = """You are an expert research report editor. You receive a report along with a peer review critique, and your job is to revise the report to address all identified weaknesses.

## Your Tools
- **search_arxiv**: Search for supplementary papers to fill gaps identified by the critic.
- **search_semantic_scholar**: Alternative academic search.
- **finish**: Submit the revised report (complete Markdown text).

## Revision Strategy
1. Read the critic's suggestions and missing_topics carefully.
2. For each missing_topic, search for 1-2 relevant papers to get supporting evidence.
3. Rewrite the report:
   - Strengthen weak sections with new evidence
   - Add coverage for missing topics
   - Keep strong sections largely intact
   - Only cite URLs from papers you actually found via search
4. Call finish with the complete revised Markdown report.

## Rules
- The revised report must be COMPLETE (not just the changed sections)
- Preserve the original structure unless the critic specifically flagged structural issues
- Add new citations with real arXiv URLs from your searches
- Output language: same as the original report (usually Chinese)
- Do NOT invent citations — only use papers you found via search tools
"""


class ReviserWorker(BaseAgent):
    """自主修订 Agent，能主动搜索补充材料后重写。"""

    def __init__(self, settings: Settings, max_steps: int = 10, max_total_tokens: int = 50_000):
        self._settings = settings
        self._arxiv = ArxivSearcher(max_results=settings.search_top_k)
        self._s2 = SemanticScholarSearcher()
        super().__init__(
            llm=LLMClient(settings),
            max_steps=max_steps,
            max_total_tokens=max_total_tokens,
            temperature=0.3,
        )

    def system_prompt(self) -> str:
        return REVISER_SYSTEM_PROMPT

    def build_tools(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(build_arxiv_search_tool(self._arxiv))
        registry.register(build_s2_search_tool(self._s2))
        return registry

    def revise(self, topic: str, report_md: str, review: CriticReview) -> str:
        """对外接口：修订报告，返回修改后的 Markdown。"""
        suggestions = "\n".join(f"- {s}" for s in review.suggestions)
        missing = "\n".join(f"- {t}" for t in review.missing_topics)
        dim_text = "\n".join(f"  {k}: {v:.2f}" for k, v in review.dimension_scores.items())

        task = f"""Revise this research report based on the critic's feedback.

Topic: {topic}
Critic score: {review.score:.2f}
Dimension scores:
{dim_text}

Suggestions:
{suggestions}

Missing topics to cover:
{missing}

Original report:
---
{report_md}
---

Search for supplementary sources on the missing topics, then rewrite the report addressing all suggestions. Call finish with the complete revised report."""

        result = self.run(task)

        if result.finished:
            output = result.final_output
            if isinstance(output, dict):
                return output.get("output", str(output))
            return str(output)

        # 未正常完成时返回原报告
        return report_md

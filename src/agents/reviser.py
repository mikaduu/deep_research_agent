"""
修改 Agent（Reviser Agent）

使用独立 system prompt 隔离角色：
- 接收评审意见，针对薄弱部分补充搜索
- 将补充搜索结果和评审建议一起注入提示词，重写报告
- 与评审 Agent 和生成 Agent 角色完全隔离
"""

from typing import List

from ..core.llm import LLMClient
from ..core.models import CriticReview, SourceItem
from ..core.prompts import reviser_prompt
from ..services.paper_search import ArxivSearcher, SemanticScholarSearcher

REVISER_SYSTEM_PROMPT = (
    "You are an expert research report editor. "
    "You receive a research report along with a peer review critique. "
    "Your job is to revise the report to address all identified weaknesses. "
    "You are independent from both the original author and the reviewer — "
    "focus purely on improving the report quality based on the critique provided."
)


class ReviserAgent:
    """独立修改 Agent，根据评审意见补充搜索并重写报告。"""

    def __init__(
        self,
        llm: LLMClient,
        arxiv: ArxivSearcher,
        s2: SemanticScholarSearcher,
        temperature: float = 0.3,
    ):
        self.llm = llm
        self.arxiv = arxiv
        self.s2 = s2
        self.temperature = temperature

    def revise(
        self,
        topic: str,
        report_md: str,
        review: CriticReview,
        max_extra_sources: int = 6,
    ) -> str:
        """根据评审意见（含补充搜索）重写报告，返回修改后的 Markdown。"""
        extra_sources = self._fetch_extra_sources(review.missing_topics, max_extra_sources)

        messages = [
            {"role": "system", "content": REVISER_SYSTEM_PROMPT},
            {"role": "user", "content": reviser_prompt(topic, report_md, review, extra_sources)},
        ]
        return self.llm.invoke(messages, self.temperature)

    def _fetch_extra_sources(
        self, missing_topics: List[str], max_total: int
    ) -> List[SourceItem]:
        """针对评审指出的缺失主题补充搜索。"""
        if not missing_topics:
            return []

        sources: List[SourceItem] = []
        per_topic = max(1, max_total // len(missing_topics))

        for topic_query in missing_topics:
            papers = self.arxiv.search(topic_query, max_results=per_topic)
            if not papers:
                papers = self.s2.search(topic_query, max_results=per_topic)
            for i, p in enumerate(papers[:per_topic]):
                sources.append(
                    SourceItem(
                        title=p.title,
                        url=p.url,
                        snippet=p.abstract[:300],
                        rank=i,
                    )
                )
            if len(sources) >= max_total:
                break

        return sources[:max_total]

"""
评审 Agent（Critic Agent）

使用独立 system prompt 隔离角色，避免自我评价：
- 严格按照评判标准（覆盖度/证据质量/连贯性/可操作性）打分
- 给出具体修改建议和需要补充搜索的主题
- 与生成报告的 LLM 实例共享 API key，但角色完全隔离
"""

from typing import Dict, List

from ..core.llm import LLMClient
from ..core.models import CriticReview
from ..core.prompts import critic_prompt
from ..core.utils import extract_json_object

CRITIC_SYSTEM_PROMPT = (
    "You are a strict academic peer reviewer evaluating research reports. "
    "Your role is to identify weaknesses, gaps, and areas for improvement. "
    "You are independent from the report author — evaluate objectively based solely "
    "on the content quality, evidence, and logical coherence. "
    "Never be lenient. A score above 0.85 means the report is publication-ready."
)


class CriticAgent:
    """独立评审 Agent，通过 system prompt 与报告生成 Agent 角色隔离。"""

    def __init__(self, llm: LLMClient, threshold: float = 0.7, temperature: float = 0.1):
        self.llm = llm
        self.threshold = threshold
        self.temperature = temperature

    def review(self, topic: str, report_md: str) -> CriticReview:
        """对报告进行独立评审，返回结构化评审结果。"""
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": critic_prompt(topic, report_md)},
        ]
        raw = self.llm.invoke(messages, self.temperature)
        data = extract_json_object(raw)

        dim_scores: Dict[str, float] = {
            k: float(v)
            for k, v in data.get("dimension_scores", {}).items()
        }
        # 若 LLM 未返回 score，用维度均值兜底
        if dim_scores:
            score = float(data.get("score", sum(dim_scores.values()) / len(dim_scores)))
        else:
            score = float(data.get("score", 0.5))

        return CriticReview(
            score=round(score, 3),
            needs_revision=score < self.threshold,
            dimension_scores=dim_scores,
            suggestions=data.get("suggestions", []),
            missing_topics=data.get("missing_topics", []),
            strengths=data.get("strengths", []),
        )

"""
CriticWorker — 自主评审 Agent

继承 BaseAgent，内部 loop 可以：
- 逐章节扫描报告结构
- 验证引用 URL 是否真实存在
- 检查声明是否有证据支撑
- 最终输出结构化评审

与老 CriticAgent 的区别：
- 老版：一次 LLM 调用，凭直觉打分
- 新版：多步验证后打分，更可靠
"""

from ..core.llm import LLMClient
from ..core.models import CriticReview
from ..core.utils import extract_json_object
from ..agents.base_agent import BaseAgent
from ..tools import Tool, ToolRegistry, ToolResult

import requests


CRITIC_SYSTEM_PROMPT = """You are a strict academic peer reviewer. You evaluate research reports independently and objectively.

## Your Tools
- **check_url**: Verify if a citation URL actually exists (HTTP HEAD check).
- **finish**: Submit your final review with scores and suggestions.

## Evaluation Process
1. Read the report carefully.
2. Check 2-3 citation URLs to verify they're real (don't check all — just spot-check).
3. Evaluate four dimensions (each 0-1):
   - **coverage**: Does the report cover the topic's core aspects?
   - **evidence_quality**: Are citations real and relevant? Are claims backed by evidence?
   - **coherence**: Is the logic clear? Is the structure well-organized?
   - **actionability**: Are recommendations specific and executable?
4. Call finish with your structured review.

## Scoring Rules
- score = average of the four dimension scores
- A score above 0.85 means publication-ready
- Be strict: most first drafts score 0.5-0.7
- If you find a fake/broken citation, deduct 0.1 from evidence_quality

## Output Format (pass to finish)
Your finish arguments must be a JSON object with:
{
  "score": 0.0,
  "dimension_scores": {"coverage": 0.0, "evidence_quality": 0.0, "coherence": 0.0, "actionability": 0.0},
  "strengths": ["specific strength 1", "specific strength 2"],
  "suggestions": ["specific actionable suggestion 1", "..."],
  "missing_topics": ["topic needing more research"],
  "verified_urls": {"url1": true, "url2": false}
}
"""


class CriticWorker(BaseAgent):
    """自主评审 Agent，能验证引用后再打分。"""

    def __init__(self, settings, max_steps: int = 8, max_total_tokens: int = 30_000):
        self._settings = settings
        super().__init__(
            llm=LLMClient(settings),
            max_steps=max_steps,
            max_total_tokens=max_total_tokens,
            temperature=0.1,
        )

    def system_prompt(self) -> str:
        return CRITIC_SYSTEM_PROMPT

    def build_tools(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(self._build_check_url_tool())
        return registry

    def review(self, topic: str, report_md: str) -> CriticReview:
        """对外接口：评审报告，返回 CriticReview。"""
        task = f"Review this research report on topic: {topic}\n\n---\n\n{report_md}"
        result = self.run(task)

        if result.finished and isinstance(result.final_output, dict):
            data = result.final_output
        else:
            data = {}

        dim_scores = {
            k: float(v)
            for k, v in data.get("dimension_scores", {}).items()
        }
        score = float(data.get("score", 0.5))
        if dim_scores and not data.get("score"):
            score = sum(dim_scores.values()) / len(dim_scores)

        return CriticReview(
            score=round(score, 3),
            needs_revision=score < self._settings.critic_threshold,
            dimension_scores=dim_scores,
            suggestions=data.get("suggestions", []),
            missing_topics=data.get("missing_topics", []),
            strengths=data.get("strengths", []),
        )

    @staticmethod
    def _build_check_url_tool() -> Tool:
        def run(args):
            url = (args.get("url") or "").strip()
            if not url:
                return ToolResult(success=False, error="url is required")
            try:
                resp = requests.head(url, timeout=5, allow_redirects=True)
                exists = resp.status_code < 400
                return ToolResult(
                    success=True,
                    content={"url": url, "exists": exists, "status_code": resp.status_code},
                )
            except Exception as e:
                return ToolResult(
                    success=True,
                    content={"url": url, "exists": False, "error": str(e)[:100]},
                )

        return Tool(
            name="check_url",
            description="Verify if a URL exists by sending an HTTP HEAD request. Returns status code.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to check"},
                },
                "required": ["url"],
            },
            run=run,
        )

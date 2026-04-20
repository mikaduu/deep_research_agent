import json
from typing import Dict, Optional

from .llm import LLMClient
from .config import Settings
from .models import PaperItem


class PaperAnalyzer:
    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm = llm_client
        self.settings = settings

    def analyze(self, paper: PaperItem, focus: Optional[str] = None) -> Dict:
        prompt = self._build_prompt(paper, focus)
        response = self.llm.invoke([{"role": "user", "content": prompt}], temperature=0.2)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"summary": response, "contributions": [], "methods": [], "limitations": []}

    def _build_prompt(self, paper: PaperItem, focus: Optional[str]) -> str:
        focus_line = f"\n特别关注: {focus}" if focus else ""
        return f"""请深度分析以下论文:{focus_line}

标题: {paper.title}
作者: {', '.join(paper.authors[:5])}
发表时间: {paper.published}
摘要: {paper.abstract}

返回JSON:
{{
  "summary": "核心内容概述(200字以内)",
  "problem": "解决的核心问题",
  "contributions": ["贡献1", "贡献2"],
  "methods": ["方法1", "方法2"],
  "results": "主要实验结果",
  "limitations": ["局限1", "局限2"],
  "future_work": "未来工作方向",
  "relevance_score": 0.0
}}""".strip()

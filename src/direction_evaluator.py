from typing import Dict, List
import json

from .llm import LLMClient
from .config import Settings
from .models import PaperItem
from .services.paper_search import ArxivSearcher, SemanticScholarSearcher


class DirectionEvaluator:
    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm = llm_client
        self.settings = settings
        self.arxiv = ArxivSearcher(max_results=settings.search_top_k)
        self.s2 = SemanticScholarSearcher()

    def evaluate_direction(self, direction: str) -> Dict:
        papers_arxiv = self.arxiv.search(direction, max_results=8)
        papers_s2 = self.s2.search(direction, max_results=5)
        all_papers = papers_arxiv + papers_s2

        papers_summary = self._format_papers(all_papers[:10])
        prompt = self._build_evaluation_prompt(direction, papers_summary)

        messages = [{"role": "user", "content": prompt}]
        response = self.llm.invoke(messages, temperature=0.3)

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {"feasibility": 0.5, "novelty": 0.5, "impact": 0.5, "analysis": response}

        result["papers"] = all_papers[:10]
        return result

    def _format_papers(self, papers: List[PaperItem]) -> str:
        lines = []
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. {p.title}")
            lines.append(f"   Authors: {', '.join(p.authors[:3])}")
            lines.append(f"   Abstract: {p.abstract[:200]}...")
            lines.append(f"   URL: {p.url}\n")
        return "\n".join(lines)

    def _build_evaluation_prompt(self, direction: str, papers_summary: str) -> str:
        return f"""
你是一位资深的学术研究顾问。请评估以下研究方向的可行性和价值。

研究方向: {direction}

相关已有论文:
{papers_summary}

请从以下维度评估(0-1分):
1. feasibility (可行性): 技术上是否可实现，资源需求是否合理
2. novelty (新颖性): 是否有创新点，与现有工作的区别
3. impact (影响力): 潜在的学术/应用价值

返回JSON格式:
{{
  "feasibility": 0.0-1.0,
  "novelty": 0.0-1.0,
  "impact": 0.0-1.0,
  "analysis": "详细分析文本，包括: 1)现有研究现状总结 2)该方向的优势和挑战 3)建议的研究切入点",
  "recommendations": ["建议1", "建议2", "建议3"],
  "related_topics": ["相关主题1", "相关主题2"]
}}
""".strip()

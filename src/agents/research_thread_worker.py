"""
ResearchThreadWorker - 聚焦单个研究子线程的 autonomous worker。

用于把一个大主题拆成多个更窄的调研线程，例如：
- baseline 对比
- 相似方法线
- 局限性与开放问题
- 代表性工作与时间线
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..agents.base_agent import BaseAgent
from ..core.config import Settings
from ..core.llm import LLMClient
from ..services.paper_search import ArxivSearcher, SemanticScholarSearcher
from ..tools.paper_tools import build_analyze_paper_tool
from ..tools.search_tools import build_arxiv_search_tool, build_s2_search_tool
from ..tools.tool import ToolRegistry


THREAD_WORKER_SYSTEM_PROMPT = """You are a focused research sub-agent responsible for one narrow research thread inside a larger literature investigation.

## Your Goal
You only work on ONE sub-thread at a time, such as:
- baseline analysis
- representative methods in a line of work
- limitations and failure modes
- competing approaches
- evaluation protocols / datasets

## Your Tools
- **search_arxiv**: Search arXiv for candidate papers.
- **search_semantic_scholar**: Search Semantic Scholar for broader coverage.
- **analyze_paper**: Deeply analyze one important paper in conservative mode.
- **finish**: Return your structured thread result.

## Working Style
1. Search from 2-3 query angles.
2. Identify 2-5 representative papers for this thread.
3. Deeply analyze at most 1-2 key papers if needed.
4. Summarize only findings directly relevant to this thread.
5. Keep output concise, evidence-based, and Chinese.

## Constraints
- Search queries must be in English.
- Do not cover the whole field; stay tightly scoped to the assigned thread.
- If evidence is weak or sparse, explicitly lower confidence.
- If you only found one weak paper or highly noisy evidence, confidence should usually be <= 0.45.

## Output Format
Call finish with a JSON object using these keys:
{
  "thread_name": "baseline analysis",
  "focus": "what exactly this thread covered",
  "summary": "Chinese summary paragraph",
  "key_findings": ["finding 1", "finding 2"],
  "representative_papers": [
    {"title": "...", "url": "...", "reason": "..."}
  ],
  "baseline_papers": [
    {"title": "...", "url": "...", "reason": "..."}
  ],
  "open_questions": ["question 1"],
  "suggested_queries": ["query 1", "query 2"],
  "confidence": 0.0
}
"""


class ResearchThreadWorker(BaseAgent):
    """单线程调研 worker。"""

    def __init__(
        self,
        settings: Settings,
        max_steps: int = 10,
        max_total_tokens: int = 40_000,
    ):
        self._settings = settings
        self._arxiv = ArxivSearcher(max_results=settings.search_top_k)
        self._s2 = SemanticScholarSearcher()
        super().__init__(
            llm=LLMClient(settings),
            max_steps=max_steps,
            max_total_tokens=max_total_tokens,
            temperature=0.2,
        )

    def system_prompt(self) -> str:
        return THREAD_WORKER_SYSTEM_PROMPT

    def build_tools(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(build_arxiv_search_tool(self._arxiv))
        registry.register(build_s2_search_tool(self._s2))
        registry.register(build_analyze_paper_tool(
            analyzer=self._build_analyzer(),
            arxiv_searcher=self._arxiv,
            memory=None,
        ))
        return registry

    def run_thread(
        self,
        topic: str,
        thread_name: str,
        focus: str,
        seed_queries: List[str] | None = None,
    ) -> Dict[str, Any]:
        query_block = ""
        if seed_queries:
            query_block = "\nSuggested search angles:\n" + "\n".join(
                f"- {q}" for q in seed_queries if q
            )

        task = f"""Main topic: {topic}
Thread name: {thread_name}
Thread focus: {focus}{query_block}

Investigate this thread only. Search, optionally analyze 1-2 key papers, then call finish with the required structured JSON object.
All narrative fields should be written in Chinese.
"""

        result = self.run(task)
        if result.finished and isinstance(result.final_output, dict):
            return self._normalize_output(result.final_output, thread_name, focus)
        return self._normalize_output({}, thread_name, focus)

    def _build_analyzer(self):
        from ..agents.paper_analyzer import PaperAnalyzer

        return PaperAnalyzer(LLMClient(self._settings), self._settings)

    @staticmethod
    def _normalize_output(data: Dict[str, Any], thread_name: str, focus: str) -> Dict[str, Any]:
        confidence = data.get("confidence", 0.3)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.3
        confidence = max(0.0, min(1.0, confidence))

        return {
            "thread_name": data.get("thread_name") or thread_name,
            "focus": data.get("focus") or focus,
            "summary": data.get("summary", ""),
            "key_findings": ResearchThreadWorker._to_str_list(data.get("key_findings")),
            "representative_papers": ResearchThreadWorker._to_paper_list(
                data.get("representative_papers")
            ),
            "baseline_papers": ResearchThreadWorker._to_paper_list(
                data.get("baseline_papers")
            ),
            "open_questions": ResearchThreadWorker._to_str_list(data.get("open_questions")),
            "suggested_queries": ResearchThreadWorker._to_str_list(data.get("suggested_queries")),
            "confidence": confidence,
        }

    @staticmethod
    def _to_str_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        result: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return result

    @staticmethod
    def _to_paper_list(value: Any) -> List[Dict[str, str]]:
        if not isinstance(value, list):
            return []
        items: List[Dict[str, str]] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    items.append({"title": text, "url": "", "reason": ""})
                continue
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if title or url or reason:
                items.append({"title": title, "url": url, "reason": reason})
        return items

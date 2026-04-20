import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .config import Settings
from .llm import LLMClient
from .models import PaperItem, ResearchResult, TaskPlanItem, TaskRunResult, Citation, SourceItem
from .memory.retriever import RagMemory
from .memory.store import NoteStore
from .context_manager import ContextManager
from .direction_evaluator import DirectionEvaluator
from .paper_analyzer import PaperAnalyzer
from .services.paper_search import ArxivSearcher, SemanticScholarSearcher
from .prompts import planner_prompt, summarizer_prompt, reflection_prompt, reporter_prompt
from .utils import extract_json_object, truncate_text


class ResearchOrchestrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMClient(settings)
        self.memory = RagMemory()
        self.store = NoteStore(settings.workspace_dir / "notes")
        self.context = ContextManager(settings.context_max_chars)
        self.evaluator = DirectionEvaluator(self.llm, settings)
        self.analyzer = PaperAnalyzer(self.llm, settings)
        self.arxiv = ArxivSearcher(max_results=settings.search_top_k)
        self.s2 = SemanticScholarSearcher()
        settings.workspace_dir.mkdir(parents=True, exist_ok=True)
        (settings.workspace_dir / "reports").mkdir(exist_ok=True)

    def evaluate_direction(self, direction: str) -> dict:
        result = self.evaluator.evaluate_direction(direction)
        self._save_to_memory(f"direction:{direction}", direction, result.get("analysis", ""))
        return result

    def analyze_paper(self, paper: PaperItem, focus: Optional[str] = None) -> dict:
        return self.analyzer.analyze(paper, focus)

    def search_papers(self, query: str) -> List[PaperItem]:
        arxiv_papers = self.arxiv.search(query)
        s2_papers = self.s2.search(query, max_results=5)
        seen, papers = set(), []
        for p in arxiv_papers + s2_papers:
            if p.title not in seen:
                seen.add(p.title)
                papers.append(p)
        return papers

    def run_deep_research(self, topic: str) -> ResearchResult:
        plan = self._plan(topic)
        task_results, all_papers = [], []

        for task in plan:
            papers = self.search_papers(task.search_query)
            all_papers.extend(papers)
            sources = [SourceItem(title=p.title, url=p.url, snippet=p.abstract[:300], rank=i)
                       for i, p in enumerate(papers[:self.settings.search_top_k])]
            rag_hits = self.memory.retrieve(task.goal, self.settings.memory_top_k)
            rag_context = "\n".join(h.content for h in rag_hits)
            result = self._summarize(topic, task, sources, rag_context)
            task_results.append(result)
            self._save_to_memory(f"task:{task.title}", task.goal, result.summary_markdown)

        report_md = self._write_report(topic, task_results)
        report_file = self._save_report(topic, report_md)

        return ResearchResult(
            topic=topic, plan=plan, task_results=task_results,
            final_report_markdown=report_md, report_file=report_file,
            papers=all_papers,
        )

    def _plan(self, topic: str) -> List[TaskPlanItem]:
        prompt = planner_prompt(topic, self.settings.max_plan_items)
        raw = self.llm.invoke([{"role": "user", "content": prompt}], self.settings.planner_temperature)
        data = extract_json_object(raw)
        return [TaskPlanItem(**t) for t in data.get("tasks", [])]

    def _summarize(self, topic: str, task: TaskPlanItem, sources: List[SourceItem], rag_context: str) -> TaskRunResult:
        prompt = summarizer_prompt(topic, task, sources, rag_context)
        raw = self.llm.invoke([{"role": "user", "content": prompt}], self.settings.researcher_temperature)
        data = extract_json_object(raw)
        return TaskRunResult(
            task=task,
            summary_markdown=data.get("summary_markdown", raw),
            key_points=data.get("key_points", []),
            citations=[Citation(**c) for c in data.get("citations", [])],
            confidence=float(data.get("confidence", 0.5)),
            sources_used=sources,
        )

    def _write_report(self, topic: str, task_results: List[TaskRunResult]) -> str:
        prompt = reporter_prompt(topic, task_results)
        return self.llm.invoke([{"role": "user", "content": prompt}], self.settings.writer_temperature)

    def _save_report(self, topic: str, content: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{topic[:40].replace(' ', '_')}.md"
        path = self.settings.workspace_dir / "reports" / filename
        path.write_text(content, encoding="utf-8")
        return str(path)

    def _save_to_memory(self, doc_id: str, title: str, body: str):
        self.memory.add(doc_id, body, {"title": title})
        self.store.save_note(doc_id.replace(":", "_"), title, body, {})

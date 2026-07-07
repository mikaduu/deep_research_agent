"""
专题线程委派工具。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from ..core.config import Settings
from .tool import Tool, ToolResult


def build_delegate_research_threads_tool(settings: Settings) -> Tool:
    def run(args):
        from ..agents.research_thread_worker import ResearchThreadWorker

        topic = (args.get("topic") or "").strip()
        threads = args.get("threads") or []
        max_workers = int(args.get("max_workers", 3))
        min_confidence = float(args.get("min_confidence", 0.35))

        if not topic:
            return ToolResult(success=False, error="topic is required")
        if not isinstance(threads, list) or not threads:
            return ToolResult(success=False, error="threads must be a non-empty list")

        normalized_threads: List[Dict[str, Any]] = []
        for idx, item in enumerate(threads):
            if not isinstance(item, dict):
                continue
            thread_name = str(item.get("thread_name") or item.get("name") or "").strip()
            focus = str(item.get("focus") or "").strip()
            queries = item.get("queries") or item.get("seed_queries") or []
            if not thread_name or not focus:
                continue
            normalized_threads.append(
                {
                    "thread_name": thread_name,
                    "focus": focus,
                    "queries": [str(q).strip() for q in queries if str(q).strip()],
                    "_idx": idx,
                }
            )

        if not normalized_threads:
            return ToolResult(success=False, error="no valid thread definitions found")

        max_workers = max(1, min(max_workers, len(normalized_threads), 4))
        raw_results: List[Dict[str, Any]] = []

        def _run_thread(item: Dict[str, Any]) -> Dict[str, Any]:
            worker = ResearchThreadWorker(settings)
            result = worker.run_thread(
                topic=topic,
                thread_name=item["thread_name"],
                focus=item["focus"],
                seed_queries=item["queries"],
            )
            result["_idx"] = item["_idx"]
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_run_thread, item): item
                for item in normalized_threads
            }
            for future in as_completed(future_map):
                item = future_map[future]
                try:
                    raw_results.append(future.result())
                except Exception as exc:
                    raw_results.append(
                        {
                            "thread_name": item["thread_name"],
                            "focus": item["focus"],
                            "summary": "",
                            "key_findings": [],
                            "representative_papers": [],
                            "baseline_papers": [],
                            "open_questions": [],
                            "suggested_queries": item["queries"],
                            "confidence": 0.0,
                            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                            "_idx": item["_idx"],
                        }
                    )

        raw_results.sort(key=lambda x: x.get("_idx", 0))
        kept_results = [
            _strip_internal_fields(item)
            for item in raw_results
            if float(item.get("confidence", 0.0)) >= min_confidence
        ]
        dropped_results = [
            {
                "thread_name": item.get("thread_name", ""),
                "confidence": float(item.get("confidence", 0.0)),
                "reason": item.get("error") or "confidence below threshold",
            }
            for item in raw_results
            if float(item.get("confidence", 0.0)) < min_confidence
        ]

        return ToolResult(
            success=True,
            content={
                "topic": topic,
                "parallel": True,
                "thread_count": len(normalized_threads),
                "max_workers": max_workers,
                "min_confidence": min_confidence,
                "threads": kept_results,
                "dropped_threads": dropped_results,
            },
        )

    return Tool(
        name="delegate_research_threads",
        description=(
            "Split a research topic into 2-4 focused sub-threads and delegate them to "
            "independent research workers that search and analyze papers in parallel. "
            "Useful for baseline analysis, related methods, limitations, evaluation setups, "
            "or competing approaches. Returns structured Chinese findings for each thread, "
            "and automatically drops low-confidence thread results."
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Main research topic",
                },
                "threads": {
                    "type": "array",
                    "description": "Focused research threads to delegate",
                    "items": {
                        "type": "object",
                        "properties": {
                            "thread_name": {"type": "string"},
                            "focus": {"type": "string"},
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional English search seed queries",
                            },
                        },
                        "required": ["thread_name", "focus"],
                    },
                },
                "max_workers": {
                    "type": "integer",
                    "description": "Maximum concurrent workers, recommend 2-4",
                    "default": 3,
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Drop thread results below this confidence threshold",
                    "default": 0.35,
                },
            },
            "required": ["topic", "threads"],
        },
        run=run,
    )


def _strip_internal_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if not key.startswith("_")
    }

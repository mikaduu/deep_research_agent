from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TaskPlanItem:
    title: str
    goal: str
    search_query: str


@dataclass
class SourceItem:
    title: str
    url: str
    snippet: str
    rank: int = 0


@dataclass
class PaperItem:
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    url: str
    published: str = ""
    updated: str = ""
    categories: List[str] = field(default_factory=list)


@dataclass
class Citation:
    title: str
    url: str
    reason: str = ""


@dataclass
class TaskRunResult:
    task: TaskPlanItem
    summary_markdown: str
    key_points: List[str] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0
    sources_used: List[SourceItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryHit:
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CriticReview:
    score: float
    needs_revision: bool
    dimension_scores: Dict[str, float]
    suggestions: List[str] = field(default_factory=list)
    missing_topics: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)


@dataclass
class ReflectionResult:
    episode_id: str
    quality_score: float
    insights_summary: str
    tags: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    skills_learned: int = 0


@dataclass
class ResearchResult:
    topic: str
    plan: List[TaskPlanItem]
    task_results: List[TaskRunResult]
    final_report_markdown: str
    report_file: str
    papers: List[PaperItem] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    reflection: Optional[ReflectionResult] = None
    critic_reviews: List[CriticReview] = field(default_factory=list)
    revision_count: int = 0

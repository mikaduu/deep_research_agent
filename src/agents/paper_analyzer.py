"""
PaperAnalyzer - 深度论文分析 Agent

分析策略：
  1. 优先尝试下载全文 PDF 并按章节提取
  2. 按章节分块发给 LLM 逐段分析（map 阶段）
  3. 汇总各段分析结果做综合（reduce 阶段）
  4. 输出符合 paper-reader skill 质量标准的结构化笔记
  5. 如果全文获取失败，降级到摘要分析

质量标准（对齐 paper-reader skill）：
  - 零遗漏：每个公式、每个表格、每个图都要覆盖
  - 具体性：引用具体数字，不说"效果好"
  - 批判性：strengths/weaknesses 必须有具体证据
  - 概念链接：关键术语用 [[Concept]] 标记
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..core.llm import LLMClient
from ..core.config import Settings
from ..core.models import PaperItem
from ..core.utils import extract_json_object
from ..services.paper_fetcher import PaperFetcher, PaperFullText


SECTION_ORDER = [
    "abstract", "introduction", "related_work",
    "method", "experiments", "discussion", "conclusion",
]

CHUNK_CHAR_LIMIT = 6000

# 笔记输出目录
NOTES_SUBDIR = "paper_notes"


class PaperAnalyzer:
    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm = llm_client
        self.settings = settings
        self.fetcher = PaperFetcher(
            cache_dir=settings.workspace_dir / "pdf_cache"
        )
        self.notes_dir = settings.workspace_dir / NOTES_SUBDIR
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        paper: PaperItem,
        focus: Optional[str] = None,
        use_fulltext: bool = True,
    ) -> Dict:
        if use_fulltext:
            fulltext = self.fetcher.fetch_fulltext(paper.paper_id)
            if fulltext and fulltext.num_chars > 500:
                result = self._analyze_fulltext(paper, fulltext, focus)
                self._save_note(paper, result)
                return result

        result = self._analyze_abstract(paper, focus)
        self._save_note(paper, result)
        return result

    # ------------------------------------------------------------------ #
    # 全文分析（两段式 map-reduce）
    # ------------------------------------------------------------------ #

    def _analyze_fulltext(
        self, paper: PaperItem, fulltext: PaperFullText, focus: Optional[str],
    ) -> Dict:
        chunks = self._build_chunks(fulltext)
        chunk_summaries: List[Dict] = []
        for chunk_name, chunk_text in chunks:
            summary = self._analyze_chunk(paper, chunk_name, chunk_text, focus)
            chunk_summaries.append({"section": chunk_name, "summary": summary})

        final = self._synthesize(paper, chunk_summaries, focus)
        final["_source"] = "fulltext"
        final["_num_pages"] = fulltext.num_pages
        final["_num_chunks"] = len(chunk_summaries)
        final["_section_summaries"] = chunk_summaries
        return final

    def _build_chunks(self, ft: PaperFullText) -> List[tuple]:
        chunks: List[tuple] = []
        if ft.sections:
            for key in SECTION_ORDER:
                text = ft.sections.get(key)
                if not text:
                    continue
                if len(text) <= CHUNK_CHAR_LIMIT:
                    chunks.append((key, text))
                else:
                    for i, sub in enumerate(self._split_by_chars(text, CHUNK_CHAR_LIMIT)):
                        chunks.append((f"{key}_part{i+1}", sub))
        if not chunks:
            for i, sub in enumerate(self._split_by_chars(ft.raw_text, CHUNK_CHAR_LIMIT)):
                chunks.append((f"chunk_{i+1}", sub))
        return chunks

    @staticmethod
    def _split_by_chars(text: str, limit: int) -> List[str]:
        if len(text) <= limit:
            return [text]
        parts = []
        paragraphs = text.split("\n\n")
        buf = ""
        for p in paragraphs:
            if len(buf) + len(p) + 2 > limit and buf:
                parts.append(buf.strip())
                buf = p
            else:
                buf += ("\n\n" if buf else "") + p
        if buf:
            parts.append(buf.strip())
        return parts

    def _analyze_chunk(
        self, paper: PaperItem, section_name: str, chunk_text: str, focus: Optional[str],
    ) -> str:
        focus_line = f"\nUser focus: {focus}" if focus else ""
        prompt = f"""You are reading the **{section_name}** section of "{paper.title}".
Extract ALL key information following these quality rules:{focus_line}

Quality rules:
- Extract EVERY formula (with variable definitions)
- Extract EVERY table (preserve all rows/columns)
- Note EVERY figure reference with its interpretation
- Use specific numbers, not vague claims ("achieves 82.3 F1" not "achieves good results")
- Wrap key technical terms in [[Concept]] on first appearance
- If a formula appears, format as: name, LaTeX block, one-line meaning, symbol legend

Section content:
{chunk_text}

Output: structured extraction in Markdown (200-500 words). Preserve all quantitative details.""".strip()

        return self.llm.invoke(
            [{"role": "user", "content": prompt}], temperature=0.2
        ).strip()

    def _synthesize(
        self, paper: PaperItem, chunk_summaries: List[Dict], focus: Optional[str],
    ) -> Dict:
        combined = "\n\n".join(
            f"## {c['section']}\n{c['summary']}" for c in chunk_summaries
        )
        focus_line = f"\nUser focus: {focus}" if focus else ""
        prompt = f"""Based on the section-by-section analysis of "{paper.title}", produce a final structured analysis.{focus_line}

Section analyses:
{combined}

Return JSON with this exact schema:
{{
  "tldr": "Single sentence, ≤50 words, capturing the core contribution",
  "problem": "What concrete problem does the paper solve?",
  "prior_limitations": "What was broken/missing in prior work?",
  "contributions": ["Contribution 1 headline — what + why it matters", "..."],
  "method_summary": "Architecture/approach overview (100-200 words, use [[Concept]] for key terms)",
  "formulas": [
    {{"name": "formula name", "latex": "LaTeX string", "meaning": "one-line", "symbols": {{"x": "meaning"}}}}
  ],
  "datasets": [{{"name": "...", "size": "...", "used_for": "train/eval"}}],
  "results": "Main results with specific numbers (reference tables)",
  "ablations": "Key ablation findings",
  "strengths": ["Specific strength citing a number or design choice"],
  "weaknesses": ["Specific weakness — missing experiment, unvalidated assumption, etc."],
  "reproducibility": {{"code": true/false, "weights": true/false, "details_sufficient": true/false, "data_public": true/false}},
  "related_work": ["[[Prior Work 1]] — relationship", "[[Prior Work 2]] — relationship"],
  "future_work": "What the authors propose next",
  "tags": ["tag1", "tag2", "tag3"],
  "relevance_score": 0.0
}}

Rules:
- Based on section analyses only, do not hallucinate
- Numbers and dataset names must be accurate
- Strengths/weaknesses must cite specific evidence, not vibes
- formulas array: include ALL formulas found in the section analyses""".strip()

        response = self.llm.invoke(
            [{"role": "user", "content": prompt}], temperature=0.2
        )
        result = extract_json_object(response)
        if not result:
            return {
                "tldr": "", "problem": "", "contributions": [],
                "method_summary": response, "formulas": [],
                "datasets": [], "results": "", "strengths": [],
                "weaknesses": [], "tags": [], "relevance_score": 0.0,
            }
        return result

    # ------------------------------------------------------------------ #
    # 降级：仅用 abstract
    # ------------------------------------------------------------------ #

    def _analyze_abstract(self, paper: PaperItem, focus: Optional[str]) -> Dict:
        focus_line = f"\nFocus: {focus}" if focus else ""
        prompt = f"""Analyze this paper based on its abstract only (full text unavailable).{focus_line}

Title: {paper.title}
Authors: {', '.join(paper.authors[:5])}
Published: {paper.published}
Abstract: {paper.abstract}

Return JSON:
{{
  "tldr": "Single sentence core contribution",
  "problem": "Core problem solved",
  "contributions": ["contribution 1", "contribution 2"],
  "method_summary": "Method overview based on abstract",
  "formulas": [],
  "datasets": [],
  "results": "Results mentioned in abstract (if any)",
  "strengths": ["Based on abstract"],
  "weaknesses": ["Cannot assess without full text"],
  "tags": ["tag1", "tag2"],
  "relevance_score": 0.0
}}""".strip()

        response = self.llm.invoke(
            [{"role": "user", "content": prompt}], temperature=0.2
        )
        result = extract_json_object(response)
        if not result:
            result = {"tldr": response, "contributions": [], "tags": []}
        result["_source"] = "abstract_only"
        return result

    # ------------------------------------------------------------------ #
    # 笔记保存（对齐 paper-reader skill 的 note 格式）
    # ------------------------------------------------------------------ #

    def _save_note(self, paper: PaperItem, analysis: Dict) -> Optional[Path]:
        """保存结构化 Markdown 笔记到 workspace/paper_notes/"""
        method_name = self._extract_method_name(paper, analysis)
        filename = f"{method_name}.md"
        path = self.notes_dir / filename

        now = datetime.utcnow().strftime("%Y-%m-%d")
        tags = analysis.get("tags", [])

        # YAML frontmatter
        frontmatter = (
            f"---\n"
            f"title: \"{paper.title}\"\n"
            f"method_name: \"{method_name}\"\n"
            f"arxiv_id: \"{paper.paper_id}\"\n"
            f"authors: [{', '.join(paper.authors[:5])}]\n"
            f"year: {paper.published[:4] if paper.published else 'unknown'}\n"
            f"tags: [{', '.join(tags)}]\n"
            f"read_mode: {'full' if analysis.get('_source') == 'fulltext' else 'abstract'}\n"
            f"created: {now}\n"
            f"---\n\n"
        )

        # Body
        body_parts = [f"# {paper.title}\n"]

        if analysis.get("tldr"):
            body_parts.append(f"## TL;DR\n\n> {analysis['tldr']}\n")

        if analysis.get("contributions"):
            body_parts.append("## Core Contributions\n")
            for i, c in enumerate(analysis["contributions"], 1):
                body_parts.append(f"{i}. **{c}**")
            body_parts.append("")

        if analysis.get("problem"):
            body_parts.append(f"## Problem & Motivation\n\n{analysis['problem']}\n")
            if analysis.get("prior_limitations"):
                body_parts.append(f"**Prior limitations**: {analysis['prior_limitations']}\n")

        if analysis.get("method_summary"):
            body_parts.append(f"## Core Method\n\n{analysis['method_summary']}\n")

        if analysis.get("formulas"):
            body_parts.append("## Formulas\n")
            for f in analysis["formulas"]:
                body_parts.append(f"### {f.get('name', 'Formula')}\n")
                body_parts.append(f"$$\n{f.get('latex', '')}\n$$\n")
                body_parts.append(f"**Meaning**: {f.get('meaning', '')}\n")
                if f.get("symbols"):
                    body_parts.append("**Symbols**:")
                    for sym, meaning in f["symbols"].items():
                        body_parts.append(f"- ${sym}$: {meaning}")
                    body_parts.append("")

        if analysis.get("datasets"):
            body_parts.append("## Datasets\n")
            body_parts.append("| Dataset | Size | Used for |")
            body_parts.append("|---------|------|----------|")
            for d in analysis["datasets"]:
                body_parts.append(f"| {d.get('name','')} | {d.get('size','')} | {d.get('used_for','')} |")
            body_parts.append("")

        if analysis.get("results"):
            body_parts.append(f"## Results\n\n{analysis['results']}\n")

        if analysis.get("ablations"):
            body_parts.append(f"## Ablations\n\n{analysis['ablations']}\n")

        if analysis.get("strengths") or analysis.get("weaknesses"):
            body_parts.append("## Critical View\n")
            if analysis.get("strengths"):
                body_parts.append("### Strengths")
                for s in analysis["strengths"]:
                    body_parts.append(f"- {s}")
                body_parts.append("")
            if analysis.get("weaknesses"):
                body_parts.append("### Weaknesses")
                for w in analysis["weaknesses"]:
                    body_parts.append(f"- {w}")
                body_parts.append("")

        if analysis.get("reproducibility"):
            r = analysis["reproducibility"]
            body_parts.append("### Reproducibility\n")
            body_parts.append(f"- [{'x' if r.get('code') else ' '}] Code released")
            body_parts.append(f"- [{'x' if r.get('weights') else ' '}] Weights released")
            body_parts.append(f"- [{'x' if r.get('details_sufficient') else ' '}] Training details sufficient")
            body_parts.append(f"- [{'x' if r.get('data_public') else ' '}] Data publicly available")
            body_parts.append("")

        if analysis.get("related_work"):
            body_parts.append("## Related Work\n")
            for rw in analysis["related_work"]:
                body_parts.append(f"- {rw}")
            body_parts.append("")

        content = frontmatter + "\n".join(body_parts)
        path.write_text(content, encoding="utf-8")

        analysis["_note_path"] = str(path)
        analysis["_method_name"] = method_name
        return path

    def _extract_method_name(self, paper: PaperItem, analysis: Dict) -> str:
        """从论文标题提取方法名作为文件名。"""
        title = paper.title or ""
        if ":" in title:
            candidate = title.split(":")[0].strip()
            if len(candidate) <= 30:
                return self._sanitize_filename(candidate)
        if analysis.get("contributions"):
            first = analysis["contributions"][0]
            if "**" in first:
                parts = first.split("**")
                if len(parts) >= 2 and len(parts[1]) <= 30:
                    return self._sanitize_filename(parts[1])
        return paper.paper_id.replace("/", "_")

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """清理文件名：去特殊字符，希腊字母转 ASCII。"""
        replacements = {"π": "Pi", "σ": "Sigma", "α": "Alpha", "β": "Beta", "γ": "Gamma"}
        for greek, ascii_name in replacements.items():
            name = name.replace(greek, ascii_name)
        safe = "".join(c if c.isalnum() or c in "-_ " else "" for c in name)
        return safe.strip().replace(" ", "_") or "unnamed"

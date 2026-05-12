"""
PaperAnalyzer - 深度论文分析 Agent

分析策略：
  1. 优先尝试下载全文 PDF 并按章节提取
  2. 按章节分块发给 LLM 逐段分析（map 阶段）
  3. 汇总各段分析结果做综合（reduce 阶段）
  4. 如果全文获取失败，降级到摘要分析（保持原有行为）
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..core.llm import LLMClient
from ..core.config import Settings
from ..core.models import PaperItem
from ..core.utils import extract_json_object
from ..services.paper_fetcher import PaperFetcher, PaperFullText


# 章节切分的优先级（按这个顺序尝试分块）
SECTION_ORDER = [
    "abstract", "introduction", "related_work",
    "method", "experiments", "discussion", "conclusion",
]

# 单个 chunk 的字符上限（对应大约 3-5k tokens，留出 LM 输出空间）
CHUNK_CHAR_LIMIT = 6000


class PaperAnalyzer:
    def __init__(self, llm_client: LLMClient, settings: Settings):
        self.llm = llm_client
        self.settings = settings
        self.fetcher = PaperFetcher(
            cache_dir=settings.workspace_dir / "pdf_cache"
        )

    # ------------------------------------------------------------------ #
    # 公开接口
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        paper: PaperItem,
        focus: Optional[str] = None,
        use_fulltext: bool = True,
    ) -> Dict:
        """
        深度分析论文。

        Args:
            paper: 论文元数据
            focus: 可选关注点
            use_fulltext: 是否尝试全文分析（失败会自动降级到摘要）
        """
        if use_fulltext:
            fulltext = self.fetcher.fetch_fulltext(paper.paper_id)
            if fulltext and fulltext.num_chars > 500:
                return self._analyze_fulltext(paper, fulltext, focus)

        # 降级：只用 abstract
        return self._analyze_abstract(paper, focus)

    # ------------------------------------------------------------------ #
    # 全文分析（两段式 map-reduce）
    # ------------------------------------------------------------------ #

    def _analyze_fulltext(
        self,
        paper: PaperItem,
        fulltext: PaperFullText,
        focus: Optional[str],
    ) -> Dict:
        # Map 阶段：按章节分析
        chunks = self._build_chunks(fulltext)
        chunk_summaries: List[Dict] = []
        for chunk_name, chunk_text in chunks:
            summary = self._analyze_chunk(paper, chunk_name, chunk_text, focus)
            chunk_summaries.append({"section": chunk_name, "summary": summary})

        # Reduce 阶段：综合所有 chunk 结果
        final = self._synthesize(paper, chunk_summaries, focus)
        final["_source"] = "fulltext"
        final["_num_pages"] = fulltext.num_pages
        final["_num_chunks"] = len(chunk_summaries)
        final["_section_summaries"] = chunk_summaries
        return final

    def _build_chunks(self, ft: PaperFullText) -> List[tuple]:
        """
        构建 (section_name, text) 的 chunk 列表。
        优先按识别到的章节切，章节太长再按字符切；完全识别不到就按字符切 raw_text。
        """
        chunks: List[tuple] = []

        if ft.sections:
            # 按 SECTION_ORDER 遍历已识别章节
            for key in SECTION_ORDER:
                text = ft.sections.get(key)
                if not text:
                    continue
                if len(text) <= CHUNK_CHAR_LIMIT:
                    chunks.append((key, text))
                else:
                    # 长章节按字符切小块
                    for i, sub in enumerate(self._split_by_chars(text, CHUNK_CHAR_LIMIT)):
                        chunks.append((f"{key}_part{i+1}", sub))
            # references 一般不分析，跳过

        if not chunks:
            # 完全没识别到章节：用 raw_text 按字符硬切
            for i, sub in enumerate(self._split_by_chars(ft.raw_text, CHUNK_CHAR_LIMIT)):
                chunks.append((f"chunk_{i+1}", sub))

        return chunks

    @staticmethod
    def _split_by_chars(text: str, limit: int) -> List[str]:
        """按字符上限切分，尽量在段落边界切。"""
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
        self,
        paper: PaperItem,
        section_name: str,
        chunk_text: str,
        focus: Optional[str],
    ) -> str:
        """对单个章节/分块做摘要式分析，返回纯文本。"""
        focus_line = f"\n用户关注点: {focus}" if focus else ""
        prompt = f"""你正在阅读论文 "{paper.title}" 的 **{section_name}** 部分。
请提取本段的核心信息，保留关键细节（算法、数据集、实验配置、数值结果等）。{focus_line}

章节内容:
{chunk_text}

要求:
- 用中文输出 200-400 字的摘要
- 保留重要公式/指标/数据集名，不要虚构
- 不做概括性评价，只陈述事实
""".strip()

        return self.llm.invoke(
            [{"role": "user", "content": prompt}], temperature=0.2
        ).strip()

    def _synthesize(
        self,
        paper: PaperItem,
        chunk_summaries: List[Dict],
        focus: Optional[str],
    ) -> Dict:
        """把分段摘要汇总成结构化分析结果。"""
        combined = "\n\n".join(
            f"## {c['section']}\n{c['summary']}" for c in chunk_summaries
        )
        focus_line = f"\n用户关注点: {focus}" if focus else ""
        prompt = f"""基于以下对论文 "{paper.title}" 各章节的分析摘要，综合输出结构化分析结果。{focus_line}

章节摘要:
{combined}

返回 JSON:
{{
  "summary": "全文核心内容概述（300字以内，覆盖问题-方法-结果）",
  "problem": "解决的核心问题（具体，不泛泛）",
  "contributions": ["贡献1（具体）", "贡献2", "贡献3"],
  "methods": ["方法细节1", "方法细节2"],
  "datasets": ["使用的数据集1", "数据集2"],
  "results": "主要实验结果（保留关键指标数字）",
  "limitations": ["局限1", "局限2"],
  "future_work": "作者提出的未来方向",
  "relevance_score": 0.0
}}

规则：
- 基于章节摘要的事实，不要虚构
- 数值、数据集、baseline 名称要准确
""".strip()

        response = self.llm.invoke(
            [{"role": "user", "content": prompt}], temperature=0.2
        )
        result = extract_json_object(response)
        if not result:
            return {
                "summary": response, "problem": "",
                "contributions": [], "methods": [],
                "datasets": [], "results": "",
                "limitations": [], "future_work": "",
                "relevance_score": 0.0,
            }
        return result

    # ------------------------------------------------------------------ #
    # 降级：仅用 abstract
    # ------------------------------------------------------------------ #

    def _analyze_abstract(self, paper: PaperItem, focus: Optional[str]) -> Dict:
        prompt = self._build_abstract_prompt(paper, focus)
        response = self.llm.invoke(
            [{"role": "user", "content": prompt}], temperature=0.2
        )
        result = extract_json_object(response)
        if not result:
            result = {
                "summary": response, "contributions": [],
                "methods": [], "limitations": [],
            }
        result["_source"] = "abstract_only"
        return result

    def _build_abstract_prompt(self, paper: PaperItem, focus: Optional[str]) -> str:
        focus_line = f"\n特别关注: {focus}" if focus else ""
        return f"""请基于论文摘要做初步分析（全文暂时无法获取，分析仅基于abstract）:{focus_line}

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
  "results": "主要实验结果（如摘要中提及）",
  "limitations": ["局限1（基于摘要推测）"],
  "future_work": "未来工作方向",
  "relevance_score": 0.0
}}""".strip()

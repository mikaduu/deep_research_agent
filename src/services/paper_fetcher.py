"""
论文全文获取服务 - 从 arXiv 下载 PDF 并提取结构化全文

核心能力：
- 按 arXiv ID 下载 PDF（带本地缓存，不重复下载）
- 用 PyMuPDF 提取文本，按页和章节切分
- 识别常见章节（Abstract / Introduction / Method / Experiments / Conclusion / References）
- 返回结构化全文给下游 Agent 分析
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests


@dataclass
class PaperFullText:
    """论文全文的结构化表示。"""
    arxiv_id: str
    pdf_path: str
    raw_text: str                            # 全文原始文本
    num_pages: int = 0
    sections: Dict[str, str] = field(default_factory=dict)  # section_name -> text
    num_chars: int = 0


class PaperFetcher:
    """
    arXiv PDF 下载与全文提取。

    使用方式:
        fetcher = PaperFetcher(cache_dir=Path("workspace/pdf_cache"))
        fulltext = fetcher.fetch_fulltext("2301.00234")
        print(fulltext.sections.get("introduction"))
    """

    # arXiv 可接受的标准 PDF URL 格式
    PDF_URL_TEMPLATES = [
        "https://arxiv.org/pdf/{id}.pdf",
        "https://arxiv.org/pdf/{id}",
    ]

    # 章节识别（按常见学术论文结构）
    SECTION_PATTERNS = [
        (r"\babstract\b", "abstract"),
        (r"\bintroduction\b|\b1\s*introduction\b", "introduction"),
        (r"\brelated\s+work\b|\bbackground\b", "related_work"),
        (r"\bmethod(s|ology)?\b|\bapproach\b|\bproposed\b", "method"),
        (r"\bexperiment(s|al)?\b|\bevaluation\b|\bresults?\b", "experiments"),
        (r"\bdiscussion\b|\banalysis\b", "discussion"),
        (r"\bconclusion(s)?\b|\bfuture\s+work\b", "conclusion"),
        (r"\breferences?\b|\bbibliography\b", "references"),
    ]

    def __init__(self, cache_dir: Path, timeout: int = 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # 公开接口
    # ------------------------------------------------------------------ #

    def fetch_fulltext(self, arxiv_id: str) -> Optional[PaperFullText]:
        """下载 PDF 并提取全文，失败返回 None。"""
        arxiv_id = self._normalize_id(arxiv_id)
        if not arxiv_id:
            return None

        pdf_path = self._download_pdf(arxiv_id)
        if pdf_path is None:
            return None

        try:
            raw_text, num_pages = self._extract_text(pdf_path)
        except Exception as e:
            print(f"[PaperFetcher] text extraction failed: {e}")
            return None

        sections = self._split_sections(raw_text)

        return PaperFullText(
            arxiv_id=arxiv_id,
            pdf_path=str(pdf_path),
            raw_text=raw_text,
            num_pages=num_pages,
            sections=sections,
            num_chars=len(raw_text),
        )

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_id(arxiv_id: str) -> str:
        """从 URL / 各种输入中提取 arXiv ID。"""
        arxiv_id = (arxiv_id or "").strip()
        if not arxiv_id:
            return ""
        # 允许传入完整 URL 或带版本号 (如 2301.00234v2)
        if "/" in arxiv_id:
            arxiv_id = arxiv_id.rstrip("/").split("/")[-1]
        if arxiv_id.endswith(".pdf"):
            arxiv_id = arxiv_id[:-4]
        return arxiv_id

    def _download_pdf(self, arxiv_id: str) -> Optional[Path]:
        """下载 PDF（有缓存），返回本地路径。"""
        cache_file = self.cache_dir / f"{arxiv_id}.pdf"
        if cache_file.exists() and cache_file.stat().st_size > 0:
            return cache_file

        last_err = None
        for tpl in self.PDF_URL_TEMPLATES:
            url = tpl.format(id=arxiv_id)
            for attempt in range(2):
                try:
                    resp = requests.get(
                        url,
                        timeout=self.timeout,
                        headers={"User-Agent": "deep-research-agent/1.0"},
                        stream=True,
                    )
                    if resp.status_code == 200:
                        cache_file.write_bytes(resp.content)
                        return cache_file
                    if resp.status_code == 429:
                        time.sleep(2 ** attempt)
                        continue
                    last_err = f"HTTP {resp.status_code}"
                    break
                except Exception as e:
                    last_err = str(e)[:120]
                    time.sleep(1)
        print(f"[PaperFetcher] download failed for {arxiv_id}: {last_err}")
        return None

    def _extract_text(self, pdf_path: Path) -> tuple:
        """用 PyMuPDF 提取全文。"""
        import fitz  # PyMuPDF

        text_parts: List[str] = []
        with fitz.open(str(pdf_path)) as doc:
            num_pages = doc.page_count
            for page in doc:
                text_parts.append(page.get_text("text"))

        raw_text = "\n".join(text_parts)
        # 合并被 PDF 换行打断的词
        raw_text = re.sub(r"-\n(\w)", r"\1", raw_text)
        return raw_text, num_pages

    def _split_sections(self, text: str) -> Dict[str, str]:
        """
        按常见章节标题粗分 section。
        识别不到就返回空 dict（调用方可用 raw_text 兜底）。
        """
        lines = text.split("\n")
        sections: Dict[str, str] = {}
        current_key: Optional[str] = None
        current_buf: List[str] = []

        for line in lines:
            clean = line.strip().lower()
            matched_key = None
            # 章节标题往往是独立一行的短文本
            if 2 <= len(clean) <= 60:
                for pat, key in self.SECTION_PATTERNS:
                    if re.search(pat, clean):
                        matched_key = key
                        break

            if matched_key:
                if current_key and current_buf:
                    sections[current_key] = "\n".join(current_buf).strip()
                current_key = matched_key
                current_buf = []
            else:
                if current_key:
                    current_buf.append(line)

        if current_key and current_buf:
            sections[current_key] = "\n".join(current_buf).strip()

        # 合并同一章节的多次出现（比如 method 被分了两次）
        return sections

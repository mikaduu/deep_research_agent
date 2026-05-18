"""
通用网页搜索服务 — DuckDuckGo 免费搜索

提供两个能力：
- search: 关键词搜索，返回标题+URL+摘要
- fetch:  抓取单个 URL 的正文文本（简单 HTML→text）
"""

import time
from dataclasses import dataclass
from typing import List, Optional

import requests


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str


class WebSearcher:
    """基于 DuckDuckGo 的免费网页搜索。"""

    def search(self, query: str, max_results: int = 5) -> List[WebResult]:
        """搜索网页，返回标题+URL+摘要。"""
        if not query.strip():
            return []
        try:
            # 兼容新旧包名（duckduckgo_search 已改名为 ddgs）
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return [
                WebResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                )
                for r in results
            ]
        except Exception as e:
            print(f"[WebSearcher] search failed: {type(e).__name__}: {str(e)[:100]}")
            return []

    def fetch(self, url: str, max_chars: int = 5000) -> Optional[str]:
        """抓取 URL 正文（简单提取，不做复杂解析）。"""
        try:
            resp = requests.get(
                url, timeout=10,
                headers={"User-Agent": "deep-research-agent/1.0"},
            )
            if resp.status_code != 200:
                return None
            # 简单去 HTML 标签
            text = self._html_to_text(resp.text)
            return text[:max_chars] if text else None
        except Exception:
            return None

    @staticmethod
    def _html_to_text(html: str) -> str:
        """极简 HTML→text：去标签，保留文本。"""
        import re
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

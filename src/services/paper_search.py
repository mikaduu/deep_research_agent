import time
import arxiv
import requests
from typing import List, Optional

from ..core.models import PaperItem


class ArxivSearcher:
    """
    arXiv搜索器，含有限制和容错：
    - query 超长自动截断到 200 字符
    - HTTP 429/其他异常返回空列表而非抛出
    - num_retries 由 arxiv.Client 内部处理，这里再加一层保护
    """
    MAX_QUERY_LEN = 200

    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        # page_size=小一些，num_retries 多一点，delay 适当拉长避免 429
        self.client = arxiv.Client(page_size=10, delay_seconds=3.0, num_retries=3)

    def search(self, query: str, max_results: Optional[int] = None) -> List[PaperItem]:
        limit = max_results or self.max_results
        safe_query = (query or "")[: self.MAX_QUERY_LEN].strip()
        if not safe_query:
            return []

        try:
            search = arxiv.Search(
                query=safe_query,
                max_results=limit,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            papers = []
            for result in self.client.results(search):
                papers.append(PaperItem(
                    paper_id=result.entry_id.split('/')[-1],
                    title=result.title,
                    authors=[a.name for a in result.authors],
                    abstract=result.summary,
                    url=result.entry_id,
                    published=result.published.isoformat() if result.published else "",
                    updated=result.updated.isoformat() if result.updated else "",
                    categories=result.categories,
                ))
            return papers
        except Exception as e:
            # arXiv 429/网络问题/解析错误 → 降级为空列表，不影响整体流程
            print(f"[ArxivSearcher] search failed ({type(e).__name__}): {str(e)[:120]}")
            return []


class SemanticScholarSearcher:
    """
    Semantic Scholar 搜索器，含容错和重试。
    """
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    MAX_QUERY_LEN = 300

    def search(self, query: str, max_results: int = 10) -> List[PaperItem]:
        safe_query = (query or "")[: self.MAX_QUERY_LEN].strip()
        if not safe_query:
            return []

        for attempt in range(2):
            try:
                resp = requests.get(
                    f"{self.BASE_URL}/paper/search",
                    params={
                        "query": safe_query,
                        "limit": max_results,
                        "fields": "title,authors,abstract,year,externalIds,url",
                    },
                    timeout=15,
                )
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code != 200:
                    return []
                papers = []
                for item in resp.json().get("data", []):
                    papers.append(PaperItem(
                        paper_id=item.get("paperId", ""),
                        title=item.get("title", ""),
                        authors=[a.get("name", "") for a in item.get("authors", [])],
                        abstract=item.get("abstract") or "",
                        url=item.get("url")
                             or f"https://www.semanticscholar.org/paper/{item.get('paperId','')}",
                        published=str(item.get("year", "")),
                    ))
                return papers
            except Exception as e:
                print(f"[S2Searcher] search failed ({type(e).__name__}): {str(e)[:120]}")
                return []
        return []

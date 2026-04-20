import arxiv
import requests
from typing import List, Optional

from ..models import PaperItem


class ArxivSearcher:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self.client = arxiv.Client()

    def search(self, query: str, max_results: Optional[int] = None) -> List[PaperItem]:
        limit = max_results or self.max_results
        search = arxiv.Search(query=query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance)
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


class SemanticScholarSearcher:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def search(self, query: str, max_results: int = 10) -> List[PaperItem]:
        resp = requests.get(
            f"{self.BASE_URL}/paper/search",
            params={"query": query, "limit": max_results, "fields": "title,authors,abstract,year,externalIds,url"},
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        papers = []
        for item in resp.json().get("data", []):
            papers.append(PaperItem(
                paper_id=item.get("paperId", ""),
                title=item.get("title", ""),
                authors=[a.get("name", "") for a in item.get("authors", [])],
                abstract=item.get("abstract") or "",
                url=item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId','')}",
                published=str(item.get("year", "")),
            ))
        return papers

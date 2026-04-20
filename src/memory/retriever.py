import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from ..models import MemoryHit


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


@dataclass
class _Doc:
    doc_id: str
    content: str
    metadata: Dict[str, str]
    tf: Counter


class RagMemory:
    def __init__(self) -> None:
        self._docs: List[_Doc] = []

    def add(self, doc_id: str, content: str, metadata: Dict[str, str]) -> None:
        tf = Counter(self._tokenize(content))
        self._docs.append(_Doc(doc_id=doc_id, content=content, metadata=metadata, tf=tf))

    def retrieve(self, query: str, top_k: int) -> List[MemoryHit]:
        if not self._docs:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        df = Counter()
        for token in set(q_tokens):
            df[token] = sum(1 for doc in self._docs if token in doc.tf)

        scored: List[MemoryHit] = []
        doc_count = len(self._docs)
        for doc in self._docs:
            score = 0.0
            for token in q_tokens:
                if token not in doc.tf:
                    continue
                idf = math.log((doc_count + 1) / (df[token] + 1)) + 1.0
                score += doc.tf[token] * idf

            if score > 0:
                scored.append(
                    MemoryHit(
                        doc_id=doc.doc_id,
                        score=score,
                        content=doc.content,
                        metadata=doc.metadata,
                    )
                )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [x.lower() for x in TOKEN_PATTERN.findall(text)]


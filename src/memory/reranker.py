"""
Cross-Encoder Reranker - 跨层统一重排

工作流程：
  1. 接收来自多路召回（Episodic/Skill/Vector）的候选列表
  2. 归一化为统一的 RerankCandidate 结构
  3. 去重（按 doc_id）
  4. 用 Cross-Encoder 对每个 (query, content) 对精细打分
  5. 按新分数排序返回 top-k

Cross-Encoder 与 Bi-Encoder（Chroma 用的）的区别：
  - Bi-Encoder：query 和 doc 分别编码，点积相似度，速度快但精度有限
  - Cross-Encoder：query 和 doc 拼接后一起编码，直接输出相关性分数，精度高但慢

因此 Cross-Encoder 最适合做"精排"：在粗召回的少量候选上用。

默认模型 BAAI/bge-reranker-base：
  - 支持中英双语
  - 输入长度 512 tokens
  - 首次使用会自动下载（约 280MB）
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math

from ..core.models import MemoryHit


@dataclass
class RerankCandidate:
    """统一的重排候选，包装来自不同来源的记忆条目。"""
    doc_id: str
    content: str
    source: str                # "episode" | "skill" | "vector"
    raw: Any                   # 原始对象（Episode / Skill / MemoryHit）
    rerank_score: float = 0.0  # Cross-Encoder 打分
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrossEncoderReranker:
    """
    基于 sentence-transformers CrossEncoder 的精排器。

    懒加载：首次调用 rerank() 时才载入模型，避免启动慢。

    分数说明：
      - bge-reranker 输出原始 logit（约 -10 ~ +10），不是概率
      - 经过 sigmoid 后的值约等于"相关性概率"，取值 (0, 1)
      - 默认 score_threshold 基于 sigmoid 后的值（0.3 是经验阈值）
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-base"
    # 默认分数阈值（作用于 sigmoid 归一化后的分数）
    # 0.3 含义：相关性概率 < 30% 的候选会被丢弃
    DEFAULT_SCORE_THRESHOLD = 0.3

    def __init__(
        self,
        model_name: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.score_threshold = (
            score_threshold if score_threshold is not None else self.DEFAULT_SCORE_THRESHOLD
        )
        self._model = None  # 懒加载

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    @staticmethod
    def _sigmoid(x: float) -> float:
        """数值稳定的 sigmoid，避免 exp overflow。"""
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[RerankCandidate]:
        """
        对候选列表重排，返回 top_k。

        Args:
            query: 查询文本
            candidates: 待重排的候选列表（来自多路召回）
            top_k: 返回数量上限
            score_threshold: 可选的单次覆盖阈值；传 None 则使用实例默认值；
                             传 0 或负值等效禁用阈值过滤

        Returns:
            按 rerank_score 降序排列、且分数 >= 阈值的候选（最多 top_k 条）
        """
        if not candidates:
            return []

        threshold = (
            self.score_threshold if score_threshold is None else score_threshold
        )

        # 去重：同一 doc_id 只保留第一次出现
        seen = set()
        unique: List[RerankCandidate] = []
        for c in candidates:
            if c.doc_id in seen:
                continue
            seen.add(c.doc_id)
            unique.append(c)

        # 构建 (query, doc) pairs
        pairs = [[query, c.content] for c in unique]

        # Cross-Encoder 打分
        model = self._get_model()
        raw_scores = model.predict(pairs)

        # 写回归一化后的分数
        for c, s in zip(unique, raw_scores):
            c.rerank_score = self._sigmoid(float(s))

        # 按分数降序排序
        unique.sort(key=lambda x: x.rerank_score, reverse=True)

        # 阈值过滤（<= 0 表示禁用过滤）
        if threshold > 0:
            filtered = [c for c in unique if c.rerank_score >= threshold]
        else:
            filtered = unique

        return filtered[:top_k]


def build_candidates_from_memory(
    episodes: List[Any],
    skills: List[Any],
    vectors: List[MemoryHit],
) -> List[RerankCandidate]:
    """
    把三路召回结果统一包装为 RerankCandidate。

    - Episode: 用 topic + insights 作为 rerank 内容（更精炼）
    - Skill:   用 name + content 作为 rerank 内容
    - Vector:  直接用 content
    """
    candidates: List[RerankCandidate] = []

    for ep in episodes:
        candidates.append(RerankCandidate(
            doc_id=f"ep:{ep.id}",
            content=f"{ep.topic}\n{ep.insights}",
            source="episode",
            raw=ep,
            metadata={"quality_score": ep.quality_score},
        ))

    for sk in skills:
        candidates.append(RerankCandidate(
            doc_id=f"sk:{sk.id}",
            content=f"{sk.name}\n{sk.content}",
            source="skill",
            raw=sk,
            metadata={"domain": sk.domain, "usage_count": sk.usage_count},
        ))

    for hit in vectors:
        candidates.append(RerankCandidate(
            doc_id=hit.doc_id,
            content=hit.content,
            source="vector",
            raw=hit,
            metadata=hit.metadata,
        ))

    return candidates

"""
技能记忆（Skill Memory）- Hermes风格第三层记忆

基于 SQLite + FTS5 存储从研究会话中学到的可复用策略：
- 研究方法、搜索策略、领域知识模式
- 使用次数和成功率追踪，支持技能演化
- FTS5全文检索，按相关性匹配技能
"""

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class Skill:
    id: str
    name: str
    description: str
    trigger_conditions: str
    content: str
    domain: str
    usage_count: int
    success_rate: float
    created_at: str
    updated_at: str


class SkillMemory:
    """SQLite + FTS5 技能记忆，存储可复用的研究策略和模式。"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id                 TEXT PRIMARY KEY,
                    name               TEXT NOT NULL,
                    description        TEXT NOT NULL,
                    trigger_conditions TEXT DEFAULT '',
                    content            TEXT NOT NULL,
                    domain             TEXT DEFAULT 'general',
                    usage_count        INTEGER DEFAULT 0,
                    success_rate       REAL DEFAULT 0.5,
                    created_at         TEXT NOT NULL,
                    updated_at         TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts
                USING fts5(
                    id UNINDEXED, name, description,
                    trigger_conditions, content, domain,
                    content='skills', content_rowid='rowid'
                )
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS skills_ai
                AFTER INSERT ON skills BEGIN
                    INSERT INTO skills_fts(
                        rowid, id, name, description,
                        trigger_conditions, content, domain
                    )
                    VALUES (
                        new.rowid, new.id, new.name, new.description,
                        new.trigger_conditions, new.content, new.domain
                    );
                END
            """)

    def add_skill(
        self,
        name: str,
        description: str,
        trigger_conditions: str,
        content: str,
        domain: str = "general",
    ) -> str:
        skill_id = uuid.uuid4().hex[:8]
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO skills VALUES (?,?,?,?,?,?,?,?,?,?)",
                (skill_id, name, description, trigger_conditions,
                 content, domain, 0, 0.5, now, now),
            )
        return skill_id

    def find_relevant(
        self, query: str, domain: Optional[str] = None, limit: int = 3
    ) -> List[Skill]:
        """FTS5检索最相关的技能。"""
        try:
            with self._conn() as conn:
                if domain:
                    rows = conn.execute(
                        """
                        SELECT s.id, s.name, s.description, s.trigger_conditions,
                               s.content, s.domain, s.usage_count, s.success_rate,
                               s.created_at, s.updated_at
                        FROM skills_fts f
                        JOIN skills s ON s.rowid = f.rowid
                        WHERE skills_fts MATCH ? AND s.domain = ?
                        ORDER BY rank LIMIT ?
                        """,
                        (query, domain, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT s.id, s.name, s.description, s.trigger_conditions,
                               s.content, s.domain, s.usage_count, s.success_rate,
                               s.created_at, s.updated_at
                        FROM skills_fts f
                        JOIN skills s ON s.rowid = f.rowid
                        WHERE skills_fts MATCH ?
                        ORDER BY rank LIMIT ?
                        """,
                        (query, limit),
                    ).fetchall()
        except sqlite3.OperationalError:
            rows = self._fallback_search(query, domain, limit)
        return [self._to_skill(r) for r in rows]

    def _fallback_search(self, query: str, domain: Optional[str], limit: int) -> list:
        like = f"%{query}%"
        with self._conn() as conn:
            if domain:
                return conn.execute(
                    "SELECT id, name, description, trigger_conditions, content, domain, "
                    "usage_count, success_rate, created_at, updated_at FROM skills "
                    "WHERE (name LIKE ? OR content LIKE ?) AND domain=? "
                    "ORDER BY usage_count DESC LIMIT ?",
                    (like, like, domain, limit),
                ).fetchall()
            return conn.execute(
                "SELECT id, name, description, trigger_conditions, content, domain, "
                "usage_count, success_rate, created_at, updated_at FROM skills "
                "WHERE name LIKE ? OR content LIKE ? "
                "ORDER BY usage_count DESC LIMIT ?",
                (like, like, limit),
            ).fetchall()

    def update_usage(self, skill_id: str, success: bool):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT usage_count, success_rate FROM skills WHERE id=?", (skill_id,)
            ).fetchone()
            if row:
                count = row[0] + 1
                rate = (row[1] * row[0] + (1.0 if success else 0.0)) / count
                conn.execute(
                    "UPDATE skills SET usage_count=?, success_rate=?, updated_at=? WHERE id=?",
                    (count, round(rate, 4), datetime.utcnow().isoformat(), skill_id),
                )

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]

    def get_by_id(self, skill_id: str) -> Optional[Skill]:
        """按 ID 取完整技能，供向量命中后回查使用。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, name, description, trigger_conditions, content, domain, "
                "usage_count, success_rate, created_at, updated_at FROM skills WHERE id = ?",
                (skill_id,),
            ).fetchone()
        return self._to_skill(row) if row else None

    def _to_skill(self, row) -> Skill:
        return Skill(
            id=row[0], name=row[1], description=row[2],
            trigger_conditions=row[3], content=row[4], domain=row[5],
            usage_count=row[6], success_rate=row[7],
            created_at=row[8], updated_at=row[9],
        )

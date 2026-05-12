"""
反思引擎（Reflection Engine）- Hermes风格自我学习循环

每次研究会话结束后自动执行：
1. 计算研究质量评分
2. 通过LLM提炼关键洞见和标签 → 存入情节记忆（Layer 2）
3. 通过LLM提取可复用研究技能 → 存入技能记忆（Layer 3）

下次研究时，MemoryManager 会自动检索这些积累的知识，
让 Agent 随着使用次数增加而越来越聪明。
"""

from typing import Any, Dict, List

from ..core.llm import LLMClient
from ..core.models import ResearchResult
from ..core.prompts import learning_reflection_prompt, skill_extraction_prompt
from ..core.utils import extract_json_object
from ..memory.memory_manager import MemoryManager


class ReflectionEngine:
    """
    自我学习引擎：研究完成后反思、提炼、存储。

    用法：
        engine = ReflectionEngine(llm, memory_manager)
        reflection = engine.reflect(result)
        # reflection: {"episode_id", "quality_score", "insights_summary",
        #              "tags", "skills_learned"}
    """

    def __init__(self, llm: LLMClient, memory: MemoryManager):
        self.llm = llm
        self.memory = memory

    def reflect(self, result: ResearchResult) -> Dict[str, Any]:
        """完整反思流程：评分 → 技能使用反馈 → 提炼洞见 → 提取新技能 → 存储。"""
        quality = self._compute_quality(result)

        # 按质量分反馈本次检索到/使用过的技能
        # 说明：MemoryManager 在 Orchestrator 规划阶段调 format_context_for_prompt
        # 时已把"本次用了哪些技能"记进 _last_used_skill_ids，这里统一反馈。
        SUCCESS_THRESHOLD = 0.65
        skills_feedback = self.memory.feedback_skills_usage(
            success=quality >= SUCCESS_THRESHOLD
        )

        insights = self._extract_insights(result, quality)
        skills = self._extract_skills(result)

        ep_id = self.memory.save_research_episode(
            topic=result.topic,
            content=result.final_report_markdown,
            insights=insights["summary"],
            tags=insights.get("tags", []),
            quality_score=quality,
        )

        skills_saved = 0
        for sk in skills:
            if sk.get("name") and sk.get("content"):
                self.memory.save_skill(
                    name=sk["name"],
                    description=sk.get("description", ""),
                    trigger_conditions=sk.get("trigger_conditions", ""),
                    content=sk["content"],
                    domain=sk.get("domain", "general"),
                )
                skills_saved += 1

        return {
            "episode_id": ep_id,
            "quality_score": quality,
            "insights_summary": insights["summary"],
            "tags": insights.get("tags", []),
            "lessons_learned": insights.get("lessons", []),
            "skills_learned": skills_saved,
            "skills_feedback": skills_feedback,
        }

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #

    def _compute_quality(self, result: ResearchResult) -> float:
        """
        综合质量分：
          - 置信度均值 * 0.4
          - Critic 最终评分 * 0.4（若无 Critic 评审则回退为置信度）
          - 有引用 * 0.1
          - 有论文召回 * 0.1
        """
        if not result.task_results:
            return 0.0
        avg_conf = sum(t.confidence for t in result.task_results) / len(result.task_results)

        # Critic 权威分（取最后一轮评审）
        if result.critic_reviews:
            critic_score = result.critic_reviews[-1].score
        else:
            critic_score = avg_conf  # 回退：没评审就用置信度本身

        has_citations = any(t.citations for t in result.task_results)
        has_papers = bool(result.papers)

        score = (
            avg_conf * 0.4
            + critic_score * 0.4
            + (0.1 if has_citations else 0.0)
            + (0.1 if has_papers else 0.0)
        )
        return round(min(score, 1.0), 3)

    def _extract_insights(self, result: ResearchResult, quality: float) -> Dict:
        prompt = learning_reflection_prompt(result, quality)
        raw = self.llm.invoke([{"role": "user", "content": prompt}], temperature=0.2)
        data = extract_json_object(raw)
        return {
            "summary": data.get("insights_summary", ""),
            "tags": data.get("tags", []),
            "lessons": data.get("lessons_learned", []),
        }

    def _extract_skills(self, result: ResearchResult) -> List[Dict]:
        prompt = skill_extraction_prompt(result)
        raw = self.llm.invoke([{"role": "user", "content": prompt}], temperature=0.2)
        data = extract_json_object(raw)
        return data.get("skills", [])

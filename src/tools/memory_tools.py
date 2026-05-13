"""
记忆系统工具：读 / 写 / 反思

- retrieve_memory       : 跨三层召回 + rerank
- save_note             : 保存任务级摘要（Vector）
- save_research_episode : 保存完整研究情节 + 反思提炼技能
"""

from typing import List

from ..learning.reflection import ReflectionEngine
from ..memory.memory_manager import MemoryManager
from .tool import Tool, ToolResult


def build_retrieve_memory_tool(memory: MemoryManager) -> Tool:
    def run(args):
        query = (args.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, error="query is required")
        text = memory.format_context_for_prompt(query)
        if not text:
            return ToolResult(
                success=True,
                content="(记忆库无相关内容，这是首次研究该主题)",
            )
        return ToolResult(success=True, content=text)

    return Tool(
        name="retrieve_memory",
        description=(
            "Search the agent's long-term memory (episodic / skill / vector) for prior "
            "research related to the query. Returns a formatted string combining hits "
            "from all three layers, already reranked. Call this at the start of a "
            "research session to avoid duplicating prior work."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantic query, natural language is fine.",
                },
            },
            "required": ["query"],
        },
        run=run,
    )


def build_save_note_tool(memory: MemoryManager) -> Tool:
    def run(args):
        doc_id = (args.get("doc_id") or "").strip()
        title = (args.get("title") or "").strip()
        body = (args.get("body") or "").strip()
        if not doc_id or not body:
            return ToolResult(success=False, error="doc_id and body are required")
        memory.save_task_result(doc_id, title or doc_id, body)
        return ToolResult(success=True, content={"saved": doc_id})

    return Tool(
        name="save_note",
        description=(
            "Save an intermediate research note (a sub-task summary) to the vector store. "
            "Use when you finish investigating a sub-topic and want it searchable later."
        ),
        parameters={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Unique id, recommended format 'task:<slug>' or 'finding:<slug>'",
                },
                "title": {"type": "string"},
                "body": {"type": "string", "description": "Markdown body"},
            },
            "required": ["doc_id", "body"],
        },
        run=run,
    )


def build_save_research_episode_tool(
    memory: MemoryManager,
    reflection_engine: ReflectionEngine,
) -> Tool:
    """
    保存完整研究会话 + 触发反思（提炼洞见和技能）。

    因为 ReflectionEngine.reflect 需要一个 ResearchResult 对象，而 agent 循环中
    我们没有那个结构，这里暴露一个简化接口：只需要 topic + final_report，
    内部构造最小的 ResearchResult stub 给 reflection_engine。
    """
    from ..core.models import ResearchResult

    def run(args):
        topic = (args.get("topic") or "").strip()
        report = (args.get("final_report") or "").strip()
        quality_hint = float(args.get("quality_hint", 0.6))
        if not topic or not report:
            return ToolResult(success=False, error="topic and final_report are required")

        # 构造最小 ResearchResult stub（没有 task_results/papers/critic_reviews）
        # _compute_quality 会因 task_results 为空返回 0.0，所以我们用 quality_hint 注入
        stub = ResearchResult(
            topic=topic,
            plan=[],
            task_results=[],
            final_report_markdown=report,
            report_file="",
            papers=[],
        )
        # 走完整反思流程
        out = reflection_engine.reflect(stub)

        # 如果 _compute_quality 给了 0（因为 task_results 为空），
        # 我们用 agent 提供的 quality_hint 覆写一下（让质量分反馈/阈值有意义）
        if out.get("quality_score", 0) == 0 and quality_hint > 0:
            out["quality_score"] = quality_hint

        return ToolResult(success=True, content=out)

    return Tool(
        name="save_research_episode",
        description=(
            "Save a completed research session as a long-term episode, and automatically "
            "extract insights + reusable skills via the reflection engine. Call this ONCE "
            "at the very end of a research session, right before finish()."
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "final_report": {
                    "type": "string",
                    "description": "The final markdown report text",
                },
                "quality_hint": {
                    "type": "number",
                    "description": "0-1, your self-assessed quality of this research. "
                                    "Used only if internal scoring can't compute.",
                    "default": 0.6,
                },
            },
            "required": ["topic", "final_report"],
        },
        run=run,
    )

"""
ReviserWorker — LangGraph 版自主修订 Agent

用 StateGraph + ToolNode 实现 agent 循环，替代手搓的 BaseAgent。

架构：
    [reviser_agent] ←→ [search_tools]
         ↓ (无 tool_calls 时结束)
        [END]

LLM 搜索补充材料后直接输出修订报告（最后一条 AIMessage 即为报告）。
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from ..core.config import Settings
from ..core.models import CriticReview
from ..services.paper_search import ArxivSearcher, SemanticScholarSearcher


REVISER_SYSTEM_PROMPT = """You are an expert research report editor. You receive a report along with a peer review critique, and your job is to revise the report to address all identified weaknesses.

## Your Tools
- **search_arxiv**: Search for supplementary papers to fill gaps identified by the critic.
- **search_semantic_scholar**: Alternative academic search.

## Revision Strategy
1. Read the critic's suggestions and missing_topics carefully.
2. For each missing_topic, search for 1-2 relevant papers to get supporting evidence.
3. Rewrite the report:
   - Strengthen weak sections with new evidence
   - Add coverage for missing topics
   - Keep strong sections largely intact
   - Only cite URLs from papers you actually found via search
4. Output the complete revised Markdown report as your final message (do NOT call any tools).

## Rules
- The revised report must be COMPLETE (not just the changed sections)
- Preserve the original structure unless the critic specifically flagged structural issues
- Add new citations with real arXiv URLs from your searches
- Output language: same as the original report (usually Chinese)
- Do NOT invent citations — only use papers you found via search tools
"""


# ------------------------------------------------------------------ #
# State
# ------------------------------------------------------------------ #

class ReviserState(TypedDict):
    messages: Annotated[list, add_messages]


# ------------------------------------------------------------------ #
# Worker
# ------------------------------------------------------------------ #

class ReviserWorker:
    """LangGraph 版自主修订 Agent。"""

    def __init__(self, settings: Settings, max_steps: int = 10):
        self._settings = settings
        self._max_steps = max_steps

    def revise(self, topic: str, report_md: str, review: CriticReview) -> str:
        """修订报告，返回修改后的 Markdown。"""
        graph = self._build_graph()

        suggestions = "\n".join(f"- {s}" for s in review.suggestions)
        missing = "\n".join(f"- {t}" for t in review.missing_topics)
        dim_text = "\n".join(f"  {k}: {v:.2f}" for k, v in review.dimension_scores.items())

        task = (
            f"Revise this research report based on the critic's feedback.\n\n"
            f"Topic: {topic}\n"
            f"Critic score: {review.score:.2f}\n"
            f"Dimension scores:\n{dim_text}\n\n"
            f"Suggestions:\n{suggestions}\n\n"
            f"Missing topics to cover:\n{missing}\n\n"
            f"Original report:\n---\n{report_md}\n---\n\n"
            f"Search for supplementary sources on the missing topics, then rewrite the report "
            f"addressing all suggestions. Output the complete revised report as your final message."
        )

        initial_messages = [
            SystemMessage(content=REVISER_SYSTEM_PROMPT),
            HumanMessage(content=task),
        ]

        config = {"recursion_limit": self._max_steps * 2}
        result = graph.invoke({"messages": initial_messages}, config=config)

        # 提取最后一条 AI 文本消息作为修订报告
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not hasattr(msg, "tool_calls"):
                return msg.content
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                return msg.content

        return report_md  # fallback：返回原报告

    def _build_graph(self):
        """构建 Reviser 修订图。"""
        arxiv = ArxivSearcher(max_results=self._settings.search_top_k)
        s2 = SemanticScholarSearcher()

        @tool
        def search_arxiv(query: str, max_results: int = 5) -> str:
            """Search arXiv for academic papers. Use short English keyword queries (3-8 words)."""
            if not query.strip():
                return "[Error] query is required"
            try:
                papers = arxiv.search(query, max_results=max_results)
            except Exception as e:
                return f"[Error] arXiv search failed: {str(e)[:200]}"
            if not papers:
                return f"[No results] arXiv returned 0 papers for '{query}'."
            lines = [
                f"- **{p.title}**\n  Authors: {', '.join(p.authors[:3])}\n  "
                f"Abstract: {(p.abstract or '')[:300]}\n  URL: {p.url}\n  ID: {p.paper_id}"
                for p in papers[:max_results]
            ]
            return f"Found {len(papers)} papers:\n\n" + "\n\n".join(lines)

        @tool
        def search_semantic_scholar(query: str, max_results: int = 5) -> str:
            """Search Semantic Scholar for academic papers. Broader coverage than arXiv."""
            if not query.strip():
                return "[Error] query is required"
            try:
                papers = s2.search(query, max_results=max_results)
            except Exception as e:
                return f"[Error] Semantic Scholar search failed: {str(e)[:200]}"
            if not papers:
                return f"[No results] Semantic Scholar returned 0 papers for '{query}'."
            lines = [
                f"- **{p.title}**\n  Authors: {', '.join(p.authors[:3])}\n  "
                f"Abstract: {(p.abstract or '')[:300]}\n  URL: {p.url}"
                for p in papers[:max_results]
            ]
            return f"Found {len(papers)} papers:\n\n" + "\n\n".join(lines)

        tools = [search_arxiv, search_semantic_scholar]

        llm = ChatOpenAI(
            model=self._settings.llm_model,
            api_key=self._settings.llm_api_key,
            base_url=self._settings.llm_base_url,
            temperature=0.3,
        ).bind_tools(tools)

        def agent_node(state: ReviserState):
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        tool_node = ToolNode(tools)

        def should_continue(state: ReviserState) -> Literal["tools", "end"]:
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        graph = StateGraph(ReviserState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)

        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
        graph.add_edge("tools", "agent")

        return graph.compile()

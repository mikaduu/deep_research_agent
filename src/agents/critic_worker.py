"""
CriticWorker — LangGraph 版自主评审 Agent

用 StateGraph + ToolNode 实现 agent 循环，替代手搓的 BaseAgent。

架构：
    [critic_agent] ←→ [check_url_tool]
         ↓ (无 tool_calls 时结束)
        [END]

LLM 通过 submit_review 工具提交结构化评审结果。
"""

import json
from typing import Annotated, Any, Dict, List, Literal
from typing_extensions import TypedDict

import requests
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from ..core.config import Settings
from ..core.models import CriticReview


CRITIC_SYSTEM_PROMPT = """You are a strict academic peer reviewer. You evaluate research reports independently and objectively.

## Your Tools
- **check_url**: Verify if a citation URL actually exists (HTTP HEAD check).
- **submit_review**: Submit your final structured review. You MUST call this when done.

## Evaluation Process
1. Read the report carefully.
2. Check 2-3 citation URLs to verify they're real (don't check all — just spot-check).
3. Evaluate four dimensions (each 0-1):
   - **coverage**: Does the report cover the topic's core aspects?
   - **evidence_quality**: Are citations real and relevant? Are claims backed by evidence?
   - **coherence**: Is the logic clear? Is the structure well-organized?
   - **actionability**: Are recommendations specific and executable?
4. Call submit_review with your structured review.

## Scoring Rules
- score = average of the four dimension scores
- A score above 0.85 means publication-ready
- Be strict: most first drafts score 0.5-0.7
- If you find a fake/broken citation, deduct 0.1 from evidence_quality
"""


# ------------------------------------------------------------------ #
# State
# ------------------------------------------------------------------ #

class CriticState(TypedDict):
    messages: Annotated[list, add_messages]
    review_result: dict  # 存储 submit_review 的结果


# ------------------------------------------------------------------ #
# Worker
# ------------------------------------------------------------------ #

class CriticWorker:
    """LangGraph 版自主评审 Agent。"""

    def __init__(self, settings: Settings, max_steps: int = 8):
        self._settings = settings
        self._max_steps = max_steps

    def review(self, topic: str, report_md: str) -> CriticReview:
        """评审报告，返回 CriticReview。"""
        graph = self._build_graph()

        initial_messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Review this research report on topic: {topic}\n\n"
                f"---\n\n{report_md}\n\n"
                f"---\n\nAfter checking URLs and evaluating, call submit_review with your structured review."
            )),
        ]

        config = {"recursion_limit": self._max_steps * 2}
        result = graph.invoke(
            {"messages": initial_messages, "review_result": {}},
            config=config,
        )

        # 从 review_result 提取结构化数据
        data = result.get("review_result", {})
        if not data:
            # fallback：尝试从最后一条 AIMessage 解析
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    try:
                        data = json.loads(msg.content)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break

        dim_scores = {k: float(v) for k, v in data.get("dimension_scores", {}).items()}
        score = float(data.get("score", 0.5))
        if dim_scores and not data.get("score"):
            score = sum(dim_scores.values()) / len(dim_scores)

        return CriticReview(
            score=round(score, 3),
            needs_revision=score < self._settings.critic_threshold,
            dimension_scores=dim_scores,
            suggestions=data.get("suggestions", []),
            missing_topics=data.get("missing_topics", []),
            strengths=data.get("strengths", []),
        )

    def _build_graph(self):
        """构建 Critic 评审图。"""
        # 闭包变量，用于捕获 submit_review 的结果
        review_holder = {"result": {}}

        @tool
        def check_url(url: str) -> str:
            """Verify if a URL exists by sending an HTTP HEAD request. Returns status code."""
            try:
                resp = requests.head(url.strip(), timeout=5, allow_redirects=True)
                exists = resp.status_code < 400
                return json.dumps({"url": url, "exists": exists, "status_code": resp.status_code})
            except Exception as e:
                return json.dumps({"url": url, "exists": False, "error": str(e)[:100]})

        @tool
        def submit_review(
            score: float,
            dimension_scores: str,
            strengths: str,
            suggestions: str,
            missing_topics: str,
        ) -> str:
            """Submit your final structured review. Call this when evaluation is complete.
            dimension_scores, strengths, suggestions, missing_topics are JSON strings."""
            try:
                review_holder["result"] = {
                    "score": score,
                    "dimension_scores": json.loads(dimension_scores) if isinstance(dimension_scores, str) else dimension_scores,
                    "strengths": json.loads(strengths) if isinstance(strengths, str) else strengths,
                    "suggestions": json.loads(suggestions) if isinstance(suggestions, str) else suggestions,
                    "missing_topics": json.loads(missing_topics) if isinstance(missing_topics, str) else missing_topics,
                }
            except (json.JSONDecodeError, TypeError):
                review_holder["result"] = {
                    "score": score,
                    "dimension_scores": {},
                    "strengths": [strengths] if strengths else [],
                    "suggestions": [suggestions] if suggestions else [],
                    "missing_topics": [missing_topics] if missing_topics else [],
                }
            return "Review submitted successfully."

        tools = [check_url, submit_review]

        llm = ChatOpenAI(
            model=self._settings.llm_model,
            api_key=self._settings.llm_api_key,
            base_url=self._settings.llm_base_url,
            temperature=0.1,
        ).bind_tools(tools)

        def agent_node(state: CriticState):
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        tool_node = ToolNode(tools)

        def should_continue(state: CriticState) -> Literal["tools", "end"]:
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        def save_review(state: CriticState):
            """把 submit_review 的结果写入 state。"""
            return {"review_result": review_holder["result"]}

        graph = StateGraph(CriticState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_node("save_review", save_review)

        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": "save_review"})
        graph.add_edge("tools", "agent")
        graph.add_edge("save_review", END)

        return graph.compile()

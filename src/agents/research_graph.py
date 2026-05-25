"""
ResearchGraph — LangGraph 版自主研究 Agent

用 LangGraph 的 StateGraph + ToolNode 替代手搓的 BaseAgent loop。

核心区别：
- 手搓版：自己写 while loop + function calling 解析 + tool dispatch
- LangGraph 版：声明式定义 graph（nodes + edges），框架处理 loop/dispatch/state

架构：
    [agent node] ←→ [tool node]
         ↓ (finish)
    [end]

agent node: 调 LLM with tools，决定下一步
tool node:  执行工具，返回结果
条件边:     有 tool_calls → tool node；无 → end
"""

from typing import Annotated, Any, Dict, List, Literal
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from ..core.config import Settings
from ..core.llm import LLMClient
from ..memory.memory_manager import MemoryManager
from ..learning.reflection import ReflectionEngine
from ..services.paper_search import ArxivSearcher, SemanticScholarSearcher
from ..services.paper_fetcher import PaperFetcher
from ..services.web_searcher import WebSearcher
from ..agents.paper_analyzer import PaperAnalyzer

from ..tools.search_tools import (
    search_arxiv, search_semantic_scholar, init_search_tools,
)
from ..tools.web_tools import web_search, web_fetch, init_web_tools
from ..tools.paper_tools import (
    fetch_paper_fulltext, analyze_paper, init_paper_tools,
)
from ..tools.memory_tools import (
    retrieve_memory, save_note, save_research_episode, init_memory_tools,
)
from ..tools.delegation_tools import (
    delegate_to_critic, delegate_to_reviser, init_delegation_tools,
)


# ------------------------------------------------------------------ #
# State 定义
# ------------------------------------------------------------------ #

class ResearchState(TypedDict):
    """Graph 的状态。messages 是核心，LangGraph 自动管理累积。"""
    messages: Annotated[list, add_messages]


# ------------------------------------------------------------------ #
# System Prompt
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are a Research Manager Agent — the central brain of a multi-agent research system.

## Your Role
You autonomously conduct deep academic research on a given topic. You have a toolbox of capabilities and must decide, step by step, what to do next based on what you've learned so far.

## Available Tools
- **search_arxiv** / **search_semantic_scholar**: Search for papers. Use short English keyword queries (3-8 words).
- **fetch_paper_fulltext**: Download a paper's PDF and get section-level text.
- **analyze_paper**: Deep analysis of one paper (map-reduce). Use on 2-3 most important papers only.
- **retrieve_memory**: Check prior research on this topic. Call this FIRST.
- **save_note**: Save intermediate findings to memory.
- **web_search** / **web_fetch**: Search general web / fetch a URL's content.
- **delegate_to_critic**: Send draft report for independent peer review.
- **delegate_to_reviser**: Send report + review for revision with supplementary search.
- **save_research_episode**: Save completed session + trigger learning. Call ONCE at the end.

## Research Strategy
1. **Recall**: retrieve_memory first.
2. **Explore**: search with 3-5 different keyword angles (max 8 total searches).
3. **Deep dive**: analyze_paper on 2-3 most relevant papers.
4. **Draft**: compose a comprehensive Markdown report.
5. **Review**: delegate_to_critic for feedback.
6. **Revise**: if needed, delegate_to_reviser.
7. **Finalize**: save_research_episode, then stop.

## HARD CONSTRAINTS
1. **Total search calls ≤ 8** (search_arxiv + search_semantic_scholar + web_search combined).
2. **By message 12, START writing the report** with whatever material you have.
3. **Never call the same tool with the same query twice.**
4. **If a tool returns "rate-limited": switch to an alternative immediately.**
5. **Report language: Chinese** (unless user specifies otherwise).
6. **Search queries: ALWAYS English.**
7. **Mandatory final sequence (MUST follow this exact order):**
   - Step A: Output the complete draft report as a message (no tools).
   - Step B: Call delegate_to_critic on the draft report.
   - Step C: If critic score < 0.7, call delegate_to_reviser. Otherwise skip.
   - Step D: Output the FINAL report as a message.
   - Step E: Call save_research_episode ONCE.
   - Step F: Done. Do NOT call any more tools after save_research_episode.

## Report Format
Your final message (when you're done) should be the complete Markdown report including:
- TL;DR
- Detailed findings by subtopic
- Specific numbers and citations
- Limitations and open questions
- Recommendations
- References
"""


# ------------------------------------------------------------------ #
# Graph 构建
# ------------------------------------------------------------------ #

def build_research_graph(settings: Settings):
    """
    构建 LangGraph 研究图。

    返回编译好的 graph，可直接 .invoke() 或 .stream()。
    """
    # 1. 初始化 services
    memory = MemoryManager(settings)
    reflection = ReflectionEngine(LLMClient(settings), memory)
    arxiv = ArxivSearcher(max_results=settings.search_top_k)
    s2 = SemanticScholarSearcher()
    fetcher = PaperFetcher(settings.workspace_dir / "pdf_cache")
    web = WebSearcher()
    analyzer = PaperAnalyzer(LLMClient(settings), settings)

    # 2. 注入工具依赖
    init_search_tools(arxiv, s2)
    init_web_tools(web)
    init_paper_tools(fetcher, analyzer, arxiv, memory)
    init_memory_tools(memory, reflection)
    init_delegation_tools(settings)

    # 3. 工具列表
    tools = [
        search_arxiv,
        search_semantic_scholar,
        web_search,
        web_fetch,
        fetch_paper_fulltext,
        analyze_paper,
        retrieve_memory,
        save_note,
        save_research_episode,
        delegate_to_critic,
        delegate_to_reviser,
    ]

    # 4. LLM with tools
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.3,
    ).bind_tools(tools)

    # 5. 定义 nodes
    def agent_node(state: ResearchState):
        """调用 LLM，决定下一步。"""
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    # 6. 条件边：有 tool_calls → tools；无 → END
    def should_continue(state: ResearchState) -> Literal["tools", "end"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # 7. 构建 graph
    graph = StateGraph(ResearchState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile(), memory


def run_research(settings: Settings, topic: str) -> Dict[str, Any]:
    """
    执行一次完整研究。

    Returns:
        {"report": str, "messages": list, "memory_stats": dict}
    """
    graph, memory = build_research_graph(settings)

    initial_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"请对以下主题进行深度研究并生成完整报告：\n\n{topic}"),
    ]

    # 用 stream 模式逐步执行，实时打印进度
    # stream_mode="updates" 每步返回增量 state
    config = {"recursion_limit": 60}
    step = 0
    final_messages = []
    reviser_report = None   # 捕获 Reviser 的修订报告
    critic_report = None    # 捕获传给 Critic 的草稿报告（或 Agent 的长文本输出）

    for event in graph.stream({"messages": initial_messages}, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            step += 1
            msgs = node_output.get("messages", [])
            if not msgs:
                continue
            last_msg = msgs[-1]

            if node_name == "agent":
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_names = [tc["name"] for tc in last_msg.tool_calls]
                    print(f"  [{step}] Agent → 调用工具: {', '.join(tool_names)}")
                    # 从 tool_call 参数中提取报告内容
                    import json as _json
                    for tc in last_msg.tool_calls:
                        tc_name = tc.get("name", "")
                        tc_args = tc.get("args", {})
                        if tc_name == "delegate_to_critic":
                            candidate = tc_args.get("report_md", "")
                            if candidate and len(candidate) > 200:
                                critic_report = candidate
                else:
                    content = getattr(last_msg, "content", "")
                    # 长文本输出也可能是报告（只在比现有报告更长时覆盖）
                    if content and len(content) > 500 and len(content) > len(critic_report or ""):
                        critic_report = content
                    preview = content[:80].replace("\n", " ") if content else "(empty)"
                    print(f"  [{step}] Agent → 输出文本 ({len(content)} chars): {preview}...")

            elif node_name == "tools":
                for msg in msgs:
                    if hasattr(msg, "name"):
                        tool_name = msg.name
                        content = getattr(msg, "content", "")
                        if tool_name == "delegate_to_reviser" and content and len(content) > 500:
                            reviser_report = content
                        preview = content[:60].replace("\n", " ") if content else ""
                        print(f"  [{step}] 工具 [{tool_name}] → {preview}...")

            final_messages.extend(msgs)

    # 报告优先级：Reviser 修订版 > Critic 草稿 > Agent 最后消息
    from langchain_core.messages import AIMessage as _AIMessage
    if reviser_report:
        report = reviser_report
    elif critic_report:
        report = critic_report
    else:
        report = ""
        for msg in reversed(final_messages):
            if isinstance(msg, _AIMessage) and msg.content and not msg.tool_calls:
                report = msg.content
                break

    return {
        "report": report,
        "total_messages": len(final_messages),
        "memory_stats": memory.stats(),
    }

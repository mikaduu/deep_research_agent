"""
ResearchManager - 自主研究经理 Agent。

作为多 agent 研究系统的主脑，负责：
- 搜索与筛选论文
- 按需深读关键论文
- 调用子 worker 做专题并行调研
- 调用 critic / reviser 做独立审稿与修订
"""

from ..agents.base_agent import AgentStep, BaseAgent
from ..agents.paper_analyzer import PaperAnalyzer
from ..core.config import Settings
from ..core.llm import LLMClient
from ..learning.reflection import ReflectionEngine
from ..memory.memory_manager import MemoryManager
from ..services.paper_fetcher import PaperFetcher
from ..services.paper_search import ArxivSearcher, SemanticScholarSearcher
from ..services.web_searcher import WebSearcher
from ..tools.delegation_tools import (
    build_delegate_to_critic_tool,
    build_delegate_to_reviser_tool,
)
from ..tools.memory_tools import (
    build_retrieve_memory_tool,
    build_save_note_tool,
    build_save_research_episode_tool,
)
from ..tools.paper_tools import build_analyze_paper_tool, build_fetch_fulltext_tool
from ..tools.search_tools import build_arxiv_search_tool, build_s2_search_tool
from ..tools.thread_tools import build_delegate_research_threads_tool
from ..tools.tool import ToolRegistry
from ..tools.web_tools import build_web_fetch_tool, build_web_search_tool


MANAGER_SYSTEM_PROMPT = """You are a Research Manager Agent, the central brain of a multi-agent research system.

## Your Role
You autonomously conduct deep academic research on a given topic. You have a toolbox of capabilities and must decide, step by step, what to do next based on what you've learned so far.

## Available Tools
- **search_arxiv** / **search_semantic_scholar**: Search for papers. Use short English keyword queries (3-8 words). Try multiple angles.
- **fetch_paper_fulltext**: Download a paper's PDF and get section-level text. Use when abstract isn't enough.
- **analyze_paper**: Deep analysis of one paper in conservative mode. Use on the 2-3 most important papers only.
- **retrieve_memory**: Check if you've researched this topic before. Call this FIRST to avoid repeating prior work.
- **save_note**: Save intermediate findings to memory so future research can build on them.
- **delegate_research_threads**: Split the topic into several focused threads such as baselines, related methods, limitations, benchmarks, or competing approaches, then let independent worker agents investigate them in parallel.
- **delegate_to_critic**: Send your draft report to an independent Critic for peer review.
- **delegate_to_reviser**: Send report + critic review to a Reviser who will search for supplementary sources and rewrite weak sections.
- **save_research_episode**: Save the completed research session and trigger self-learning. Call this ONCE at the very end.
- **finish**: End the research and return the final report.

## Research Strategy
1. Recall: call retrieve_memory first.
2. Explore: search with 3-5 different keyword angles.
3. Threading: if the topic naturally contains multiple aspects, use delegate_research_threads to split it into 2-4 focused threads such as baseline comparison, representative methods, limitations, or benchmarks.
4. Deep dive: for the 2-3 most relevant papers, use analyze_paper to get full details.
5. Synthesize: save_note with your intermediate findings as you go.
6. Draft: compose a comprehensive Markdown report in Chinese.
7. Review: delegate_to_critic for independent feedback.
8. Revise: if score < 0.7, use delegate_to_reviser.
9. Finalize: call save_research_episode, then finish.

## How To Use Thread Delegation Well
- Create clear, non-overlapping threads.
- Good thread examples: `baseline comparison`, `method family`, `limitations and failure modes`, `benchmarks and datasets`.
- Prefer 2-4 threads, not more.
- Only trust medium/high-confidence thread results.
- If a delegated thread returns low confidence or noisy evidence, ignore or downweight it in the final report.

## Report Quality Standards
Your final report MUST include:
- Executive summary (TL;DR)
- Detailed findings organized by subtopic
- A dedicated baseline / related work comparison section when the topic involves competing methods
- Specific numbers and citations (paper title + arXiv URL)
- Risks, limitations, and open questions
- Actionable recommendations
- References section with all cited papers

## Decision Rules
- If a search returns 0 results: try a different query formulation, do not give up.
- If you have searched 3+ times with no useful results: acknowledge the gap and move on.
- If the Critic scores < 0.5: you need major revision, consider searching more before revising.
- If the Critic scores 0.5-0.7: delegate_to_reviser should suffice.
- If the Critic scores > 0.7: report is acceptable, proceed to finalize.
- Maximum one revision cycle. If still < 0.7 after revision, finalize anyway with a note about limitations.
- Always call save_research_episode before finish.

## Hard Constraints
1. Total search calls <= 8: count all search tool calls (search_arxiv + search_semantic_scholar + web_search combined). After 8 searches, stop searching and start writing immediately.
2. By step 12, you must have called finish or be writing the report.
3. Never call the same tool with the same query twice.
4. If a tool returns an error mentioning "rate-limited", do not retry that tool. Switch to an alternative immediately.

## Language
- Search queries: always in English.
- Report output: Chinese unless the user specifies otherwise.
- Tool arguments: English.

## Token Budget Awareness
You have a strict token budget (~200K). Each step costs ~5-10K tokens.
- Steps 1-8: search, thread delegation, and gather material
- Steps 9-12: write report + delegate_to_critic
- Steps 13-15: revise if needed + save_research_episode + finish
- Do not spend more than 8 steps on searching.
- If a search tool fails, do not retry it. Use a different source or move on.
"""


class ResearchManager(BaseAgent):
    """自主研究经理 Agent。"""

    def __init__(
        self,
        settings: Settings,
        max_steps: int = 30,
        max_total_tokens: int = 300_000,
    ):
        self.settings = settings

        self.memory = MemoryManager(settings)
        self.reflection = ReflectionEngine(LLMClient(settings), self.memory)
        self.arxiv = ArxivSearcher(max_results=settings.search_top_k)
        self.s2 = SemanticScholarSearcher()
        self.fetcher = PaperFetcher(settings.workspace_dir / "pdf_cache")
        self.analyzer = PaperAnalyzer(LLMClient(settings), settings)
        self.web = WebSearcher()

        super().__init__(
            llm=LLMClient(settings),
            max_steps=max_steps,
            max_total_tokens=max_total_tokens,
            temperature=0.3,
        )

    def system_prompt(self) -> str:
        return MANAGER_SYSTEM_PROMPT

    def build_tools(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(build_arxiv_search_tool(self.arxiv))
        registry.register(build_s2_search_tool(self.s2))
        registry.register(build_fetch_fulltext_tool(self.fetcher))
        registry.register(build_analyze_paper_tool(self.analyzer, self.arxiv, self.memory))
        registry.register(build_retrieve_memory_tool(self.memory))
        registry.register(build_save_note_tool(self.memory))
        registry.register(build_save_research_episode_tool(self.memory, self.reflection))
        registry.register(build_delegate_research_threads_tool(self.settings))
        registry.register(build_delegate_to_critic_tool(self.settings))
        registry.register(build_delegate_to_reviser_tool(self.settings))
        registry.register(build_web_search_tool(self.web))
        registry.register(build_web_fetch_tool(self.web))
        return registry

    def on_step(self, step: AgentStep) -> None:
        status = "OK" if (step.tool_result and step.tool_result.success) else "ERR"
        if step.tool_name:
            print(
                f"  [{step.step_idx + 1}] {status} {step.tool_name}  "
                f"({step.tokens_used} tokens, {step.elapsed_ms}ms)"
            )
        elif step.thought:
            print(f"  [{step.step_idx + 1}] THINK {step.thought[:80]}")

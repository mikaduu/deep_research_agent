from typing import List

from .models import CriticReview, ResearchResult, SourceItem, TaskPlanItem, TaskRunResult


def critic_prompt(topic: str, report_md: str) -> str:
    """
    评审 Agent 提示词（独立 system prompt 隔离角色）

    评判维度：
    - coverage: 是否覆盖主题的核心方面（0-1）
    - evidence_quality: 引用是否充分、来源是否可信（0-1）
    - coherence: 逻辑是否连贯、结构是否清晰（0-1）
    - actionability: 建议是否具体可执行（0-1）
    """
    return f"""Topic under review: {topic}

Report to review:
{report_md}

Evaluate this research report strictly and objectively.
Return JSON only:
{{
  "dimension_scores": {{
    "coverage": 0.0,
    "evidence_quality": 0.0,
    "coherence": 0.0,
    "actionability": 0.0
  }},
  "score": 0.0,
  "strengths": ["strength 1", "strength 2"],
  "suggestions": ["specific improvement 1", "specific improvement 2"],
  "missing_topics": ["topic that needs more research 1"]
}}

Rules:
- score = average of dimension_scores
- suggestions must be specific and actionable, not vague
- missing_topics: list subtopics that lack sufficient evidence and need supplementary search
- be strict: a score above 0.85 means the report is publication-ready
""".strip()


def reviser_prompt(
    topic: str,
    report_md: str,
    review: CriticReview,
    extra_sources: List[SourceItem],
) -> str:
    """
    修改 Agent 提示词（独立 system prompt 隔离角色）

    基于评审意见和补充搜索结果，重写报告中的薄弱部分。
    """
    suggestions_text = "\n".join(f"- {s}" for s in review.suggestions)
    missing_text = "\n".join(f"- {t}" for t in review.missing_topics)
    dim_text = "\n".join(
        f"  {k}: {v:.2f}" for k, v in review.dimension_scores.items()
    )
    sources_text = ""
    if extra_sources:
        sources_text = "\nSupplementary sources found:\n" + "\n".join(
            f"[{i+1}] {s.title}\nURL: {s.url}\nSnippet: {s.snippet}"
            for i, s in enumerate(extra_sources)
        )

    return f"""Topic: {topic}

Critic score: {review.score:.2f}
Dimension scores:
{dim_text}

Suggestions from critic:
{suggestions_text}

Topics needing more coverage:
{missing_text}
{sources_text}

Original report:
{report_md}

Rewrite the report addressing all critic suggestions.
- Incorporate supplementary sources where relevant (only cite provided URLs)
- Strengthen weak sections identified by the critic
- Keep strong sections largely intact
- Return the complete revised report in Markdown format (no JSON wrapper)
""".strip()


def memory_augmented_planner_prompt(topic: str, max_items: int, memory_context: str) -> str:
    """
    记忆增强规划提示词

    在标准规划基础上注入历史研究情节和已学技能，
    让规划Agent避免重复已知内容，直接深入未知领域。
    """
    return f"""
You are a planner agent for deep research with access to accumulated research memory.
Generate a concise research plan for this topic: {topic}

Accumulated memory context (use to avoid repeating known findings and go deeper):
{memory_context if memory_context else "No prior memory available."}

Return JSON only with schema:
{{
  "tasks": [
    {{
      "title": "task short title",
      "goal": "what this task should answer",
      "search_query": "web search query"
    }}
  ]
}}

Constraints:
- 3 to {max_items} tasks
- no duplicated task intent
- tasks should progressively build toward a final report
- if memory context exists, build on it rather than repeating what is already known
""".strip()


def learning_reflection_prompt(result: ResearchResult, quality_score: float) -> str:
    """
    自我学习反思提示词

    研究完成后，提炼关键洞见、标签和经验教训，
    存入情节记忆供未来研究参考。
    """
    task_summaries = "\n\n".join(
        f"Task: {t.task.title}\nSummary: {t.summary_markdown[:500]}"
        for t in result.task_results
    )
    return f"""
You are a self-learning research agent reflecting on a completed research session.

Topic: {result.topic}
Quality Score: {quality_score:.2f} (0=poor, 1=excellent)

Task Summaries:
{task_summaries}

Extract the most important insights from this research session.
Return JSON only:
{{
  "insights_summary": "2-3 sentence summary of the most important findings worth remembering",
  "tags": ["tag1", "tag2", "tag3"],
  "lessons_learned": ["lesson about research approach 1", "lesson 2"]
}}
""".strip()


def skill_extraction_prompt(result: ResearchResult) -> str:
    """
    技能提取提示词

    从研究会话中识别可复用的研究策略和模式，
    存入技能记忆供未来研究调用。
    """
    task_summaries = "\n\n".join(
        f"Task: {t.task.title}\nGoal: {t.task.goal}\nSearch Query: {t.task.search_query}"
        for t in result.task_results
    )
    return f"""
You are a self-learning research agent extracting reusable research skills from a completed session.

Topic: {result.topic}

Research Plan Used:
{task_summaries}

Identify 1-3 reusable research strategies or patterns from this session that would help future research.
Return JSON only:
{{
  "skills": [
    {{
      "name": "short skill name",
      "description": "what this skill does",
      "trigger_conditions": "when to apply this skill (e.g. 'when researching survey papers')",
      "content": "the actual strategy or pattern in 1-3 sentences",
      "domain": "research domain (e.g. machine_learning, nlp, computer_vision, general)"
    }}
  ]
}}

Only extract genuinely reusable patterns. Return empty skills list if nothing is worth saving.
""".strip()


def planner_prompt(topic: str, max_items: int) -> str:
    """
    规划Agent提示词

    功能：为研究主题生成结构化的任务计划
    输入：研究主题、最大任务数
    输出：JSON格式的任务列表，每个任务包含标题、目标、搜索查询

    约束：
    - 生成3到max_items个任务
    - 任务之间不重复
    - 任务应逐步构建最终报告
    """
    return f"""
You are a planner agent for deep research.
Generate a concise research plan for this topic: {topic}

Return JSON only with schema:
{{
  "tasks": [
    {{
      "title": "task short title",
      "goal": "what this task should answer",
      "search_query": "web search query"
    }}
  ]
}}

Constraints:
- 3 to {max_items} tasks
- no duplicated task intent
- tasks should progressively build toward a final report
""".strip()


def summarizer_prompt(
    topic: str,
    task: TaskPlanItem,
    sources: List[SourceItem],
    rag_context: str,
) -> str:
    """
    摘要Agent提示词

    功能：基于网络搜索结果和RAG记忆，为单个任务生成摘要
    输入：
    - topic: 研究主题
    - task: 当前任务信息（标题、目标）
    - sources: 网络搜索到的论文/资料列表
    - rag_context: 从记忆系统检索到的相关历史内容

    输出：JSON格式，包含：
    - summary_markdown: 任务摘要（Markdown格式）
    - key_points: 关键要点列表
    - citations: 引用列表（标题、URL、引用原因）
    - confidence: 置信度（0-1）

    规则：
    - 只能引用提供的网络证据中的URL
    - 置信度范围0到1
    - 如果证据不足，明确标记不确定性
    """
    sources_text = "\n".join(
        f"[{i+1}] {src.title}\nURL: {src.url}\nSnippet: {src.snippet}"
        for i, src in enumerate(sources)
    )
    return f"""
You are a research summarizer.
Topic: {topic}
Current task title: {task.title}
Current task goal: {task.goal}

Retrieved memory context (RAG):
{rag_context}

Web evidence:
{sources_text}

Return JSON only with schema:
{{
  "summary_markdown": "concise markdown summary for this task",
  "key_points": ["point 1", "point 2"],
  "citations": [
    {{
      "title": "source title",
      "url": "https://...",
      "reason": "why this source supports your point"
    }}
  ],
  "confidence": 0.0
}}

Rules:
- only cite URLs from provided web evidence
- confidence range is 0 to 1
- explicitly mark uncertainty if evidence is weak
""".strip()


def reflection_prompt(topic: str, result: TaskRunResult) -> str:
    """
    反思Agent提示词

    功能：检查研究质量，判断是否需要补充搜索
    输入：
    - topic: 研究主题
    - result: 当前任务的执行结果（包含摘要）

    输出：JSON格式，包含：
    - needs_more_research: 是否需要更多研究（布尔值）
    - follow_up_query: 后续搜索查询（如果需要）
    - reason: 判断原因

    规则：
    - 只有在存在重要事实空白时才设置needs_more_research=true
    """
    return f"""
You are a critic agent checking research quality.
Topic: {topic}
Task title: {result.task.title}
Task goal: {result.task.goal}
Current summary:
{result.summary_markdown}

Return JSON only:
{{
  "needs_more_research": true or false,
  "follow_up_query": "new query string if needed else empty",
  "reason": "short reason"
}}

Set needs_more_research=true only if there is an important factual gap.
""".strip()


def reporter_prompt(topic: str, task_results: List[TaskRunResult]) -> str:
    """
    报告写作Agent提示词

    功能：将所有任务结果整合为完整的研究报告
    输入：
    - topic: 研究主题
    - task_results: 所有任务的执行结果列表

    输出：Markdown格式的完整报告，必须包含：
    1. 执行摘要（Executive summary）
    2. 分章节的详细发现（Detailed findings）
    3. 风险和不确定性（Risks and uncertainty）
    4. 可行建议（Actionable recommendations）
    5. 带URL的参考文献（References）
    """
    blocks = []
    for i, item in enumerate(task_results):
        cits = "\n".join(f"- {c.title} ({c.url})" for c in item.citations)
        blocks.append(
            f"""
Task {i+1}: {item.task.title}
Goal: {item.task.goal}
Summary:
{item.summary_markdown}

Citations:
{cits or "- None"}
""".strip()
        )
    payload = "\n\n".join(blocks)
    return f"""
You are a report writer agent.
Produce a final markdown report for topic: {topic}

You must include:
1. Executive summary
2. Detailed findings by section
3. Risks and uncertainty
4. Actionable recommendations
5. References with URLs

Use these task materials:
{payload}
""".strip()

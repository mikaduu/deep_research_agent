from typing import List

from .models import SourceItem, TaskPlanItem, TaskRunResult


def planner_prompt(topic: str, max_items: int) -> str:
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


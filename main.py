#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

from src.core.config import Settings
from src.orchestrator import ResearchOrchestrator

app = typer.Typer()
console = Console()


def get_orchestrator() -> ResearchOrchestrator:
    root = Path(__file__).parent
    settings = Settings.from_env(root)
    return ResearchOrchestrator(settings)


@app.command()
def evaluate(direction: str = typer.Argument(..., help="研究方向描述")):
    """评估研究方向的可行性和价值，检索相关论文"""
    console.print(Panel(f"[bold]评估研究方向:[/bold] {direction}", style="blue"))
    orch = get_orchestrator()

    with console.status("正在检索相关论文并评估..."):
        result = orch.evaluate_direction(direction)

    table = Table(title="评估结果")
    table.add_column("维度", style="cyan")
    table.add_column("得分", style="green")
    table.add_row("可行性", f"{result.get('feasibility', 0):.2f}")
    table.add_row("新颖性", f"{result.get('novelty', 0):.2f}")
    table.add_row("影响力", f"{result.get('impact', 0):.2f}")
    console.print(table)

    console.print(Panel(result.get("analysis", ""), title="详细分析"))

    papers = result.get("papers", [])
    if papers:
        console.print(f"\n[bold]找到 {len(papers)} 篇相关论文:[/bold]")
        for i, p in enumerate(papers[:5], 1):
            console.print(f"  {i}. {p.title}")
            console.print(f"     {p.url}")


@app.command()
def search(query: str = typer.Argument(..., help="搜索关键词")):
    """搜索相关论文"""
    orch = get_orchestrator()
    with console.status("搜索中..."):
        papers = orch.search_papers(query)

    console.print(f"\n[bold]找到 {len(papers)} 篇论文:[/bold]")
    for i, p in enumerate(papers, 1):
        console.print(f"\n[cyan]{i}. {p.title}[/cyan]")
        console.print(f"   作者: {', '.join(p.authors[:3])}")
        console.print(f"   摘要: {p.abstract[:150]}...")
        console.print(f"   链接: {p.url}")


@app.command()
def analyze(url: str = typer.Argument(..., help="论文arXiv ID或URL"),
            focus: str = typer.Option(None, "--focus", "-f", help="关注点")):
    """深度分析一篇论文"""
    from src.core.models import PaperItem
    orch = get_orchestrator()

    paper_id = url.split("/")[-1]
    with console.status("分析论文中..."):
        papers = orch.arxiv.search(f"id:{paper_id}", max_results=1)
        if not papers:
            console.print("[red]未找到该论文[/red]")
            raise typer.Exit(1)
        result = orch.analyze_paper(papers[0], focus)

    console.print(Panel(f"[bold]{papers[0].title}[/bold]"))
    source = result.get("_source", "unknown")
    if source == "fulltext":
        console.print(f"[dim]分析来源: 全文 PDF ({result.get('_num_pages', '?')} 页, "
                      f"{result.get('_num_chunks', '?')} 个章节块)[/dim]")
    else:
        console.print("[yellow]分析来源: 仅摘要（全文下载失败，已降级）[/yellow]")
    console.print(f"\n[bold]核心问题:[/bold] {result.get('problem', '')}")
    console.print(f"\n[bold]主要贡献:[/bold]")
    for c in result.get("contributions", []):
        console.print(f"  • {c}")
    console.print(f"\n[bold]方法:[/bold]")
    for m in result.get("methods", []):
        console.print(f"  • {m}")
    if result.get("datasets"):
        console.print(f"\n[bold]数据集:[/bold] {', '.join(result['datasets'])}")
    if result.get("results"):
        console.print(f"\n[bold]实验结果:[/bold] {result['results']}")
    console.print(f"\n[bold]局限性:[/bold]")
    for l in result.get("limitations", []):
        console.print(f"  • {l}")
    console.print(f"\n[bold]未来方向:[/bold] {result.get('future_work', '')}")


@app.command()
def research(
    topic: str = typer.Argument(..., help="研究主题"),
    max_steps: int = typer.Option(30, "--max-steps", "-s", help="最大决策步数"),
    max_tokens: int = typer.Option(200000, "--max-tokens", "-t", help="总 token 上限"),
    legacy: bool = typer.Option(False, "--legacy", help="使用老的编排式流程（降级）"),
):
    """对主题进行深度研究并生成报告（默认自主模式）"""
    from src.agents.manager import ResearchManager

    root = Path(__file__).parent
    settings = Settings.from_env(root)

    if legacy:
        # 降级：走老的编排式 Orchestrator
        console.print(Panel(f"[bold]深度研究 (编排模式):[/bold] {topic}", style="yellow"))
        orch = get_orchestrator()
        with console.status("研究中，请稍候..."):
            result = orch.run_deep_research(topic)
        console.print(f"\n[green]报告已保存至: {result.report_file}[/green]")
        stats = orch.memory_stats()
        console.print(f"[dim]记忆库: {stats['episodes']} 情节 | {stats['skills']} 技能 | {stats['vectors']} 向量[/dim]")
        return

    # 默认：自主模式
    console.print(Panel(
        f"[bold cyan]自主研究模式[/bold cyan]\n"
        f"主题: {topic}\n"
        f"预算: {max_steps} 步 / {max_tokens:,} tokens",
        border_style="cyan",
    ))

    manager = ResearchManager(settings, max_steps=max_steps, max_total_tokens=max_tokens)
    console.print("[dim]Agent 开始自主研究...[/dim]\n")
    result = manager.run(topic)

    # 结果展示
    console.print(f"\n{'─' * 60}")
    if result.finished:
        console.print(f"[green]研究完成[/green] ({result.finish_reason})")
        output = result.final_output
        if isinstance(output, dict):
            report = output.get("output", str(output))
        else:
            report = str(output)

        # 保存报告到 workspace/reports/
        from datetime import datetime
        reports_dir = settings.workspace_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)[:40].strip()
        report_path = reports_dir / f"{timestamp}_{safe_topic}.md"
        report_path.write_text(report, encoding="utf-8")

        console.print(f"[green]报告已保存至: {report_path}[/green]")
        console.print(Panel(report[:3000], title="最终报告（前 3000 字）", border_style="green"))
    else:
        console.print(f"[yellow]研究未完成[/yellow] (原因: {result.finish_reason})")
        if result.steps:
            last = result.steps[-1]
            console.print(f"[dim]最后一步: {last.tool_name} → {last.tool_result.error if last.tool_result and not last.tool_result.success else 'ok'}[/dim]")

    # 统计
    console.print(f"\n[dim]总步数: {len(result.steps)} | 总 tokens: {result.total_tokens:,} | 耗时: {result.total_elapsed_ms/1000:.1f}s[/dim]")

    # 执行轨迹
    if result.steps:
        console.print("\n[bold]执行轨迹:[/bold]")
        for s in result.steps:
            status = "✓" if (s.tool_result and s.tool_result.success) else "✗"
            name = s.tool_name or "(thinking)"
            console.print(f"  {s.step_idx+1}. [{status}] {name}  ({s.tokens_used} tok, {s.elapsed_ms}ms)")


@app.command()
def chat():
    """对话式交互：用自然语言描述你的想法，Agent 自动决定做什么。"""
    from rich.prompt import Prompt
    from src.agents.conversational_agent import ConversationalAgent
    from src.core.llm import LLMClient

    orch = get_orchestrator()
    agent = ConversationalAgent(LLMClient(orch.settings), orch.memory)

    console.print(Panel.fit(
        "[bold cyan]Deep Research Agent - 对话模式[/bold cyan]\n"
        "直接用自然语言描述你的研究想法或问题。\n"
        "命令：[dim]/quit 退出 | /clear 清空历史 | /stats 查看记忆统计[/dim]",
        border_style="cyan",
    ))

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]你[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]再见[/yellow]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit"):
            console.print("[yellow]再见[/yellow]")
            break
        if user_input.lower() == "/clear":
            agent.clear()
            console.print("[dim]对话历史已清空[/dim]")
            continue
        if user_input.lower() == "/stats":
            stats = orch.memory_stats()
            console.print(f"[dim]情节: {stats['episodes']} | 技能: {stats['skills']} | 向量: {stats['vectors']}[/dim]")
            continue

        agent.add_user_message(user_input)

        # 1. 路由决策
        with console.status("[cyan]思考中...[/cyan]"):
            action = agent.decide()

        console.print(f"[dim]→ 意图: {action.raw_intent}  |  动作: {action.action}[/dim]")
        if action.queries:
            console.print(f"[dim]→ 检索词: {' / '.join(action.queries)}[/dim]")

        # 2. 根据动作调用后端
        try:
            reply = _dispatch(orch, agent, action)
        except Exception as e:
            reply = f"[red]执行出错: {type(e).__name__}: {e}[/red]"

        agent.add_assistant_message(reply)
        console.print(Panel(reply, title="[bold magenta]Agent[/bold magenta]", border_style="magenta"))


def _dispatch(orch, agent, action) -> str:
    """把路由决策翻译成具体工具调用，并让Agent用自然语言总结结果。"""
    a = action.action

    if a in ("ask_user", "chitchat"):
        return action.reply or "我在听，请继续说。"

    if a == "memory_query":
        stats = orch.memory_stats()
        ctx = orch.memory.format_context_for_prompt(action.topic or "research")
        agent.add_tool_result(f"memory stats: {stats}\n\ncontext:\n{ctx[:1500]}")
        return agent.summarize_result(action, {"stats": stats, "context": ctx[:2000]})

    if a == "evaluate":
        topic = action.topic or (action.queries[0] if action.queries else "")
        result = orch.evaluate_direction(topic, queries=action.queries or None)
        brief = {
            "feasibility": result.get("feasibility"),
            "novelty": result.get("novelty"),
            "impact": result.get("impact"),
            "analysis": result.get("analysis", "")[:1500],
            "recommendations": result.get("recommendations", []),
            "benchmarks": result.get("benchmarks", []),
            "paper_count": len(result.get("papers", [])),
            "sample_papers": [
                {"title": p.title, "url": p.url}
                for p in result.get("papers", [])[:5]
            ],
        }
        agent.add_tool_result(f"evaluate done. {brief}")
        return agent.summarize_result(action, brief)

    if a == "search":
        papers = orch.search_papers_multi(action.queries or [action.topic], per_query=4)
        brief = {
            "count": len(papers),
            "papers": [
                {"title": p.title, "authors": p.authors[:3],
                 "abstract": (p.abstract or "")[:200], "url": p.url}
                for p in papers[:8]
            ],
        }
        agent.add_tool_result(f"search done. got {len(papers)} papers.")
        return agent.summarize_result(action, brief)

    if a == "analyze":
        from src.core.models import PaperItem
        pid = action.paper_id.strip().split("/")[-1]
        if not pid:
            return "要分析论文的话，请给我 arXiv ID 或链接。"
        papers = orch.arxiv.search(f"id:{pid}", max_results=1)
        if not papers:
            return f"没找到 arXiv ID 为 {pid} 的论文，确认一下？"
        result = orch.analyze_paper(papers[0], action.focus or None)
        result["_title"] = papers[0].title
        agent.add_tool_result(f"analyze done for {papers[0].title}")
        return agent.summarize_result(action, result)

    if a == "research":
        topic = action.topic or (action.queries[0] if action.queries else "")
        console.print("[yellow]深度研究通常需要几分钟，请稍候...[/yellow]")
        result = orch.run_deep_research(topic)
        brief = {
            "topic": result.topic,
            "report_file": result.report_file,
            "plan_size": len(result.plan),
            "paper_count": len(result.papers),
            "critic_score": result.critic_reviews[-1].score if result.critic_reviews else None,
            "report_preview": result.final_report_markdown[:1500],
        }
        agent.add_tool_result(f"research done. report at {result.report_file}")
        return agent.summarize_result(action, brief)

    return action.reply or "我暂时不确定该怎么帮你，能再说得具体一点吗？"


if __name__ == "__main__":
    app()

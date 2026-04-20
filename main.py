#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

from src.config import Settings
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
    from src.models import PaperItem
    orch = get_orchestrator()

    paper_id = url.split("/")[-1]
    with console.status("分析论文中..."):
        papers = orch.arxiv.search(f"id:{paper_id}", max_results=1)
        if not papers:
            console.print("[red]未找到该论文[/red]")
            raise typer.Exit(1)
        result = orch.analyze_paper(papers[0], focus)

    console.print(Panel(f"[bold]{papers[0].title}[/bold]"))
    console.print(f"\n[bold]核心问题:[/bold] {result.get('problem', '')}")
    console.print(f"\n[bold]主要贡献:[/bold]")
    for c in result.get("contributions", []):
        console.print(f"  • {c}")
    console.print(f"\n[bold]方法:[/bold]")
    for m in result.get("methods", []):
        console.print(f"  • {m}")
    console.print(f"\n[bold]局限性:[/bold]")
    for l in result.get("limitations", []):
        console.print(f"  • {l}")
    console.print(f"\n[bold]未来方向:[/bold] {result.get('future_work', '')}")


@app.command()
def research(topic: str = typer.Argument(..., help="研究主题")):
    """对主题进行深度研究并生成报告"""
    console.print(Panel(f"[bold]深度研究:[/bold] {topic}", style="green"))
    orch = get_orchestrator()

    with console.status("研究中，请稍候..."):
        result = orch.run_deep_research(topic)

    console.print(f"\n[green]报告已保存至: {result.report_file}[/green]")
    console.print(f"共分析 {len(result.papers)} 篇论文，完成 {len(result.task_results)} 个研究任务")


if __name__ == "__main__":
    app()

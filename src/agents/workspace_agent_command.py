"""CLI registration for the stateful workspace agent."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from ..core.config import Settings
from .workspace_agent import WorkspaceAgent


def register_workspace_agent_command(app: typer.Typer, console, project_root: Path) -> None:
    @app.command()
    def agent(
        allow_write: bool = typer.Option(
            False,
            "--allow-write",
            help="启用项目源码编辑工具；研究笔记/报告/记忆始终允许写入 workspace",
        ),
        max_steps: int = typer.Option(12, "--max-steps", "-s", help="每轮最多连续 tool-calling 步数"),
    ):
        """启动真正的常驻 workspace agent。"""
        settings = Settings.from_env(project_root)
        workspace_agent = WorkspaceAgent(
            settings=settings,
            project_root=project_root,
            allow_write=allow_write,
            max_tool_steps_per_turn=max_steps,
        )
        _print_startup(console, workspace_agent, project_root, allow_write, max_steps)

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]你[/bold green]").strip()
            except (KeyboardInterrupt, EOFError):
                workspace_agent.save_session()
                console.print("\n[yellow]再见[/yellow]")
                break

            if not user_input:
                continue
            command, _, arg = user_input.partition(" ")
            command = command.lower()
            arg = arg.strip()

            if command in ("/quit", "/exit"):
                workspace_agent.save_session()
                console.print("[yellow]再见[/yellow]")
                break
            if command == "/clear":
                workspace_agent.reset()
                console.print(f"[dim]已开启新会话: {workspace_agent.session_id}[/dim]")
                continue
            if command == "/help":
                _print_help(console)
                continue
            if command == "/tools":
                _print_tools(console, workspace_agent)
                continue
            if command == "/status":
                _print_status(console, workspace_agent, project_root, allow_write, max_steps)
                continue
            if command == "/memory":
                _print_memory(console, workspace_agent)
                continue
            if command == "/sessions":
                _print_sessions(console, workspace_agent)
                continue
            if command == "/resume":
                _resume_session(console, workspace_agent, arg)
                continue
            if command == "/recall":
                _recall_memory(console, workspace_agent, arg)
                continue

            try:
                with console.status("[cyan]Agent 正在调用工具和整理上下文...[/cyan]"):
                    result = workspace_agent.handle(user_input)
            except Exception as exc:
                console.print(f"[red]执行出错: {type(exc).__name__}: {exc}[/red]")
                continue

            if result.tool_calls:
                console.print("[dim]工具轨迹:[/dim]")
                for item in result.tool_calls:
                    status = "OK" if item.get("success") else "ERR"
                    suffix = f" - {item.get('error')}" if item.get("error") else ""
                    console.print(f"  [dim]{item['step']}. {status} {item['tool']}{suffix}[/dim]")

            console.print(
                Panel(
                    result.reply,
                    title=(
                        "[bold magenta]Agent[/bold magenta] "
                        f"[dim]{result.tokens_used} tok, {result.elapsed_ms}ms[/dim]"
                    ),
                    border_style="magenta",
                )
            )


def _print_startup(console, agent: WorkspaceAgent, project_root: Path, allow_write: bool, max_steps: int) -> None:
    stats = agent.memory.stats()
    mode = "project-write-enabled" if allow_write else "project-read-only"
    body = (
        "[bold cyan]Deep Research Workspace Agent[/bold cyan]\n"
        "Persistent tool-using agent for this project.\n\n"
        f"[bold]Project[/bold]  {project_root}\n"
        f"[bold]Session[/bold]  {agent.session_id}\n"
        f"[bold]Mode[/bold]     {mode}\n"
        "[bold]Outputs[/bold]  research notes/reports/memory writes enabled\n"
        f"[bold]Budget[/bold]   {max_steps} tool-calling steps per turn\n\n"
        f"[bold]Memory[/bold]   {stats.get('episodes', 0)} episodes | "
        f"{stats.get('skills', 0)} skills | {stats.get('vectors', 0)} vectors | "
        f"{stats.get('paper_nodes', 0)} paper nodes\n\n"
        "[dim]/help | /resume latest | /sessions | /recall [query] | /quit[/dim]"
    )
    console.print(Panel.fit(body, border_style="cyan"))


def _print_help(console) -> None:
    table = Table(title="Slash Commands")
    table.add_column("Command", style="cyan")
    table.add_column("What it does")
    table.add_row("/resume", "List saved conversation sessions.")
    table.add_row("/resume latest", "Restore the latest saved conversation transcript.")
    table.add_row("/resume <id>", "Restore a saved conversation by full or prefix session id.")
    table.add_row("/sessions", "Alias for listing saved sessions.")
    table.add_row("/recall [query]", "Inject multi-layer long-term memory for a query into this session.")
    table.add_row("/memory", "Show memory database stats and recent research episodes.")
    table.add_row("/tools", "List available tools.")
    table.add_row("/status", "Show current session, project, mode, history length, and budgets.")
    table.add_row("/clear", "Start a new live conversation session.")
    table.add_row("/quit", "Save and exit the workspace agent.")
    console.print(table)


def _print_sessions(console, agent: WorkspaceAgent) -> None:
    sessions = agent.list_sessions(limit=12)
    table = Table(title="Saved Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Updated")
    table.add_column("Messages")
    table.add_column("Title")
    for item in sessions:
        table.add_row(
            item["session_id"],
            item.get("updated_at", ""),
            str(item.get("message_count", 0)),
            item.get("title", "")[:80],
        )
    console.print(table)


def _resume_session(console, agent: WorkspaceAgent, session_id: str) -> None:
    if not session_id:
        _print_sessions(console, agent)
        console.print("[dim]使用 /resume latest 或 /resume <id> 恢复某个会话。[/dim]")
        return
    try:
        info = agent.load_session(session_id)
    except Exception as exc:
        console.print(f"[red]恢复失败: {type(exc).__name__}: {exc}[/red]")
        return
    console.print(
        Panel.fit(
            f"[green]已恢复会话[/green]\n"
            f"ID: {info['session_id']}\n"
            f"标题: {info['title']}\n"
            f"消息数: {info['message_count']}\n"
            f"上次更新: {info.get('updated_at', '')}",
            border_style="green",
        )
    )


def _print_tools(console, agent: WorkspaceAgent) -> None:
    names = agent.tools.names()
    table = Table(title="Available Tools")
    table.add_column("Group", style="cyan")
    table.add_column("Tools")
    groups = {
        "project": [n for n in names if "project" in n or n == "replace_in_project_file"],
        "memory": [n for n in names if "memory" in n or "note" in n or "episode" in n],
        "paper": [n for n in names if "paper" in n or "arxiv" in n or "semantic" in n],
        "web": [n for n in names if n.startswith("web_")],
    }
    for group, group_names in groups.items():
        table.add_row(group, ", ".join(group_names) or "-")
    console.print(table)


def _print_status(console, agent: WorkspaceAgent, project_root: Path, allow_write: bool, max_steps: int) -> None:
    stats = agent.memory.stats()
    mode = "project-write-enabled" if allow_write else "project-read-only"
    console.print(
        Panel.fit(
            f"[bold]Project[/bold] {project_root}\n"
            f"[bold]Session[/bold] {agent.session_id}\n"
            f"[bold]Title[/bold] {agent.session_title}\n"
            f"[bold]Mode[/bold] {mode}\n"
            "[bold]Research outputs[/bold] enabled\n"
            f"[bold]History messages[/bold] {len(agent.messages)} / {agent.max_history_messages}\n"
            f"[bold]Tool step budget[/bold] {max_steps} per turn\n"
            f"[bold]Memory[/bold] {stats}",
            border_style="blue",
        )
    )


def _print_memory(console, agent: WorkspaceAgent) -> None:
    stats = agent.memory.stats()
    recent = agent.memory.get_recent_episodes(limit=5)
    table = Table(title="Memory")
    table.add_column("Layer", style="cyan")
    table.add_column("Count")
    for key in ("episodes", "skills", "vectors", "paper_nodes", "paper_edges"):
        table.add_row(key, str(stats.get(key, 0)))
    console.print(table)
    if recent:
        recent_table = Table(title="Recent Episodes")
        recent_table.add_column("ID", style="cyan")
        recent_table.add_column("Topic")
        recent_table.add_column("Quality")
        for ep in recent:
            recent_table.add_row(ep.id, ep.topic[:60], f"{ep.quality_score:.2f}")
        console.print(recent_table)


def _recall_memory(console, agent: WorkspaceAgent, query: str) -> None:
    topic = query or "research history"
    stats = agent.memory.stats()
    context = agent.memory.format_context_for_prompt(topic)
    recent = agent.memory.get_recent_episodes(limit=5)
    graph = agent.memory.get_paper_graph_context(topic, top_k=5)

    recent_block = "\n".join(
        f"- {ep.topic} (quality={ep.quality_score:.2f}, id={ep.id}): {ep.insights[:240]}"
        for ep in recent
    ) or "(no recent episodes)"
    graph_block = "\n".join(
        f"- {paper.get('title', '')}: {paper.get('tldr', '') or paper.get('problem', '')}"
        for paper in graph.get("papers", [])
    ) or "(no confident paper graph context)"

    memory_block = (
        "[RECALLED LONG-TERM MEMORY]\n"
        f"Query: {topic}\n"
        f"Stats: {stats}\n\n"
        "Recent episodes:\n"
        f"{recent_block}\n\n"
        "Paper graph context:\n"
        f"{graph_block}\n\n"
        "Retrieved multi-layer context:\n"
        f"{context[:5000] if context else '(no matching multi-layer context)'}"
    )
    agent.messages.append({"role": "system", "content": memory_block})
    agent._trim_history()
    agent.save_session()
    console.print(
        Panel.fit(
            f"[green]已召回长期记忆[/green]\n"
            f"查询: {topic}\n"
            f"注入上下文字符数: {len(memory_block)}\n"
            "来源: episodic / skill / vector / paper graph",
            border_style="green",
        )
    )

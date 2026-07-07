"""CLI registration for the stateful workspace agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import typer
from rich import box
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from ..core.config import Settings
from .workspace_agent import WorkspaceAgent, WorkspaceTurnResult


def register_workspace_agent_command(app: typer.Typer, console, project_root: Path) -> None:
    @app.command()
    def agent(
        allow_write: bool = typer.Option(
            False,
            "--allow-write",
            help="启用项目源码编辑工具；研究笔记/报告/记忆始终允许写入 workspace",
        ),
        max_steps: int = typer.Option(12, "--max-steps", "-s", help="每轮最多连续 tool-calling 步数"),
        clear_screen: bool = typer.Option(
            True,
            "--clear-screen/--no-clear-screen",
            help="启动 CLI agent 时先清屏并重绘启动页",
        ),
    ):
        """启动真正的常驻 workspace agent。"""
        settings = Settings.from_env(project_root)
        workspace_agent = WorkspaceAgent(
            settings=settings,
            project_root=project_root,
            allow_write=allow_write,
            max_tool_steps_per_turn=max_steps,
        )
        last_result: WorkspaceTurnResult | None = None
        if clear_screen:
            console.clear()
        _print_startup(console, workspace_agent, settings, project_root, allow_write, max_steps)

        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]deep-research[/bold cyan]").strip()
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
                last_result = None
                console.clear()
                _print_startup(console, workspace_agent, settings, project_root, allow_write, max_steps)
                console.print(f"[dim]已开启新会话: {workspace_agent.session_id}[/dim]")
                continue
            if command in ("/cls", "/clear-screen"):
                console.clear()
                _print_startup(console, workspace_agent, settings, project_root, allow_write, max_steps)
                continue
            if command == "/help":
                _print_help(console)
                continue
            if command == "/tools":
                _print_tools(console, workspace_agent)
                continue
            if command == "/status":
                _print_status(console, workspace_agent, settings, project_root, allow_write, max_steps)
                continue
            if command == "/memory":
                _print_memory(console, workspace_agent)
                continue
            if command == "/sessions":
                _print_sessions(console, workspace_agent)
                continue
            if command == "/resume":
                restored_result = _resume_session(console, workspace_agent, arg)
                if restored_result is not None:
                    last_result = restored_result
                continue
            if command == "/recall":
                _recall_memory(console, workspace_agent, arg)
                continue
            if command == "/trace":
                _print_trace(console, last_result)
                continue

            _print_working_header(console, user_input, workspace_agent)
            try:
                with console.status("[cyan]Thinking and using tools...[/cyan]"):
                    result = workspace_agent.handle(user_input)
            except Exception as exc:
                console.print(f"[red]执行出错: {type(exc).__name__}: {exc}[/red]")
                continue

            last_result = result
            _print_compact_trace(console, result)
            _print_answer(console, result)


def _print_startup(
    console,
    agent: WorkspaceAgent,
    settings: Settings,
    project_root: Path,
    allow_write: bool,
    max_steps: int,
) -> None:
    stats = agent.memory.stats()
    mode = "project-write-enabled" if allow_write else "project-read-only"
    body = (
        "[bold cyan]Deep Research Workspace Agent[/bold cyan]\n"
        "Persistent tool-using agent for this project.\n\n"
        f"[bold]cwd[/bold]       {project_root}\n"
        f"[bold]session[/bold]   {agent.session_id}\n"
        f"[bold]model[/bold]     {settings.llm_model}\n"
        f"[bold]mode[/bold]      {mode}\n"
        "[bold]outputs[/bold]   research notes/reports/memory writes enabled\n"
        f"[bold]budget[/bold]    {max_steps} tool calls per turn\n\n"
        f"[bold]memory[/bold]    {stats.get('episodes', 0)} episodes | "
        f"{stats.get('skills', 0)} skills | {stats.get('vectors', 0)} vectors | "
        f"{stats.get('paper_nodes', 0)} paper nodes\n\n"
        "[dim]/help  /resume latest  /sessions  /recall [query]  /trace  /quit[/dim]"
    )
    console.print(Panel.fit(body, title="Deep Research", border_style="cyan", box=box.ROUNDED))


def _print_working_header(console, user_input: str, agent: WorkspaceAgent) -> None:
    preview = user_input if len(user_input) <= 120 else user_input[:117] + "..."
    console.print(
        Panel.fit(
            f"[bold]request[/bold] {preview}\n"
            f"[dim]session {agent.session_id} | trace collapsed by default; use /trace to expand[/dim]",
            title="Working",
            border_style="blue",
            box=box.ROUNDED,
        )
    )


def _print_compact_trace(console, result: WorkspaceTurnResult) -> None:
    if not result.tool_calls:
        console.print("[dim]No tools called.[/dim]")
        return
    table = Table(title="Tool Calls (collapsed)", box=box.SIMPLE_HEAVY)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Tool")
    table.add_column("Status")
    table.add_column("Args Preview")
    for idx, item in enumerate(result.tool_calls, 1):
        status = "ok" if item.get("success") else "error"
        style = "green" if item.get("success") else "red"
        table.add_row(
            str(idx),
            str(item.get("tool", "")),
            f"[{style}]{status}[/{style}]",
            _format_args_preview(item.get("args") or {}),
        )
    console.print(table)
    console.print("[dim]Use /trace to expand the last turn's tool details.[/dim]")


def _print_answer(console, result: WorkspaceTurnResult) -> None:
    title = f"Agent  {result.tokens_used} tok | {result.elapsed_ms}ms | {result.stopped_reason}"
    console.print(Panel(result.reply, title=title, border_style="magenta", box=box.ROUNDED))


def _print_trace(console, result: WorkspaceTurnResult | None) -> None:
    if result is None:
        console.print("[dim]No previous turn trace.[/dim]")
        return
    if not result.tool_calls:
        console.print("[dim]Previous turn did not call tools.[/dim]")
        return
    for idx, item in enumerate(result.tool_calls, 1):
        status = "ok" if item.get("success") else "error"
        args = json.dumps(item.get("args") or {}, ensure_ascii=False, indent=2)
        body = f"status: {status}\nargs:\n{args}"
        if item.get("error"):
            body += f"\nerror: {item['error']}"
        console.print(Panel(body, title=f"Tool {idx}: {item.get('tool', '')}", border_style="cyan"))


def _print_help(console) -> None:
    table = Table(title="Slash Commands", box=box.SIMPLE_HEAVY)
    table.add_column("Command", style="cyan")
    table.add_column("What it does")
    table.add_row("/resume", "List saved conversation sessions.")
    table.add_row("/resume latest", "Restore and redraw the latest saved conversation transcript.")
    table.add_row("/resume <id>", "Restore and redraw a saved conversation by full or prefix session id.")
    table.add_row("/sessions", "Alias for listing saved sessions.")
    table.add_row("/recall [query]", "Inject multi-layer long-term memory into this session.")
    table.add_row("/trace", "Expand the previous turn's collapsed tool trace.")
    table.add_row("/memory", "Show memory database stats and recent research episodes.")
    table.add_row("/tools", "List available tools.")
    table.add_row("/status", "Show current session, project, mode, history length, and budgets.")
    table.add_row("/cls", "Clear the terminal and redraw the startup page without resetting session.")
    table.add_row("/clear", "Start a new live conversation session and redraw the page.")
    table.add_row("/quit", "Save and exit the workspace agent.")
    console.print(table)


def _print_sessions(console, agent: WorkspaceAgent) -> None:
    sessions = agent.list_sessions(limit=12)
    table = Table(title="Saved Sessions", box=box.SIMPLE_HEAVY)
    table.add_column("ID", style="cyan")
    table.add_column("Updated")
    table.add_column("Messages", justify="right")
    table.add_column("Title")
    for item in sessions:
        table.add_row(
            item["session_id"],
            item.get("updated_at", ""),
            str(item.get("message_count", 0)),
            item.get("title", "")[:80],
        )
    console.print(table)


def _resume_session(console, agent: WorkspaceAgent, session_id: str) -> WorkspaceTurnResult | None:
    if not session_id:
        _print_sessions(console, agent)
        console.print("[dim]Use /resume latest or /resume <id> to restore and redraw a session.[/dim]")
        return None
    try:
        info = agent.load_session(session_id)
    except Exception as exc:
        console.print(f"[red]恢复失败: {type(exc).__name__}: {exc}[/red]")
        return None

    summary = (
        "[green]已恢复会话[/green]\n"
        f"ID: {info['session_id']}\n"
        f"标题: {info['title']}\n"
        f"消息数: {info['message_count']}\n"
        f"上次更新: {info.get('updated_at', '')}\n"
        "[dim]下面是恢复后的会话内容；继续输入即可接着聊。[/dim]"
    )
    console.print(Panel.fit(summary, border_style="green", box=box.ROUNDED))
    _print_restored_transcript(console, agent)
    return _rebuild_last_turn_result(_session_transcript(agent))



def _session_transcript(agent: WorkspaceAgent) -> List[Dict[str, Any]]:
    return list(getattr(agent, "transcript_messages", agent.messages))


def _print_restored_transcript(console, agent: WorkspaceAgent) -> None:
    messages = [msg for msg in _session_transcript(agent)[1:] if msg.get("role") != "system"]
    if not messages:
        console.print("[dim]恢复的会话没有可显示内容。[/dim]")
        return

    console.rule("[bold cyan]Restored Conversation[/bold cyan]")
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            console.print(
                Panel(
                    _clip_text(str(msg.get("content") or ""), 5000),
                    title="You",
                    border_style="green",
                    box=box.ROUNDED,
                )
            )
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            content = str(msg.get("content") or "").strip()
            if tool_calls:
                _print_restored_tool_requests(console, tool_calls)
            if content:
                console.print(
                    Panel(
                        _clip_text(content, 7000),
                        title="Agent",
                        border_style="magenta",
                        box=box.ROUNDED,
                    )
                )
            continue

        if role == "tool":
            tool_call_id = str(msg.get("tool_call_id") or "")[:18]
            content = _clip_text(str(msg.get("content") or ""), 1200)
            console.print(
                Panel(
                    content,
                    title=f"Tool observation {tool_call_id}" if tool_call_id else "Tool observation",
                    border_style="cyan",
                    box=box.ROUNDED,
                )
            )
    console.rule("[dim]session restored[/dim]")


def _print_restored_tool_requests(console, tool_calls: List[Dict[str, Any]]) -> None:
    table = Table(title="Tool Calls (restored)", box=box.SIMPLE_HEAVY)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Tool")
    table.add_column("Args Preview")
    for idx, call in enumerate(tool_calls, 1):
        fn = (call.get("function") or {}) if isinstance(call, dict) else {}
        name = str(fn.get("name") or "")
        args = _parse_json_object(fn.get("arguments") or "{}")
        table.add_row(str(idx), name, _format_args_preview(args))
    console.print(table)


def _rebuild_last_turn_result(messages: List[Dict[str, Any]]) -> WorkspaceTurnResult:
    tool_calls: List[Dict[str, Any]] = []
    reply = "已恢复会话。"
    current_turn_tools: List[Dict[str, Any]] = []
    tool_index_by_id: Dict[str, int] = {}

    for msg in messages[1:]:
        role = msg.get("role")
        if role == "user":
            current_turn_tools = []
            tool_index_by_id = {}
            reply = "已恢复会话。"
            continue

        if role == "assistant":
            restored_calls = msg.get("tool_calls") or []
            if restored_calls:
                current_turn_tools = []
                tool_index_by_id = {}
                for restored_call in restored_calls:
                    fn = restored_call.get("function") or {}
                    args = _parse_json_object(fn.get("arguments") or "{}")
                    item = {
                        "step": len(current_turn_tools) + 1,
                        "tool": str(fn.get("name") or ""),
                        "args": args,
                        "success": True,
                        "error": "",
                    }
                    tool_index_by_id[str(restored_call.get("id") or "")] = len(current_turn_tools)
                    current_turn_tools.append(item)
                tool_calls = current_turn_tools
                continue
            content = str(msg.get("content") or "").strip()
            if content:
                reply = content
            continue

        if role == "tool" and current_turn_tools:
            call_id = str(msg.get("tool_call_id") or "")
            content = str(msg.get("content") or "")
            idx = tool_index_by_id.get(call_id)
            if idx is not None:
                current_turn_tools[idx]["success"] = not content.startswith("[Tool failed]")
                current_turn_tools[idx]["error"] = (
                    content.replace("[Tool failed]", "", 1).strip()
                    if content.startswith("[Tool failed]")
                    else ""
                )

    return WorkspaceTurnResult(reply=reply, tool_calls=tool_calls, stopped_reason="restored")


def _parse_json_object(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated, total {len(text)} chars]"


def _print_tools(console, agent: WorkspaceAgent) -> None:
    names = agent.tools.names()
    table = Table(title="Available Tools", box=box.SIMPLE_HEAVY)
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


def _print_status(
    console,
    agent: WorkspaceAgent,
    settings: Settings,
    project_root: Path,
    allow_write: bool,
    max_steps: int,
) -> None:
    stats = agent.memory.stats()
    mode = "project-write-enabled" if allow_write else "project-read-only"
    console.print(
        Panel.fit(
            f"[bold]project[/bold] {project_root}\n"
            f"[bold]session[/bold] {agent.session_id}\n"
            f"[bold]title[/bold] {agent.session_title}\n"
            f"[bold]model[/bold] {settings.llm_model}\n"
            f"[bold]mode[/bold] {mode}\n"
            "[bold]research outputs[/bold] enabled\n"
            f"[bold]history[/bold] {len(agent.messages)} / {agent.max_history_messages} messages\n"
            f"[bold]budget[/bold] {max_steps} tool calls per turn\n"
            f"[bold]memory[/bold] {stats}",
            border_style="blue",
            box=box.ROUNDED,
        )
    )


def _print_memory(console, agent: WorkspaceAgent) -> None:
    stats = agent.memory.stats()
    recent = agent.memory.get_recent_episodes(limit=5)
    table = Table(title="Memory", box=box.SIMPLE_HEAVY)
    table.add_column("Layer", style="cyan")
    table.add_column("Count", justify="right")
    for key in ("episodes", "skills", "vectors", "paper_nodes", "paper_edges"):
        table.add_row(key, str(stats.get(key, 0)))
    console.print(table)
    if recent:
        recent_table = Table(title="Recent Episodes", box=box.SIMPLE_HEAVY)
        recent_table.add_column("ID", style="cyan")
        recent_table.add_column("Topic")
        recent_table.add_column("Quality", justify="right")
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
    memory_message = {"role": "system", "content": memory_block}
    agent.messages.append(memory_message)
    if hasattr(agent, "transcript_messages"):
        agent.transcript_messages.append(memory_message)
    agent._trim_history()
    agent.save_session()
    console.print(
        Panel.fit(
            f"[green]已召回长期记忆[/green]\n"
            f"查询: {topic}\n"
            f"注入上下文字符数: {len(memory_block)}\n"
            "来源: episodic / skill / vector / paper graph",
            border_style="green",
            box=box.ROUNDED,
        )
    )


def _format_args_preview(args: Dict[str, Any]) -> str:
    if not args:
        return "-"
    parts: List[str] = []
    for key, value in args.items():
        text = str(value).replace("\n", " ")
        if len(text) > 60:
            text = text[:57] + "..."
        parts.append(f"{key}={text}")
    joined = "; ".join(parts)
    return joined if len(joined) <= 120 else joined[:117] + "..."

"""
Stateful workspace agent.

Unlike ConversationalAgent, this is not a one-shot router. It keeps a running
conversation, can call multiple tools per user turn, and can inspect the current
project before deciding how to answer.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from ..core.config import Settings
from ..core.llm import LLMClient
from ..learning.reflection import ReflectionEngine
from ..memory.memory_manager import MemoryManager
from ..services.paper_fetcher import PaperFetcher
from ..services.paper_search import ArxivSearcher, SemanticScholarSearcher
from ..services.web_searcher import WebSearcher
from ..tools.memory_tools import (
    build_retrieve_memory_tool,
    build_save_note_tool,
    build_save_research_episode_tool,
)
from ..tools.paper_tools import build_analyze_paper_tool, build_fetch_fulltext_tool
from ..tools.project_tools import (
    build_list_project_files_tool,
    build_project_overview_tool,
    build_read_project_file_tool,
    build_replace_in_project_file_tool,
    build_search_project_tool,
)
from ..tools.search_tools import build_arxiv_search_tool, build_s2_search_tool
from ..tools.tool import ToolRegistry
from ..tools.web_tools import build_web_fetch_tool, build_web_search_tool


WORKSPACE_AGENT_SYSTEM_PROMPT = """You are Deep Research Workspace Agent, a persistent tool-using agent inside the user's project.

You are closer to Claude Code than to a one-shot chatbot:
- You keep conversation context across turns.
- You inspect project files before making claims about the codebase.
- You can call multiple tools in one user turn, observe results, then continue.
- You can combine project context, long-term memory, paper search, web search, and paper reading.

Core behavior:
1. For project/code questions, use project_overview/search_project/read_project_file as needed.
2. For research questions, first retrieve_memory, then search academic/web sources if needed.
3. Be concrete. Cite project files and commands when relevant.
4. Do not claim you changed files unless a write tool succeeded.
5. If project file write tools are unavailable and the user asks to edit source files, explain that project editing is disabled and ask them to restart with --allow-write.
6. Keep final answers concise and in Chinese unless the user asks otherwise.

Safety and scope:
- Project tools are scoped to the project root.
- Research outputs are allowed: paper notes, reports, vector memory, episodic memory, skill memory, and paper graph memory may be written by research tools.
- Prefer exact, small reads over loading huge files.
- Do not invent tool results. If you have not inspected something, say so or inspect it.
"""


@dataclass
class WorkspaceTurnResult:
    reply: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    elapsed_ms: int = 0
    stopped_reason: str = "answered"


class WorkspaceAgent:
    """Persistent multi-turn agent with project and research tools."""

    def __init__(
        self,
        settings: Settings,
        project_root: Path,
        allow_write: bool = False,
        max_tool_steps_per_turn: int = 12,
        max_history_messages: int = 60,
        temperature: float = 0.2,
    ):
        self.settings = settings
        self.project_root = Path(project_root).resolve()
        self.allow_write = allow_write
        self.max_tool_steps_per_turn = max_tool_steps_per_turn
        self.max_history_messages = max_history_messages
        self.temperature = temperature

        self.llm = LLMClient(settings)
        self.memory = MemoryManager(settings)
        self.reflection = ReflectionEngine(LLMClient(settings), self.memory)
        self.arxiv = ArxivSearcher(max_results=settings.search_top_k)
        self.s2 = SemanticScholarSearcher()
        self.fetcher = PaperFetcher(settings.workspace_dir / "pdf_cache")
        self.web = WebSearcher()

        self.session_dir = settings.workspace_dir / "agent_sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = self._new_session_id()
        self.session_title = "new session"
        self.session_path = self._session_path(self.session_id)

        self.tools = self._build_tools()
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
        ]
        self.transcript_messages: List[Dict[str, Any]] = list(self.messages)

    def _system_prompt(self) -> str:
        mode = "project-write-enabled" if self.allow_write else "project-read-only"
        return (
            WORKSPACE_AGENT_SYSTEM_PROMPT
            + f"\n\nProject root: {self.project_root}\n"
            + f"Project filesystem mode: {mode}\n"
            + "Research output mode: enabled (workspace notes/reports/memory may be written by research tools)\n"
        )

    def _build_tools(self) -> ToolRegistry:
        registry = ToolRegistry()

        registry.register(build_project_overview_tool(self.project_root))
        registry.register(build_list_project_files_tool(self.project_root))
        registry.register(build_read_project_file_tool(self.project_root))
        registry.register(build_search_project_tool(self.project_root))
        if self.allow_write:
            registry.register(build_replace_in_project_file_tool(self.project_root))

        registry.register(build_retrieve_memory_tool(self.memory))
        registry.register(build_save_note_tool(self.memory))
        registry.register(build_save_research_episode_tool(self.memory, self.reflection))

        analyzer = self._build_paper_analyzer()
        registry.register(build_arxiv_search_tool(self.arxiv))
        registry.register(build_s2_search_tool(self.s2))
        registry.register(build_fetch_fulltext_tool(self.fetcher))
        registry.register(build_analyze_paper_tool(analyzer, self.arxiv, self.memory))
        registry.register(build_web_search_tool(self.web))
        registry.register(build_web_fetch_tool(self.web))
        return registry

    def _build_paper_analyzer(self):
        from ..agents.paper_analyzer import PaperAnalyzer

        return PaperAnalyzer(LLMClient(self.settings), self.settings, memory=self.memory)

    def reset(self) -> None:
        self.session_id = self._new_session_id()
        self.session_title = "new session"
        self.session_path = self._session_path(self.session_id)
        self.messages = [{"role": "system", "content": self._system_prompt()}]
        self.transcript_messages = list(self.messages)

    def handle(self, user_input: str) -> WorkspaceTurnResult:
        start = time.time()
        if self.session_title == "new session":
            self.session_title = user_input[:80].strip() or "untitled session"
        self._append_message({"role": "user", "content": user_input})
        self._trim_history()
        self.save_session()

        tool_trace: List[Dict[str, Any]] = []
        total_tokens = 0

        for step_idx in range(self.max_tool_steps_per_turn):
            completion = self.llm.invoke_with_tools(
                messages=self.messages,
                tools=self.tools.openai_schemas(),
                temperature=self.temperature,
                tool_choice="auto",
            )
            msg = completion.choices[0].message
            usage = getattr(completion, "usage", None)
            if usage:
                total_tokens += int(getattr(usage, "total_tokens", 0) or 0)

            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                reply = (msg.content or "").strip()
                self._append_message({"role": "assistant", "content": reply})
                self._trim_history()
                self.save_session()
                return WorkspaceTurnResult(
                    reply=reply or "我没有生成有效回复。",
                    tool_calls=tool_trace,
                    tokens_used=total_tokens,
                    elapsed_ms=int((time.time() - start) * 1000),
                    stopped_reason="answered",
                )

            assistant_msg = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments or "{}",
                        },
                    }
                    for call in tool_calls
                ],
            }
            self._append_message(assistant_msg)

            for call in tool_calls:
                fn_name = call.function.name
                try:
                    fn_args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    fn_args = {}

                result = self.tools.invoke(fn_name, fn_args)
                tool_trace.append(
                    {
                        "step": step_idx + 1,
                        "tool": fn_name,
                        "args": self._clip_args(fn_args),
                        "success": result.success,
                        "error": result.error,
                    }
                )
                self._append_message(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.to_llm_string(max_chars=5000),
                    }
                )

            self._trim_history()
            self.save_session()

        warning = (
            "这一轮已经达到工具调用步数上限。我已经保留了目前的观察结果，"
            "你可以继续追问，我会接着当前上下文往下做。"
        )
        self._append_message({"role": "assistant", "content": warning})
        self.save_session()
        return WorkspaceTurnResult(
            reply=warning,
            tool_calls=tool_trace,
            tokens_used=total_tokens,
            elapsed_ms=int((time.time() - start) * 1000),
            stopped_reason="max_tool_steps",
        )

    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        sessions: List[Dict[str, Any]] = []
        for path in sorted(self.session_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            sessions.append(
                {
                    "session_id": data.get("session_id", path.stem),
                    "title": data.get("title", "untitled session"),
                    "updated_at": data.get("updated_at", ""),
                    "message_count": len(data.get("messages", [])),
                    "path": str(path),
                }
            )
            if len(sessions) >= limit:
                break
        return sessions

    def load_session(self, session_id: str = "latest") -> Dict[str, Any]:
        sessions = self.list_sessions(limit=50)
        if not sessions:
            raise FileNotFoundError("no saved agent sessions")

        selected = None
        if not session_id or session_id == "latest":
            selected = sessions[0]
        else:
            for item in sessions:
                if item["session_id"] == session_id or item["session_id"].startswith(session_id):
                    selected = item
                    break
        if selected is None:
            raise FileNotFoundError(f"session not found: {session_id}")

        data = json.loads(Path(selected["path"]).read_text(encoding="utf-8"))
        saved_messages = data.get("messages", [])
        current_system = {"role": "system", "content": self._system_prompt()}
        restored = []
        for idx, msg in enumerate(saved_messages):
            if idx == 0 and msg.get("role") == "system":
                continue
            if msg.get("role") in {"system", "user", "assistant", "tool"}:
                restored.append(msg)
        self.session_id = data.get("session_id", selected["session_id"])
        self.session_title = data.get("title", selected.get("title", "untitled session"))
        self.session_path = self._session_path(self.session_id)
        self.transcript_messages = [current_system] + restored
        self.messages = list(self.transcript_messages)
        self._trim_history()
        self.save_session()
        return {
            "session_id": self.session_id,
            "title": self.session_title,
            "message_count": len(self.transcript_messages),
            "updated_at": data.get("updated_at", ""),
        }


    def _append_message(self, message: Dict[str, Any]) -> None:
        self.messages.append(message)
        self.transcript_messages.append(message)

    def save_session(self) -> None:
        payload = {
            "session_id": self.session_id,
            "title": self.session_title,
            "project_root": str(self.project_root),
            "allow_write": self.allow_write,
            "created_at": self.session_id[:15],
            "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
            "messages": self.transcript_messages,
        }
        self.session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _trim_history(self) -> None:
        if len(self.messages) <= self.max_history_messages:
            return
        system = self.messages[0]
        tail = self.messages[-(self.max_history_messages - 1):]
        while tail and tail[0].get("role") == "tool":
            tail.pop(0)
        self.messages = [system] + tail

    def _session_path(self, session_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self.session_dir / f"{safe}.json"

    @staticmethod
    def _new_session_id() -> str:
        return f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"

    @staticmethod
    def _clip_args(args: Dict[str, Any]) -> Dict[str, Any]:
        clipped: Dict[str, Any] = {}
        for key, value in (args or {}).items():
            text = str(value)
            clipped[key] = text[:240] + ("..." if len(text) > 240 else "")
        return clipped


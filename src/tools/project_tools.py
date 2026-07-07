"""
Project workspace tools for the stateful agent.

These tools intentionally keep filesystem access scoped to the project root.
Write tools are registered only when the caller explicitly enables them.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List

from .tool import Tool, ToolResult


SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
}

SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".bin",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
}


def _safe_resolve(root: Path, user_path: str) -> Path:
    root = root.resolve()
    candidate = (root / (user_path or ".")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {user_path}") from exc
    return candidate


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _iter_text_files(root: Path, directory: str = "") -> Iterable[Path]:
    start = _safe_resolve(root, directory)
    if start.is_file():
        yield start
        return
    if not start.exists():
        return

    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        base = Path(dirpath)
        for filename in filenames:
            path = base / filename
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            yield path


def build_project_overview_tool(project_root: Path) -> Tool:
    root = Path(project_root)

    def run(args: Dict) -> ToolResult:
        max_files = int(args.get("max_files", 120))
        files: List[str] = []
        dirs: Dict[str, int] = {}
        for path in _iter_text_files(root):
            rel = _rel(root, path)
            files.append(rel)
            top = rel.split("/", 1)[0]
            dirs[top] = dirs.get(top, 0) + 1
            if len(files) >= max_files:
                break

        return ToolResult(
            success=True,
            content={
                "project_root": str(root.resolve()),
                "top_level_counts": dirs,
                "sample_files": files,
                "truncated": len(files) >= max_files,
            },
        )

    return Tool(
        name="project_overview",
        description=(
            "Inspect the current project structure. Use this first when you need "
            "orientation before answering or changing code."
        ),
        parameters={
            "type": "object",
            "properties": {
                "max_files": {
                    "type": "integer",
                    "description": "Maximum file paths to return.",
                    "default": 120,
                },
            },
        },
        run=run,
    )


def build_list_project_files_tool(project_root: Path) -> Tool:
    root = Path(project_root)

    def run(args: Dict) -> ToolResult:
        directory = str(args.get("directory") or "")
        pattern = str(args.get("pattern") or "*")
        max_files = int(args.get("max_files", 200))
        files = []
        for path in _iter_text_files(root, directory):
            rel = _rel(root, path)
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
                files.append(rel)
            if len(files) >= max_files:
                break
        return ToolResult(success=True, content={"files": files, "truncated": len(files) >= max_files})

    return Tool(
        name="list_project_files",
        description="List text-like files inside the project, optionally filtered by glob pattern.",
        parameters={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Project-relative directory."},
                "pattern": {"type": "string", "description": "Glob pattern such as '*.py' or 'src/**/*.py'.", "default": "*"},
                "max_files": {"type": "integer", "default": 200},
            },
        },
        run=run,
    )


def build_read_project_file_tool(project_root: Path) -> Tool:
    root = Path(project_root)

    def run(args: Dict) -> ToolResult:
        path_arg = str(args.get("path") or "").strip()
        if not path_arg:
            return ToolResult(success=False, error="path is required")
        start_line = max(1, int(args.get("start_line", 1)))
        max_lines = max(1, min(int(args.get("max_lines", 220)), 500))
        path = _safe_resolve(root, path_arg)
        if not path.exists() or not path.is_file():
            return ToolResult(success=False, error=f"file not found: {path_arg}")
        if path.suffix.lower() in SKIP_SUFFIXES:
            return ToolResult(success=False, error=f"refusing to read likely binary file: {path_arg}")

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

        end = min(len(lines), start_line + max_lines - 1)
        numbered = [
            f"{idx}: {lines[idx - 1]}"
            for idx in range(start_line, end + 1)
        ]
        return ToolResult(
            success=True,
            content={
                "path": _rel(root, path),
                "start_line": start_line,
                "end_line": end,
                "total_lines": len(lines),
                "content": "\n".join(numbered),
            },
        )

    return Tool(
        name="read_project_file",
        description="Read a project file with line numbers. Use focused ranges instead of reading huge files.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Project-relative file path."},
                "start_line": {"type": "integer", "default": 1},
                "max_lines": {"type": "integer", "default": 220},
            },
            "required": ["path"],
        },
        run=run,
    )


def build_search_project_tool(project_root: Path) -> Tool:
    root = Path(project_root)

    def run(args: Dict) -> ToolResult:
        query = str(args.get("query") or "")
        if not query:
            return ToolResult(success=False, error="query is required")
        directory = str(args.get("directory") or "")
        pattern = str(args.get("pattern") or "*")
        use_regex = bool(args.get("regex", False))
        case_sensitive = bool(args.get("case_sensitive", False))
        max_matches = max(1, min(int(args.get("max_matches", 80)), 200))

        flags = 0 if case_sensitive else re.IGNORECASE
        rx = None
        if use_regex:
            try:
                rx = re.compile(query, flags)
            except re.error as exc:
                return ToolResult(success=False, error=f"invalid regex: {exc}")
        needle = query if case_sensitive else query.lower()

        matches = []
        for path in _iter_text_files(root, directory):
            rel = _rel(root, path)
            if not (fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern)):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue

            for idx, line in enumerate(lines, 1):
                haystack = line if case_sensitive else line.lower()
                found = bool(rx.search(line)) if rx else needle in haystack
                if found:
                    matches.append({"path": rel, "line": idx, "text": line[:500]})
                    if len(matches) >= max_matches:
                        return ToolResult(success=True, content={"matches": matches, "truncated": True})

        return ToolResult(success=True, content={"matches": matches, "truncated": False})

    return Tool(
        name="search_project",
        description="Search text inside project files. Use this to find symbols, docs, commands, or examples.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "directory": {"type": "string", "default": ""},
                "pattern": {"type": "string", "default": "*"},
                "regex": {"type": "boolean", "default": False},
                "case_sensitive": {"type": "boolean", "default": False},
                "max_matches": {"type": "integer", "default": 80},
            },
            "required": ["query"],
        },
        run=run,
    )


def build_replace_in_project_file_tool(project_root: Path) -> Tool:
    root = Path(project_root)

    def run(args: Dict) -> ToolResult:
        path_arg = str(args.get("path") or "").strip()
        old_text = str(args.get("old_text") or "")
        new_text = str(args.get("new_text") or "")
        if not path_arg or not old_text:
            return ToolResult(success=False, error="path and old_text are required")
        path = _safe_resolve(root, path_arg)
        if not path.exists() or not path.is_file():
            return ToolResult(success=False, error=f"file not found: {path_arg}")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            return ToolResult(
                success=False,
                error=f"old_text must match exactly once; found {count} matches",
            )
        path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return ToolResult(success=True, content={"path": _rel(root, path), "replacements": 1})

    return Tool(
        name="replace_in_project_file",
        description=(
            "Edit a project file by replacing an exact old_text block with new_text. "
            "Only use after reading the file. Available only in --allow-write mode."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
        run=run,
    )


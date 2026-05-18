"""
Tool 抽象 —— Agent 可调用的统一工具接口

设计原则：
1. Tool 是 pure function 的包装：接受参数 → 返回 ToolResult
2. 每个 Tool 自带 JSON Schema，LLM 可通过 function calling 调用
3. 失败不抛异常，包装成 ToolResult(success=False, ...)
4. 支持元数据（tokens_used / latency），用于预算追踪

使用方式：
    tool = Tool(
        name="search_arxiv",
        description="Search arXiv for academic papers.",
        parameters={"type": "object", "properties": {...}},
        run=lambda args: arxiv.search(args["query"]),
    )
    result = tool.invoke({"query": "DPO"})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class ToolResult:
    """工具执行结果，统一的返回类型。"""
    success: bool
    content: Any = None
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_llm_string(self, max_chars: int = 2000) -> str:
        """转换为能回灌给 LLM 的字符串，过大会被截断。"""
        if not self.success:
            return f"[Tool failed] {self.error}"
        text = str(self.content)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n...[truncated, total {len(text)} chars]"
        return text


@dataclass
class Tool:
    """
    工具定义。包装一个可调用对象并声明 JSON Schema。

    Fields:
      name:        工具名（LLM 在 function calling 中看到的名字）
      description: 工具用途的自然语言描述（给 LLM 看）
      parameters:  JSON Schema，描述工具参数
      run:         实际执行函数，签名 (args: dict) -> Any 或 ToolResult
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    run: Callable[[Dict[str, Any]], Any]

    def invoke(self, args: Dict[str, Any]) -> ToolResult:
        """调用工具，返回统一的 ToolResult。"""
        try:
            raw = self.run(args or {})
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            )
        if isinstance(raw, ToolResult):
            return raw
        return ToolResult(success=True, content=raw)

    def to_openai_schema(self) -> Dict[str, Any]:
        """生成 OpenAI function calling 格式的工具定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """
    工具注册表。按 name 索引工具，支持批量导出 schema。

    使用：
        registry = ToolRegistry()
        registry.register(search_arxiv_tool)
        schemas = registry.openai_schemas()   # 给 LLM
        result = registry.invoke("search_arxiv", {"query": "DPO"})
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> "ToolRegistry":
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def invoke(self, name: str, args: Dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {name}. Available: {list(self._tools.keys())}",
            )
        return tool.invoke(args)

    def openai_schemas(self) -> list:
        return [t.to_openai_schema() for t in self._tools.values()]

    def names(self) -> list:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

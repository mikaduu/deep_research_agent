"""
LangGraph 版工具定义

使用 langchain_core 的 @tool 装饰器，替代手搓的 Tool/ToolRegistry/builder 模式。
LangGraph 会自动从函数签名和 docstring 生成 JSON Schema。

工具分组：
- 搜索：search_arxiv / search_semantic_scholar / web_search / web_fetch
- 论文：fetch_paper_fulltext / analyze_paper
- 记忆：retrieve_memory / save_note / save_research_episode
- 委派：delegate_to_critic / delegate_to_reviser
"""

from langchain_core.tools import tool
from typing import Optional

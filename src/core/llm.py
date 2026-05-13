from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import Settings


class LLMClient:
    """
    LLM 调用的薄封装。

    两种模式：
      - invoke()            : 纯文本输入输出，返回 str
      - invoke_with_tools() : Function calling，返回原始 ChatCompletion
                              （agent loop 需要访问 tool_calls/usage）
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required in .env.")
        self.settings = settings
        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

    def invoke(self, messages: List[Dict[str, str]], temperature: float) -> str:
        """纯文本 completion。绝大多数 prompt-driven agent 用这个。"""
        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def invoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float,
        tool_choice: str = "auto",
    ):
        """
        Function calling 模式。

        Args:
            messages: 对话历史（含 system / user / assistant / tool 角色）
            tools:    OpenAI function calling schema 列表
            temperature: 采样温度
            tool_choice: "auto" / "required" / "none" 或特定工具 dict

        Returns:
            原始 ChatCompletion 对象（含 choices[0].message.tool_calls / usage）
        """
        return self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
        )

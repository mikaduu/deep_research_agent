from typing import Dict, List

from openai import OpenAI

from .config import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required in .env.")
        self.settings = settings
        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

    def invoke(self, messages: List[Dict[str, str]], temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


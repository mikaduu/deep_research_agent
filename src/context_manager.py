from typing import List, Dict
from collections import deque


class ContextManager:
    def __init__(self, max_chars: int = 7000):
        self.max_chars = max_chars
        self.history: deque = deque(maxlen=20)

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_context(self, system_prompt: str = "") -> List[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        total_chars = len(system_prompt)
        for msg in reversed(self.history):
            msg_len = len(msg["content"])
            if total_chars + msg_len > self.max_chars:
                break
            messages.insert(1 if system_prompt else 0, msg)
            total_chars += msg_len

        return messages

    def clear(self):
        self.history.clear()

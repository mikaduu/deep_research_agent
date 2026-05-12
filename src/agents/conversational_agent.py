"""
对话式路由 Agent - 把自然语言意图翻译成结构化工具调用

工作流程：
  1. 接收用户自然语言输入（可能是中英混杂、含上下文指代）
  2. 基于对话历史 + 长期记忆 推理用户意图
  3. 输出结构化 action：
       - ask_user         : 信息不足，追问澄清
       - evaluate         : 评估一个研究方向
       - search           : 搜索论文
       - analyze          : 分析指定论文
       - research         : 对主题做深度研究
       - memory_query     : 查询历史记忆
       - chitchat         : 闲聊 / 普通回复
  4. 由调用方（chat loop）执行 action，把结果回填给 Agent 生成自然语言回复

与记忆系统的集成：
  - 所有对话消息同步写入 MemoryManager.session（Layer 1）
  - 路由决策时会查询 Episodic/Skill/Vector 三层长期记忆，帮助识别
    "用户在追问以前研究过的话题"这种场景
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.llm import LLMClient
from ..core.utils import extract_json_object
from ..memory.memory_manager import MemoryManager


ROUTER_SYSTEM_PROMPT = """你是一个研究助手的对话路由器。你的职责：
1. 理解用户的自然语言意图（中英文皆可）
2. 结合长期记忆，识别用户是在追问之前研究过的话题还是提出新话题
3. 从对话历史中提炼检索所需的英文关键词（重要！arXiv/S2 对中文支持差）
4. 输出结构化的下一步动作，由后端执行

可用动作：
  - "ask_user": 信息不足，需要追问用户
  - "evaluate": 评估研究方向的可行性/新颖性/影响力，会检索相关论文
  - "search": 只做论文搜索，不做评估
  - "analyze": 深度分析一篇具体论文（需 arxiv_id 或 url）
  - "research": 对一个主题做完整深度研究并生成报告
  - "memory_query": 查询历史研究记忆（之前研究过什么）
  - "chitchat": 普通对话回复（打招呼、解释功能等）

决策原则：
- 用户第一次说"帮我看看XXX方向可不可行" → evaluate
- 用户说"详细研究XXX" / "写一份关于XXX的报告" → research
- 用户给了 arxiv ID 或论文 URL → analyze
- 用户只说"查查XXX的论文" → search
- 用户意图模糊（如"我想做点东西"） → ask_user
- 用户在追问上一条结果的细节 → chitchat（直接基于已有上下文回答）
- 用户问"之前研究过什么" / "我们聊过哪些话题" → memory_query

关键词提炼要求：
- 3-5 个英文检索查询
- 每个 3-8 个词，避免整句
- 避免中文字符
"""


@dataclass
class AgentAction:
    """路由 Agent 的决策输出。"""
    action: str
    reply: str = ""
    queries: List[str] = field(default_factory=list)
    topic: str = ""
    paper_id: str = ""
    focus: str = ""
    raw_intent: str = ""


class ConversationalAgent:
    """
    对话式路由 Agent。

    与 MemoryManager 的集成：
      - session 层：共用 memory.session，保证整个系统看到的会话历史一致
      - 长期记忆：每次决策前查询三层长期记忆，注入决策上下文
    """

    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryManager,
        max_history: int = 16,
    ):
        self.llm = llm
        self.memory = memory
        self.session = memory.session  # 直接复用 Layer 1
        self.max_history = max_history

    # ------------------------------------------------------------------ #
    # 对话历史管理（统一走 MemoryManager.session）
    # ------------------------------------------------------------------ #

    @property
    def history(self) -> List[Dict[str, str]]:
        """从 session memory 读取对话历史。"""
        # deque 不支持切片，需要转 list
        return list(self.session.history)

    def add_user_message(self, content: str):
        self.session.add_message("user", content)

    def add_assistant_message(self, content: str):
        self.session.add_message("assistant", content)

    def add_tool_result(self, summary: str):
        """把工具执行结果作为 assistant 的内部观察写入。"""
        self.session.add_message("assistant", f"[tool_observation]\n{summary}")

    def clear(self):
        self.session.clear()

    # ------------------------------------------------------------------ #
    # 路由决策
    # ------------------------------------------------------------------ #

    def decide(self) -> AgentAction:
        """
        基于 (长期记忆 + 会话历史) 决定下一步动作。
        """
        # 1. 取最近一条用户消息作为长期记忆查询
        user_msgs = [m for m in self.history if m["role"] == "user"]
        last_user = user_msgs[-1]["content"] if user_msgs else ""

        # 2. 查询长期记忆（情节+技能+向量），注入决策上下文
        long_term_context = ""
        if last_user:
            try:
                long_term_context = self.memory.format_context_for_prompt(last_user)
            except Exception:
                long_term_context = ""

        # 3. 构建 LLM 消息序列
        system_content = ROUTER_SYSTEM_PROMPT
        if long_term_context:
            system_content += (
                "\n\n# 与当前输入相关的长期记忆\n"
                "（基于用户最近一条消息从三层记忆中召回+精排得到，"
                "用来判断是否是追问或已研究过的话题）\n"
                f"{long_term_context[:2000]}"
            )

        messages = [{"role": "system", "content": system_content}]
        # 只取最近 N 条对话注入
        messages.extend(self.history[-self.max_history:])
        messages.append({"role": "user", "content": self._decision_prompt()})

        raw = self.llm.invoke(messages, temperature=0.2)
        data = extract_json_object(raw) or {}

        return AgentAction(
            action=data.get("action", "chitchat"),
            reply=data.get("reply", ""),
            queries=[q for q in data.get("queries", []) if isinstance(q, str) and q.strip()][:5],
            topic=data.get("topic", "").strip(),
            paper_id=data.get("paper_id", "").strip(),
            focus=data.get("focus", "").strip(),
            raw_intent=data.get("intent", "").strip(),
        )

    def _decision_prompt(self) -> str:
        return """基于以上对话历史和长期记忆，输出你决定的下一步动作。

返回 JSON 格式（严格）：
{
  "intent": "你理解的用户当前意图（一句话）",
  "action": "ask_user | evaluate | search | analyze | research | memory_query | chitchat",
  "reply": "ask_user/chitchat 时给用户的中文回复；其他动作可留空或写简短确认语",
  "topic": "evaluate/research/memory_query 时的主题原文（保留用户语言）",
  "queries": ["英文关键词1", "英文关键词2", ...],
  "paper_id": "analyze 时的 arXiv ID 或 URL",
  "focus": "analyze 时的关注点（可选）"
}

只返回 JSON，不要其他文字。"""

    # ------------------------------------------------------------------ #
    # 自然语言回复生成
    # ------------------------------------------------------------------ #

    def summarize_result(self, action: AgentAction, tool_output: Dict[str, Any]) -> str:
        """把工具的结构化返回翻译成对用户友好的中文回复。"""
        messages = [
            {"role": "system", "content":
             "你是一位友好的研究助手。基于后端工具返回的结果，用简洁自然的中文回答用户。"
             "必要时给出建议或追问，但不要冗长。"}
        ]
        messages.extend(self.history[-self.max_history:])
        messages.append({
            "role": "user",
            "content": (
                f"工具 [{action.action}] 的执行结果：\n"
                f"{json.dumps(tool_output, ensure_ascii=False, default=str)[:4000]}\n\n"
                f"请基于这些结果给用户一个自然、有帮助的回复。"
            ),
        })
        reply = self.llm.invoke(messages, temperature=0.4)
        return reply.strip()

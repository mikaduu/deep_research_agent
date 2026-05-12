# 架构文档

> 基于 Hermes Agent 设计理念，在深度论文研究能力之上增加三层记忆分层管理与自我学习循环。

---

## 项目结构

```
deep_research_agent/
├── main.py                          # CLI 入口（typer + rich）
├── src/
│   ├── orchestrator.py              # 核心编排器
│   ├── core/                        # 基础设施层
│   │   ├── config.py                # 配置管理（Settings dataclass）
│   │   ├── context_manager.py       # 会话记忆 Layer 1
│   │   ├── llm.py                   # OpenAI 兼容 LLM 客户端
│   │   ├── models.py                # 全局数据模型
│   │   ├── prompts.py               # 所有提示词模板
│   │   ├── mcp_client.py            # MCP 协议客户端（stub）
│   │   └── utils.py                 # 工具函数
│   ├── agents/                      # 专项 Agent
│   │   ├── direction_evaluator.py   # 研究方向评估 Agent
│   │   └── paper_analyzer.py        # 论文深度分析 Agent
│   ├── memory/                      # 三层记忆系统
│   │   ├── __init__.py
│   │   ├── memory_manager.py        # 统一记忆管理器
│   │   ├── episodic_memory.py       # 情节记忆 Layer 2（SQLite+FTS5）
│   │   ├── skill_memory.py          # 技能记忆 Layer 3（SQLite+FTS5）
│   │   ├── vector_store.py          # 语义向量存储（Chroma）
│   │   ├── retriever.py             # TF-IDF 检索（轻量备用）
│   │   └── store.py                 # Markdown 笔记持久化
│   ├── learning/                    # 自我学习模块
│   │   ├── __init__.py
│   │   └── reflection.py            # 反思引擎
│   └── services/
│       └── paper_search.py          # arXiv + Semantic Scholar 搜索
├── workspace/                       # 运行时数据目录
│   ├── memory/
│   │   ├── episodic.db              # 情节记忆数据库
│   │   └── skills.db                # 技能记忆数据库
│   ├── vector_db/                   # Chroma 向量数据库
│   ├── notes/                       # Markdown 笔记
│   └── reports/                     # 生成的研究报告
├── requirements.txt
├── environment.yml
├── ARCHITECTURE.md
└── ROADMAP.md
```

---

## 三层记忆架构

Hermes Agent 的核心设计是将记忆按时效和用途分层，每层独立存储、统一查询。

```
┌─────────────────────────────────────────────────────────┐
│                      MemoryManager                      │
│   查询：同时检索三层，格式化后注入提示词                  │
│   写入：按内容类型路由到对应层                            │
├──────────────┬──────────────────┬───────────────────────┤
│   Layer 1    │    Layer 2       │      Layer 3          │
│  Session     │   Episodic       │      Skill            │
│  会话记忆    │   情节记忆        │     技能记忆           │
│              │                  │                       │
│ContextManager│ EpisodicMemory   │    SkillMemory        │
│ 内存 deque   │ SQLite + FTS5    │   SQLite + FTS5       │
│ 当前会话     │ 跨会话持久化      │   跨会话持久化         │
│ 对话上下文   │ 历史研究情节      │   研究策略/模式        │
└──────────────┴──────────────────┴───────────────────────┘
                         +
               Vector Store（Chroma）
               语义向量检索，覆盖所有存储内容
```

### 各层职责

| 层级 | 类 | 存储介质 | 生命周期 | 存储内容 |
|------|----|----------|----------|----------|
| Layer 1 | `ContextManager` | 内存 deque | 当前进程 | 对话消息历史 |
| Layer 2 | `EpisodicMemory` | SQLite + FTS5 | 永久 | 完整研究情节、洞见、标签 |
| Layer 3 | `SkillMemory` | SQLite + FTS5 | 永久 | 可复用研究策略、触发条件 |
| 向量层 | `VectorMemory` | Chroma | 永久 | 所有内容的语义向量索引 |

---

## 自我学习循环

每次 `run_deep_research` 完成后，`ReflectionEngine` 自动执行反思循环：

```
研究完成（ResearchResult）
        │
        ▼
_compute_quality()
  基于置信度 × 0.6 + 有引用 × 0.2 + 有论文 × 0.2
  → quality_score ∈ [0, 1]
        │
        ▼
_extract_insights()  ── LLM ──►  insights_summary + tags + lessons_learned
        │                              │
        │                              ▼
        │                     EpisodicMemory.add_episode()
        │                     VectorMemory.add()
        ▼
_extract_skills()    ── LLM ──►  skills[]（name, trigger_conditions, content, domain）
                                       │
                                       ▼
                              SkillMemory.add_skill()

        │
        ▼
返回 ReflectionResult → 写入 ResearchResult.reflection
```

**效果**：随着使用次数增加，记忆库积累越来越多的情节和技能。下次研究同类主题时，`memory_augmented_planner_prompt` 会将历史上下文注入规划阶段，让 Agent 直接跳过已知内容、深入未知领域。

---

## 完整数据流

```
用户: python main.py research "transformer attention"
                │
                ▼
        ResearchOrchestrator
                │
                ├─ 1. memory.format_context_for_prompt(topic)
                │       ├─ episodic.search(topic)      ← Layer 2
                │       ├─ skill.find_relevant(topic)  ← Layer 3
                │       └─ vector.retrieve(topic)      ← 向量层
                │
                ├─ 2. _plan(topic, memory_context)
                │       └─ LLMClient  ← memory_augmented_planner_prompt
                │                        （有历史时注入，无历史时用标准规划）
                │
                ├─ 3. 遍历每个 TaskPlanItem：
                │       ├─ search_papers(task.search_query)
                │       │     ├─ ArxivSearcher
                │       │     └─ SemanticScholarSearcher
                │       ├─ vector.retrieve(task.goal)  ← 语义 RAG
                │       ├─ LLMClient  ← summarizer_prompt
                │       └─ memory.save_task_result()   → 向量层
                │
                ├─ 4. _write_report()
                │       └─ LLMClient  ← reporter_prompt
                │
                ├─ 5. _save_report()  → workspace/reports/*.md
                │
                └─ 6. reflection_engine.reflect(result)  ← 自我学习
                        ├─ → EpisodicMemory（Layer 2）
                        ├─ → SkillMemory（Layer 3）
                        └─ → ResearchResult.reflection
```

---

## 各文件详解

### `src/core/config.py` — 配置管理

`Settings` dataclass，通过 `Settings.from_env(project_root)` 从 `.env` 加载。

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `llm_model` | `gpt-4o-mini` | LLM 模型名称 |
| `llm_api_key` | — | API 密钥 |
| `llm_base_url` | OpenAI 官方 | 支持代理/国内镜像 |
| `planner_temperature` | 0.2 | 规划 Agent 温度（保守） |
| `researcher_temperature` | 0.3 | 研究 Agent 温度 |
| `writer_temperature` | 0.2 | 写作 Agent 温度 |
| `max_plan_items` | 5 | 研究计划最多任务数 |
| `search_top_k` | 6 | 每次搜索最多结果数 |
| `memory_top_k` | 4 | 记忆检索最多返回条数 |
| `context_max_chars` | 7000 | 会话上下文最大字符数 |
| `workspace_dir` | `workspace/` | 工作目录（记忆库、报告均在此） |

---

### `src/core/models.py` — 数据模型

系统中所有数据结构的定义，贯穿从搜索、研究、记忆到反思的完整流程。

#### `TaskPlanItem` — 研究计划中的单个任务

由 `planner_prompt` / `memory_augmented_planner_prompt` 驱动 LLM 生成，`orchestrator._plan()` 解析后返回列表。

| 字段 | 类型 | 含义 |
|------|------|------|
| `title` | `str` | 任务的简短标题，如 `"RAG 检索机制综述"`，用于日志和报告章节标题 |
| `goal` | `str` | 这个任务要回答的具体问题，传给 `summarizer_prompt` 作为任务目标 |
| `search_query` | `str` | 实际发给 arXiv / Semantic Scholar 的搜索词，由 LLM 生成以提高命中率 |

#### `SourceItem` — 单条搜索结果

由 `orchestrator.run_deep_research()` 将 `PaperItem` 转换而来，传给 `summarizer_prompt` 作为证据。

| 字段 | 类型 | 含义 |
|------|------|------|
| `title` | `str` | 论文标题 |
| `url` | `str` | 论文链接，提示词中要求 LLM 只能引用这里出现的 URL，防止幻觉 |
| `snippet` | `str` | 摘要前 300 字，注入提示词供 LLM 阅读，控制 token 用量 |
| `rank` | `int` | 在搜索结果中的排名（0-based），arXiv 按相关性排序，值越小越相关 |

#### `PaperItem` — 论文完整数据

由 `ArxivSearcher` 和 `SemanticScholarSearcher` 返回，是系统中论文的统一表示。

| 字段 | 类型 | 含义 |
|------|------|------|
| `paper_id` | `str` | 论文唯一标识，arXiv 格式如 `2305.10601`，Semantic Scholar 格式如 `CorpusId:12345` |
| `title` | `str` | 论文标题，用于去重（`seen` set）和展示 |
| `authors` | `List[str]` | 作者列表，CLI 展示时取前 3 位 |
| `abstract` | `str` | 完整摘要，转为 `SourceItem` 时截取前 300 字作为 `snippet` |
| `url` | `str` | 可访问的论文链接，作为引用 URL 的来源 |
| `published` | `str` | 首次发布日期，ISO 格式字符串，arXiv 有值，Semantic Scholar 可能为空 |
| `updated` | `str` | 最后更新日期，仅 arXiv 有值 |
| `categories` | `List[str]` | arXiv 分类标签，如 `["cs.CL", "cs.AI"]`，Semantic Scholar 为空列表 |

#### `Citation` — LLM 生成的引用

由 `summarizer_prompt` 要求 LLM 从 `SourceItem` 中选取并生成，存在 `TaskRunResult.citations` 里。

| 字段 | 类型 | 含义 |
|------|------|------|
| `title` | `str` | 被引用来源的标题 |
| `url` | `str` | 被引用来源的 URL，只能来自已提供的 `SourceItem` |
| `reason` | `str` | 为什么引用这篇，如 `"提出了 HyDE 方法，直接支持本节论点"` |

#### `TaskRunResult` — 单个任务的执行结果

由 `orchestrator._summarize()` 生成，是研究过程的核心中间产物。

| 字段 | 类型 | 含义 |
|------|------|------|
| `task` | `TaskPlanItem` | 对应的原始任务，保留引用以便报告阶段知道每段内容对应哪个目标 |
| `summary_markdown` | `str` | LLM 生成的 Markdown 格式摘要，直接用于最终报告的章节内容 |
| `key_points` | `List[str]` | 3-5 条核心要点，供 CLI 快速展示和报告提炼 |
| `citations` | `List[Citation]` | 本任务用到的引用列表，最终报告的参考文献来源 |
| `confidence` | `float` | LLM 对本次摘要的置信度，范围 0-1；同时作为 `ReflectionEngine._compute_quality()` 的输入 |
| `sources_used` | `List[SourceItem]` | 本任务实际使用的搜索结果，记录研究过程的可追溯性 |
| `metadata` | `Dict` | 预留扩展字段，可存耗时、token 用量等 |

#### `MemoryHit` — 记忆检索命中

由 `VectorMemory.retrieve()` 返回，注入 `summarizer_prompt` 的 `rag_context` 参数。

| 字段 | 类型 | 含义 |
|------|------|------|
| `doc_id` | `str` | 文档 ID，格式为 `task:任务标题` 或 `episode:ep_id`，标识来源 |
| `score` | `float` | 相似度分数，由 Chroma 距离转换（`1.0 - distance`），越接近 1 越相关 |
| `content` | `str` | 命中的文档内容，直接拼接后注入提示词作为历史上下文 |
| `metadata` | `Dict` | 存储时附带的元数据，如 `{"title": "...", "type": "episode"}` |

#### `ReflectionResult` — 反思结果

由 `ReflectionEngine.reflect()` 返回，挂载在 `ResearchResult.reflection` 上，同时在 CLI 中展示。

| 字段 | 类型 | 含义 |
|------|------|------|
| `episode_id` | `str` | 本次研究写入情节记忆后得到的 ID，可用于后续追踪 |
| `quality_score` | `float` | 本次研究质量评分，公式：`avg_confidence×0.6 + has_citations×0.2 + has_papers×0.2` |
| `insights_summary` | `str` | LLM 提炼的核心洞见，同时存入 `Episode.insights` 并在 CLI 中以 Panel 展示 |
| `tags` | `List[str]` | 主题标签，存入 `Episode.tags`，如 `["rag", "retrieval", "llm"]` |
| `lessons_learned` | `List[str]` | 关于研究方法的经验教训，如 `"加 survey 关键词效果更好"`，在 CLI 中以列表展示 |
| `skills_learned` | `int` | 本次新写入技能记忆的技能数量，0 表示 LLM 认为没有值得保存的新模式 |

#### `ResearchResult` — 完整研究结果

`run_deep_research()` 的最终返回值，包含整个研究过程的所有产物。

| 字段 | 类型 | 含义 |
|------|------|------|
| `topic` | `str` | 用户输入的研究主题，贯穿整个流程 |
| `plan` | `List[TaskPlanItem]` | LLM 生成的研究计划，3-5 个任务 |
| `task_results` | `List[TaskRunResult]` | 每个任务的执行结果，顺序与 `plan` 对应 |
| `final_report_markdown` | `str` | `reporter_prompt` 生成的完整 Markdown 报告，包含执行摘要、详细发现、风险、建议、参考文献 |
| `report_file` | `str` | 报告保存的绝对路径，格式 `workspace/reports/20240429_123456_主题.md` |
| `papers` | `List[PaperItem]` | 所有任务搜索到的论文合集（未去重），CLI 用于展示"共分析 N 篇论文" |
| `created_at` | `str` | 研究开始的 UTC 时间戳，自动生成 |
| `reflection` | `Optional[ReflectionResult]` | 反思结果，研究完成后由 `ReflectionEngine` 填充；首次运行时也有值，只是记忆库为空 |

---

### `src/core/prompts.py` — 提示词模板

| 函数 | 用途 |
|------|------|
| `planner_prompt(topic, max_items)` | 标准研究规划，无历史记忆时使用 |
| `memory_augmented_planner_prompt(topic, max_items, memory_context)` | 注入历史记忆的规划，避免重复已知内容 |
| `summarizer_prompt(topic, task, sources, rag_context)` | 单任务摘要，结合搜索结果和 RAG 记忆 |
| `reflection_prompt(topic, result)` | 检查研究质量，判断是否需要补充搜索 |
| `reporter_prompt(topic, task_results)` | 汇总所有任务，生成完整 Markdown 报告 |
| `learning_reflection_prompt(result, quality_score)` | 自我学习：提炼洞见、标签、经验教训 |
| `skill_extraction_prompt(result)` | 自我学习：识别可复用研究策略 |

---

### `src/core/context_manager.py` — Layer 1 会话记忆

`ContextManager` 使用 `deque(maxlen=20)` 存储最近 20 条消息。

- `add_message(role, content)` — 追加消息
- `get_context(system_prompt)` — 从最新消息往前累加，超出 `max_chars` 停止
- `clear()` — 清空历史

滑动窗口策略：优先保留最近对话，自动丢弃过旧历史。

---

### `src/memory/episodic_memory.py` — Layer 2 情节记忆

跨会话持久化每次研究会话的完整结果和提炼洞见。底层：SQLite 主表 + FTS5 虚拟表（INSERT 触发器自动同步）。

#### `Episode` 字段说明

| 字段 | 类型 | 含义 |
|------|------|------|
| `id` | `str` | 8 位 hex 随机 ID，如 `a3f2c1b0` |
| `topic` | `str` | 研究主题，FTS5 检索的主要匹配字段 |
| `content` | `str` | 完整的最终报告 Markdown，供语义检索时提供详细上下文 |
| `insights` | `str` | LLM 提炼的 2-3 句核心洞见，是注入下次规划提示词的主要内容，比 `content` 更精炼 |
| `tags` | `List[str]` | 主题标签，如 `["rag", "retrieval", "llm"]`，FTS5 检索和 CLI 展示用 |
| `quality_score` | `float` | 本次研究的质量评分 0-1，由 `_compute_quality()` 计算 |
| `created_at` | `str` | UTC 时间戳，`get_recent()` 按此排序 |
| `metadata` | `Dict` | 预留扩展字段 |

#### 方法

| 方法 | 说明 |
|------|------|
| `add_episode(...)` | 写入新情节，返回 episode_id |
| `search(query, limit)` | FTS5 全文检索，失败自动降级为 LIKE |
| `get_recent(limit)` | 按时间倒序返回最近 n 条 |
| `count()` | 返回情节总数 |

---

### `src/memory/skill_memory.py` — Layer 3 技能记忆

跨会话持久化从研究中学到的可复用策略和模式。底层同样使用 SQLite + FTS5。

#### `Skill` 字段说明

| 字段 | 类型 | 含义 |
|------|------|------|
| `id` | `str` | 8 位 hex 随机 ID |
| `name` | `str` | 技能名称，如 `"Survey-First Strategy"`，简洁可识别 |
| `description` | `str` | 技能的功能描述，说明这个技能做什么 |
| `trigger_conditions` | `str` | 何时应用此技能，如 `"当研究一个新领域时，先搜索 survey 论文"`，注入规划提示词时帮助 LLM 判断是否适用 |
| `content` | `str` | 具体策略内容，1-3 句，是实际注入提示词的核心文本 |
| `domain` | `str` | 所属研究领域，如 `nlp`、`cv`、`general`，`find_relevant()` 支持按领域过滤 |
| `usage_count` | `int` | 被检索并使用的次数，通过 `update_usage()` 更新 |
| `success_rate` | `float` | 使用后研究质量高的比例，初始 0.5，随使用积累趋向真实水平 |
| `created_at` | `str` | 创建时间 |
| `updated_at` | `str` | 最后一次 `update_usage()` 的时间 |

#### 方法

| 方法 | 说明 |
|------|------|
| `add_skill(...)` | 写入新技能，返回 skill_id |
| `find_relevant(query, domain, limit)` | FTS5 检索最相关技能，支持按领域过滤 |
| `update_usage(skill_id, success)` | 更新使用次数和成功率，支持技能演化 |
| `count()` | 返回技能总数 |

---

### `src/memory/memory_manager.py` — 统一记忆管理器

三层记忆的统一入口，屏蔽底层细节。

```python
MemoryManager
├── session  : ContextManager    # Layer 1
├── episodic : EpisodicMemory    # Layer 2
├── skill    : SkillMemory       # Layer 3
└── vector   : VectorMemory      # 语义向量层
```

| 方法 | 说明 |
|------|------|
| `get_context_for_task(query)` | 同时查询三层 + 向量，返回结构化字典 |
| `format_context_for_prompt(query)` | 格式化为可直接注入提示词的字符串 |
| `save_task_result(doc_id, title, body)` | 单任务结果 → 向量存储 |
| `save_research_episode(...)` | 完整研究 → 情节记忆 + 向量存储 |
| `save_skill(...)` | 学到的技能 → 技能记忆 |
| `stats()` | 返回三层记忆的文档数量统计 |

---

### `src/memory/vector_store.py` — 语义向量存储

基于 Chroma 持久化向量数据库，使用 `all-MiniLM-L6-v2` 嵌入模型（支持中英文）。

| 方法 | 说明 |
|------|------|
| `add(doc_id, content, metadata)` | 添加文档，自动生成向量 |
| `retrieve(query, top_k)` | 语义检索，返回 `MemoryHit` 列表（距离转相似度） |
| `delete(doc_id)` | 删除指定文档 |
| `clear()` | 清空所有文档 |
| `count()` | 返回文档总数 |

---

### `src/learning/reflection.py` — 反思引擎

自我学习的核心，每次研究结束后自动执行。

| 方法 | 说明 |
|------|------|
| `reflect(result)` | 完整反思流程入口，返回 `ReflectionResult` 字典 |
| `_compute_quality(result)` | 基于置信度/引用/论文数计算质量分 |
| `_extract_insights(result, quality)` | LLM 提炼洞见和标签 |
| `_extract_skills(result)` | LLM 识别可复用研究策略 |

质量评分公式：`score = avg_confidence × 0.6 + has_citations × 0.2 + has_papers × 0.2`

---

### `src/agents/direction_evaluator.py` — 研究方向评估 Agent

`evaluate_direction(direction) -> Dict`

1. arXiv 搜索 8 篇 + Semantic Scholar 搜索 5 篇
2. 格式化论文摘要
3. LLM 评估返回 `feasibility` / `novelty` / `impact` 三个 0-1 分数
4. 结果存入向量记忆

---

### `src/agents/paper_analyzer.py` — 论文分析 Agent

`analyze(paper, focus) -> Dict`

深度分析单篇论文，返回：`summary`、`problem`、`contributions`、`methods`、`results`、`limitations`、`future_work`、`relevance_score`。支持 `focus` 参数指定关注点。

---

### `src/services/paper_search.py` — 论文搜索服务

| 类 | 说明 |
|----|------|
| `ArxivSearcher` | 调用 arXiv 官方 SDK，按相关性搜索，返回 `List[PaperItem]` |
| `SemanticScholarSearcher` | 调用 Semantic Scholar REST API（无需密钥），返回 `List[PaperItem]` |

---

### `src/orchestrator.py` — 核心编排器

`ResearchOrchestrator` 协调所有组件。

| 方法 | 说明 |
|------|------|
| `run_deep_research(topic)` | 完整研究流程（含记忆查询和反思循环） |
| `evaluate_direction(direction)` | 委托给 `DirectionEvaluator` |
| `analyze_paper(paper, focus)` | 委托给 `PaperAnalyzer` |
| `search_papers(query)` | 合并 arXiv + Semantic Scholar 结果，去重 |
| `memory_stats()` | 返回三层记忆统计 |

---

### `main.py` — CLI 入口

使用 `typer` + `rich` 构建美化命令行界面。

| 命令 | 说明 |
|------|------|
| `evaluate <方向>` | 评估研究方向，输出评分表格和分析 |
| `search <关键词>` | 搜索论文，列表展示 |
| `analyze <arXiv_ID>` | 深度分析单篇论文 |
| `research <主题>` | 完整深度研究，生成报告，研究结束后展示反思结果和记忆库统计 |

---

## 记忆积累效果示意

```
第 1 次: research "transformer attention"
  → 无历史记忆，标准规划
  → 研究完成 → 反思 → episodic.db 写入 1 条情节，skills.db 写入 N 个技能

第 2 次: research "transformer attention variants"
  → 检索到第 1 次的情节和技能
  → memory_augmented_planner_prompt 注入历史上下文
  → 规划跳过已知内容，直接深入未知领域
  → 研究完成 → 反思 → 记忆库继续积累

第 N 次: 记忆库越来越丰富，规划越来越精准
```

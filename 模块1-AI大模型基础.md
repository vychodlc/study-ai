# 模块1: AI大模型基础

## 课程概述

本模块是AI大模型全栈开发课程的基础模块，旨在帮助学员建立对大模型技术的系统性认知。我们将从提示工程（Prompt Engineering）出发，深入理解RAG技术的核心价值，探索Agent的设计与实现，最后延伸到多模态前沿技术。

---

## 第一章：从提示工程到RAG：构建大模型的知识与交互基础

### 1.1 Prompt Engineering 与 Context Engineering

#### 1.1.1 提示工程基础

**什么是提示工程（Prompt Engineering）？**

提示工程是一门设计和优化输入提示（Prompt）的技术，目的是引导大语言模型（LLM）生成更准确、更符合预期的输出。提示工程是与大模型交互的核心技能。

**提示的基本构成要素：**

1. **指令（Instruction）**：明确告诉模型需要做什么
2. **上下文（Context）**：提供背景信息帮助模型理解任务
3. **输入数据（Input Data）**：需要处理的具体内容
4. **输出指示（Output Indicator）**：期望的输出格式

**基础提示示例：**

```
# 简单提示
请将以下英文翻译成中文：Hello, how are you?

# 结构化提示
角色：你是一位专业的技术文档翻译专家
任务：将以下英文技术文档翻译成中文
要求：
1. 保持专业术语的准确性
2. 译文通顺自然
3. 保留代码块不翻译

原文：
[待翻译内容]
```

**提示设计的核心原则：**

| 原则 | 说明 | 示例 |
|------|------|------|
| 清晰明确 | 避免模糊表述 | "写一篇500字的文章" vs "写一篇文章" |
| 具体详细 | 提供足够的细节 | 指定格式、长度、风格等 |
| 结构化 | 使用分隔符和标记 | 使用###、"""等分隔不同部分 |
| 渐进式 | 复杂任务分步骤 | 先分析，再总结，最后给出建议 |

#### 1.1.2 高级提示技巧

**1. 少样本学习（Few-Shot Learning）**

通过在提示中提供几个示例，帮助模型理解任务模式：

```
请对以下评论进行情感分类：

评论：这家餐厅的服务太棒了！
情感：正面

评论：等了一个小时才上菜，太慢了。
情感：负面

评论：菜品一般，价格偏贵。
情感：负面

评论：今天的用餐体验非常愉快，会再来的。
情感：
```

**2. 思维链（Chain-of-Thought, CoT）**

引导模型展示推理过程，提高复杂问题的解决能力：

```
问题：小明有15个苹果，给了小红3个，又买了7个，最后有多少个苹果？

让我们一步步思考：
1. 小明最初有15个苹果
2. 给了小红3个后：15 - 3 = 12个
3. 又买了7个：12 + 7 = 19个
答案：小明最后有19个苹果
```

**3. 自我一致性（Self-Consistency）**

多次生成答案，选择最一致的结果：

```
请用三种不同的方法解决以下问题，然后给出最终答案：
[问题描述]

方法1：...
方法2：...
方法3：...
综合以上分析，最终答案是：...
```

**4. 角色扮演（Role-Playing）**

```
你是一位拥有20年经验的资深架构师。请从架构设计的角度，
评估以下微服务方案的优缺点，并给出改进建议。
```

#### 1.1.3 上下文工程（Context Engineering）的核心

**什么是上下文工程？**

上下文工程是指如何有效地组织、管理和利用输入给模型的上下文信息。随着大模型上下文窗口的扩大（从4K到100K+），上下文工程变得越来越重要。

**上下文工程的核心要素：**

1. **信息筛选**：选择最相关的信息放入上下文
2. **信息组织**：合理安排信息的顺序和结构
3. **信息压缩**：在有限窗口内最大化信息密度
4. **动态管理**：根据对话进展动态调整上下文

**上下文组织策略：**

```
┌─────────────────────────────────────────┐
│           系统提示（System Prompt）        │
│         定义角色、能力边界、行为准则         │
├─────────────────────────────────────────┤
│           知识上下文（Knowledge）           │
│      RAG检索的文档、知识库片段等            │
├─────────────────────────────────────────┤
│           对话历史（History）              │
│         之前的对话轮次，保持连贯性           │
├─────────────────────────────────────────┤
│           当前输入（Current Input）         │
│              用户的最新问题                 │
└─────────────────────────────────────────┘
```

#### 1.1.4 长上下文管理

**长上下文带来的挑战：**

1. **注意力稀释**：上下文过长时，模型对关键信息的关注度下降
2. **计算成本**：上下文越长，推理成本呈二次方增长
3. **信息遗忘**：中间位置的信息容易被忽略（Lost in the Middle问题）

**解决方案：**

**1. 分层上下文管理**
```
第一层：核心指令和约束（始终保留）
第二层：当前任务相关的知识（动态更新）
第三层：历史对话摘要（压缩存储）
第四层：详细历史（必要时检索）
```

**2. 滑动窗口策略**
```python
def manage_context(messages, max_tokens=4000):
    """保留最近的对话，压缩历史"""
    system_prompt = messages[0]  # 始终保留系统提示
    recent_messages = messages[-10:]  # 保留最近10轮

    # 压缩中间的历史对话
    if len(messages) > 11:
        middle_messages = messages[1:-10]
        summary = summarize(middle_messages)
        return [system_prompt, {"role": "system", "content": f"历史摘要：{summary}"}] + recent_messages

    return messages
```

**3. 重要性排序**
- 将最重要的信息放在开头和结尾
- 使用明确的标记突出关键内容
- 定期重申重要约束

---

### 1.2 RAG与私有知识库的构建与作用

#### 1.2.1 RAG的核心价值

**什么是RAG（Retrieval-Augmented Generation）？**

RAG（检索增强生成）是一种结合了信息检索和文本生成的技术框架。它通过从外部知识库中检索相关信息，将其作为上下文提供给大语言模型，从而增强模型的生成能力。

**RAG解决的核心问题：**

| 问题 | 说明 | RAG如何解决 |
|------|------|-------------|
| 知识截止 | 模型训练后无法获取新知识 | 实时检索最新信息 |
| 幻觉问题 | 模型可能编造不存在的信息 | 基于真实文档生成 |
| 领域知识 | 通用模型缺乏专业深度 | 接入领域知识库 |
| 数据隐私 | 敏感数据不能用于训练 | 本地部署，数据不出域 |
| 可追溯性 | 难以验证答案来源 | 提供原文引用 |

**RAG vs 微调（Fine-tuning）对比：**

```
┌─────────────────┬─────────────────┬─────────────────┐
│     维度        │      RAG        │     微调        │
├─────────────────┼─────────────────┼─────────────────┤
│  知识更新       │  实时更新        │  需要重新训练    │
│  成本          │  较低           │  较高           │
│  实现难度       │  中等           │  较高           │
│  知识可控性     │  高             │  中             │
│  专业深度       │  中             │  高             │
│  适用场景       │  知识密集型      │  风格/能力增强   │
└─────────────────┴─────────────────┴─────────────────┘
```

#### 1.2.2 私有知识库构建三部曲

**第一步：数据采集与预处理**

```
数据来源：
├── 结构化数据
│   ├── 数据库记录
│   ├── Excel/CSV文件
│   └── JSON/XML文件
├── 半结构化数据
│   ├── 网页内容
│   ├── Markdown文档
│   └── 邮件
└── 非结构化数据
    ├── PDF文档
    ├── Word文档
    ├── 图片（OCR）
    └── 音视频（ASR）
```

**数据预处理流程：**

```python
# 文档处理流水线示例
class DocumentProcessor:
    def process(self, document):
        # 1. 格式转换
        text = self.extract_text(document)

        # 2. 清洗
        text = self.clean(text)  # 去除噪音、特殊字符

        # 3. 分块
        chunks = self.chunk(text, chunk_size=512, overlap=50)

        # 4. 元数据提取
        metadata = self.extract_metadata(document)

        return [(chunk, metadata) for chunk in chunks]
```

**第二步：向量化与索引**

```python
from sentence_transformers import SentenceTransformer

# 加载Embedding模型
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# 文档向量化
def vectorize_documents(chunks):
    embeddings = model.encode(chunks, normalize_embeddings=True)
    return embeddings

# 构建索引
import faiss

def build_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 内积相似度
    index.add(embeddings)
    return index
```

**第三步：检索与集成**

```python
def retrieve(query, index, chunks, top_k=5):
    # 查询向量化
    query_embedding = model.encode([query], normalize_embeddings=True)

    # 检索最相似的文档
    scores, indices = index.search(query_embedding, top_k)

    # 返回检索结果
    results = [(chunks[i], scores[0][j]) for j, i in enumerate(indices[0])]
    return results
```

#### 1.2.3 向量数据库（Vector DB）

**主流向量数据库对比：**

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| **FAISS** | Facebook开源，高性能，纯CPU/GPU | 研究、原型开发 |
| **Milvus** | 云原生，分布式，高可用 | 企业级生产环境 |
| **Pinecone** | 全托管服务，开箱即用 | 快速上线，无运维 |
| **Chroma** | 轻量级，易于集成 | 小规模应用 |
| **Weaviate** | GraphQL接口，混合搜索 | 复杂查询需求 |
| **Elasticsearch** | 成熟生态，混合检索 | 已有ES基础设施 |

**向量检索的核心概念：**

```
相似度度量方法：

1. 余弦相似度（Cosine Similarity）
   similarity = (A · B) / (||A|| × ||B||)
   范围：[-1, 1]，1表示完全相同

2. 欧氏距离（Euclidean Distance）
   distance = √(Σ(ai - bi)²)
   值越小越相似

3. 内积（Inner Product）
   score = A · B
   适用于归一化向量

索引类型：
- Flat：精确搜索，适合小规模
- IVF：倒排索引，平衡精度和速度
- HNSW：图索引，高召回率
- PQ：量化压缩，节省内存
```

#### 1.2.4 RAG的工作流

**完整的RAG工作流程：**

```
┌─────────────────────────────────────────────────────────────┐
│                        离线阶段                              │
├─────────────────────────────────────────────────────────────┤
│  文档收集 → 文档解析 → 文本分块 → 向量化 → 存入向量数据库      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        在线阶段                              │
├─────────────────────────────────────────────────────────────┤
│  用户提问                                                    │
│     ↓                                                       │
│  Query改写/扩展（可选）                                       │
│     ↓                                                       │
│  Query向量化                                                 │
│     ↓                                                       │
│  向量检索 → 获取Top-K相关文档                                 │
│     ↓                                                       │
│  重排序（Rerank，可选）                                       │
│     ↓                                                       │
│  构建Prompt = 系统提示 + 检索文档 + 用户问题                   │
│     ↓                                                       │
│  LLM生成答案                                                 │
│     ↓                                                       │
│  后处理（引用标注、格式化等）                                  │
│     ↓                                                       │
│  返回给用户                                                   │
└─────────────────────────────────────────────────────────────┘
```

**RAG的Prompt模板示例：**

```
你是一个专业的知识问答助手。请根据以下参考资料回答用户的问题。

## 参考资料
{retrieved_documents}

## 回答要求
1. 仅基于参考资料中的信息回答
2. 如果资料中没有相关信息，请明确说明
3. 在回答中标注信息来源
4. 保持回答简洁准确

## 用户问题
{user_question}

## 你的回答：
```

---

## 第二章：Agent：从可控性到自主反思

### 2.1 Agent如何控制幻觉、提升任务可控性

#### 2.1.1 Agent幻觉的成因

**什么是Agent幻觉？**

Agent幻觉是指AI Agent在执行任务时生成与事实不符或逻辑不一致的输出。这是大语言模型的固有问题，在Agent场景下可能导致更严重的后果。

**幻觉的主要成因：**

```
┌─────────────────────────────────────────────────────────────┐
│                      幻觉成因分析                            │
├─────────────────────────────────────────────────────────────┤
│ 1. 知识边界模糊                                              │
│    - 模型不知道自己不知道什么                                 │
│    - 对于训练数据之外的问题，倾向于"编造"答案                   │
│                                                             │
│ 2. 推理链断裂                                                │
│    - 多步推理中某一步出错，导致后续全部错误                     │
│    - 缺乏中间结果验证机制                                     │
│                                                             │
│ 3. 上下文理解偏差                                            │
│    - 对指令或上下文的误解                                     │
│    - 长上下文中信息丢失                                       │
│                                                             │
│ 4. 过度自信                                                  │
│    - 模型通常不表达不确定性                                   │
│    - 即使不确定也会给出看似确定的答案                          │
└─────────────────────────────────────────────────────────────┘
```

#### 2.1.2 提升可控性1：利用RAG或工具进行事实校验

**RAG事实校验机制：**

```python
class FactCheckingAgent:
    def __init__(self, llm, retriever, verifier):
        self.llm = llm
        self.retriever = retriever
        self.verifier = verifier

    def answer(self, question):
        # 1. 生成初步答案
        initial_answer = self.llm.generate(question)

        # 2. 从知识库检索相关事实
        facts = self.retriever.retrieve(question)

        # 3. 验证答案与事实的一致性
        verification = self.verifier.verify(
            answer=initial_answer,
            facts=facts
        )

        # 4. 如果不一致，基于事实重新生成
        if not verification.is_consistent:
            return self.llm.generate(
                question,
                context=facts,
                instruction="仅基于以下事实回答"
            )

        return initial_answer
```

**工具调用进行验证：**

```python
tools = [
    {
        "name": "search_database",
        "description": "查询数据库获取真实数据",
        "function": search_database
    },
    {
        "name": "calculate",
        "description": "执行数学计算，确保计算准确",
        "function": calculate
    },
    {
        "name": "verify_date",
        "description": "验证日期和时间信息",
        "function": verify_date
    }
]

# Agent在回答涉及事实的问题时，会调用相应工具验证
```

#### 2.1.3 提升可控性2：Function Calling和JSON模式约束

**Function Calling约束输出：**

```python
# 定义函数规范
functions = [
    {
        "name": "create_order",
        "description": "创建订单",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "产品ID"
                },
                "quantity": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100
                },
                "shipping_address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "zip_code": {"type": "string", "pattern": "^[0-9]{6}$"}
                    },
                    "required": ["street", "city", "zip_code"]
                }
            },
            "required": ["product_id", "quantity", "shipping_address"]
        }
    }
]

# 模型输出被强制约束为符合schema的JSON
```

**JSON模式确保结构化输出：**

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class AnalysisResult(BaseModel):
    """分析结果的结构化定义"""
    summary: str = Field(description="分析摘要")
    key_points: List[str] = Field(description="关键点列表")
    sentiment: str = Field(description="情感倾向", pattern="^(positive|negative|neutral)$")
    confidence: float = Field(description="置信度", ge=0, le=1)
    sources: Optional[List[str]] = Field(description="信息来源")

# 使用结构化输出
response = llm.generate(
    prompt="分析以下文本...",
    response_format=AnalysisResult
)
```

#### 2.1.4 提升可控性3：Human-in-the-Loop审批与干预

**设计人工干预节点：**

```
┌─────────────────────────────────────────────────────────────┐
│                   Human-in-the-Loop流程                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户输入 → Agent分析 → 生成方案                              │
│                           ↓                                 │
│                    ┌──────────────┐                         │
│                    │  风险评估    │                         │
│                    └──────┬───────┘                         │
│                           ↓                                 │
│              ┌────────────┴────────────┐                    │
│              ↓                         ↓                    │
│         低风险操作                  高风险操作                │
│         自动执行                    请求人工审批              │
│              ↓                         ↓                    │
│           完成                  ┌──────────────┐            │
│                                │ 人工审核     │            │
│                                └──────┬───────┘            │
│                                       ↓                     │
│                          ┌────────────┴────────────┐        │
│                          ↓                         ↓        │
│                        批准                       拒绝       │
│                          ↓                         ↓        │
│                        执行                    反馈修改       │
└─────────────────────────────────────────────────────────────┘
```

**实现代码示例：**

```python
class HumanInTheLoopAgent:
    HIGH_RISK_OPERATIONS = ["delete", "transfer", "modify_config", "send_email"]

    async def execute(self, action):
        # 评估风险等级
        risk_level = self.assess_risk(action)

        if risk_level == "high":
            # 高风险操作需要人工审批
            approval = await self.request_human_approval(action)
            if not approval.approved:
                return {"status": "rejected", "reason": approval.reason}

        # 执行操作
        result = await self.perform_action(action)
        return result

    def assess_risk(self, action):
        if action.type in self.HIGH_RISK_OPERATIONS:
            return "high"
        if action.affects_production:
            return "high"
        return "low"

    async def request_human_approval(self, action):
        # 发送审批请求（可以是Slack、邮件、Web界面等）
        notification = self.create_approval_request(action)
        await self.notify_approvers(notification)

        # 等待审批结果
        approval = await self.wait_for_approval(timeout=3600)
        return approval
```

---

### 2.2 使用思维链构建有自主反思能力的Agent

#### 2.2.1 思维链（CoT）的原理

**思维链是什么？**

思维链（Chain-of-Thought）是一种提示技术，通过引导模型展示中间推理步骤，显著提升复杂推理任务的准确性。

**CoT的工作原理：**

```
传统方式：
问题 → 答案

思维链方式：
问题 → 步骤1 → 步骤2 → ... → 步骤N → 答案
         ↑        ↑              ↑
      可验证   可验证          可验证
```

**CoT的实现方式：**

```python
# Zero-shot CoT
prompt = """
问题：一个农场有鸡和兔共35只，它们的脚共有94只，请问鸡和兔各有多少只？

让我们一步一步思考这个问题：
"""

# Few-shot CoT
prompt = """
示例问题：小明有5个苹果，小红给了他3个，他又吃了2个，现在有几个？
思考过程：
1. 小明最初有5个苹果
2. 小红给了3个，现在有：5 + 3 = 8个
3. 吃了2个，现在有：8 - 2 = 6个
答案：6个

问题：一个农场有鸡和兔共35只，它们的脚共有94只，请问鸡和兔各有多少只？
思考过程：
"""
```

#### 2.2.2 自主反思（Self-Reflection）的实现

**反思机制的设计：**

```python
class ReflectiveAgent:
    def __init__(self, llm):
        self.llm = llm
        self.memory = []

    def solve(self, problem, max_iterations=3):
        for i in range(max_iterations):
            # 生成解决方案
            solution = self.generate_solution(problem)

            # 自我反思
            reflection = self.reflect(problem, solution)

            # 如果反思认为方案正确，返回
            if reflection.is_satisfactory:
                return solution

            # 否则，基于反思结果改进
            problem = self.incorporate_feedback(problem, reflection)

        return solution

    def reflect(self, problem, solution):
        reflection_prompt = f"""
        问题：{problem}

        我的解决方案：{solution}

        请检查这个解决方案：
        1. 方案是否完全回答了问题？
        2. 推理过程是否有逻辑错误？
        3. 是否遗漏了重要的考虑因素？
        4. 答案是否可以验证？

        如果有问题，请指出具体的改进建议。
        """
        return self.llm.generate(reflection_prompt)
```

#### 2.2.3 案例框架：ReAct

**ReAct（Reasoning + Acting）框架：**

ReAct是一个将推理（Reasoning）和行动（Acting）结合的Agent框架，Agent交替进行思考和行动。

```
┌─────────────────────────────────────────────────────────────┐
│                      ReAct循环                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  问题 → Thought（思考）→ Action（行动）→ Observation（观察）  │
│            ↑                                    │           │
│            └────────────────────────────────────┘           │
│                      循环直到得到答案                         │
└─────────────────────────────────────────────────────────────┘
```

**ReAct示例：**

```
问题：刘亦菲的出生城市的人口是多少？

Thought 1: 我需要先找出刘亦菲的出生城市
Action 1: Search[刘亦菲 出生地]
Observation 1: 刘亦菲出生于湖北省武汉市

Thought 2: 刘亦菲出生于武汉，现在我需要查询武汉的人口
Action 2: Search[武汉市 人口]
Observation 2: 武汉市常住人口约1374万人

Thought 3: 我已经得到了答案
Action 3: Finish[武汉市常住人口约1374万人]
```

**ReAct代码实现：**

```python
class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def run(self, question, max_steps=10):
        history = []

        for step in range(max_steps):
            # 生成思考和行动
            prompt = self.build_prompt(question, history)
            response = self.llm.generate(prompt)

            # 解析响应
            thought, action = self.parse_response(response)
            history.append({"thought": thought, "action": action})

            # 检查是否完成
            if action.startswith("Finish"):
                return self.extract_answer(action)

            # 执行行动，获取观察结果
            observation = self.execute_action(action)
            history.append({"observation": observation})

        return "达到最大步骤数，未能得出答案"

    def execute_action(self, action):
        action_type, param = self.parse_action(action)
        if action_type == "Search":
            return self.tools["search"](param)
        elif action_type == "Calculate":
            return self.tools["calculate"](param)
        # ... 其他工具
```

#### 2.2.4 自我修正：设计批评家或评估提示

**多角色自我批评模式：**

```python
class CriticAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate_with_critique(self, task):
        # 1. 生成初始方案
        initial_solution = self.generate(task)

        # 2. 批评家角色评估
        critique = self.critique(task, initial_solution)

        # 3. 基于批评改进
        improved_solution = self.improve(task, initial_solution, critique)

        return improved_solution

    def critique(self, task, solution):
        prompt = f"""
        你是一个严格的代码评审专家。请评估以下解决方案：

        任务：{task}
        方案：{solution}

        请从以下角度进行批评：
        1. 正确性：方案是否正确解决了问题？
        2. 完整性：是否考虑了所有边界情况？
        3. 效率：是否存在性能问题？
        4. 可维护性：代码是否清晰易懂？
        5. 安全性：是否存在潜在的安全隐患？

        请具体指出问题并给出改进建议。
        """
        return self.llm.generate(prompt)

    def improve(self, task, solution, critique):
        prompt = f"""
        原始任务：{task}
        初始方案：{solution}
        批评意见：{critique}

        请根据批评意见，生成改进后的方案。
        """
        return self.llm.generate(prompt)
```

---

## 第三章：多模态前沿：从Agent构建到视频AIGC

### 3.1 多模态Agent的构建要点

#### 3.1.1 多模态大模型的能力基座

**主流多模态大模型对比：**

| 模型 | 厂商 | 模态支持 | 特点 |
|------|------|----------|------|
| GPT-4V/GPT-4o | OpenAI | 文本、图像、音频 | 综合能力最强 |
| Gemini | Google | 文本、图像、音频、视频 | 原生多模态 |
| Claude 3 | Anthropic | 文本、图像 | 长上下文、安全性 |
| Qwen-VL | 阿里 | 文本、图像 | 中文优化 |
| LLaVA | 开源 | 文本、图像 | 可本地部署 |

**多模态理解能力：**

```
┌─────────────────────────────────────────────────────────────┐
│                    多模态感知能力                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  图像理解                                                    │
│  ├── 物体识别：识别图像中的物体、人物、场景                    │
│  ├── OCR：识别图像中的文字                                   │
│  ├── 图表理解：解读图表、表格数据                             │
│  └── 空间关系：理解物体之间的位置关系                          │
│                                                             │
│  视频理解                                                    │
│  ├── 动作识别：识别视频中的动作和行为                          │
│  ├── 时序理解：理解事件的先后顺序                             │
│  └── 场景切换：识别不同场景的转换                             │
│                                                             │
│  音频理解                                                    │
│  ├── 语音识别：转换语音为文字                                 │
│  ├── 情感分析：识别说话者的情绪                               │
│  └── 音频事件：识别环境音、音乐等                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 3.1.2 核心构建点1：将视觉感知封装为Tool

```python
# 定义视觉感知工具
vision_tools = [
    {
        "name": "analyze_image",
        "description": "分析图像内容，识别物体、场景和文字",
        "parameters": {
            "image_url": {"type": "string", "description": "图像URL或base64编码"}
        }
    },
    {
        "name": "extract_text_from_image",
        "description": "从图像中提取文字（OCR）",
        "parameters": {
            "image_url": {"type": "string"},
            "language": {"type": "string", "default": "auto"}
        }
    },
    {
        "name": "compare_images",
        "description": "比较两张图像的差异",
        "parameters": {
            "image_url_1": {"type": "string"},
            "image_url_2": {"type": "string"}
        }
    }
]

# 工具实现
class VisionTools:
    def __init__(self, vision_model):
        self.model = vision_model

    def analyze_image(self, image_url):
        """分析图像内容"""
        response = self.model.analyze(
            image=image_url,
            prompt="请详细描述这张图片的内容，包括主要物体、场景、颜色、文字等。"
        )
        return response

    def extract_text_from_image(self, image_url, language="auto"):
        """OCR文字提取"""
        response = self.model.analyze(
            image=image_url,
            prompt="请提取图片中的所有文字内容。"
        )
        return response
```

#### 3.1.3 核心构建点2：多模态输入的统一表征与分发

```python
class MultimodalInputHandler:
    """处理多模态输入的统一接口"""

    def __init__(self):
        self.handlers = {
            "text": self.handle_text,
            "image": self.handle_image,
            "audio": self.handle_audio,
            "video": self.handle_video
        }

    def process(self, inputs):
        """
        处理多模态输入，转换为统一格式

        输入示例：
        [
            {"type": "text", "content": "这是一张什么图片？"},
            {"type": "image", "content": "base64_encoded_image..."},
        ]
        """
        processed_inputs = []
        for item in inputs:
            handler = self.handlers.get(item["type"])
            if handler:
                processed = handler(item["content"])
                processed_inputs.append(processed)

        return self.compose_prompt(processed_inputs)

    def handle_image(self, content):
        """处理图像输入"""
        if content.startswith("http"):
            return {"type": "image_url", "url": content}
        else:
            return {"type": "image_base64", "data": content}

    def handle_audio(self, content):
        """处理音频输入 - 转换为文本"""
        transcript = self.speech_to_text(content)
        return {"type": "text", "content": f"[音频内容]: {transcript}"}
```

#### 3.1.4 多模态Agent的应用

**应用场景示例：**

```
1. 智能文档分析
   输入：PDF文档图片
   能力：OCR + 表格解析 + 图表理解 + 问答

2. 视觉问答（VQA）
   输入：图像 + 问题
   能力：理解图像内容，回答相关问题

3. 电商商品分析
   输入：商品图片
   能力：识别商品属性、对比竞品、生成描述

4. 医疗影像辅助
   输入：X光/CT图像
   能力：识别异常区域、生成诊断建议

5. 工业质检
   输入：产品图片/视频
   能力：缺陷检测、质量评估
```

---

### 3.2 视频检索及视频生成的技术方案

#### 3.2.1 视频检索（Video Retrieval）

**多模态视频向量化检索架构：**

```
┌─────────────────────────────────────────────────────────────┐
│                     视频检索系统                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  视频输入                                                    │
│     ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    特征提取层                        │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │   │
│  │  │关键帧   │  │  音频   │  │  字幕   │  │ 元数据  │ │   │
│  │  │视觉特征 │  │ASR转文本│  │OCR提取  │  │时间戳等 │ │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘ │   │
│  │       ↓            ↓            ↓            ↓      │   │
│  │  ┌─────────────────────────────────────────────────┐│   │
│  │  │              Embedding模型                      ││   │
│  │  │     (CLIP / Chinese-CLIP / Multimodal-LLM)     ││   │
│  │  └────────────────────┬────────────────────────────┘│   │
│  └───────────────────────┼──────────────────────────────┘   │
│                          ↓                                  │
│                   向量数据库存储                             │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    检索层                            │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  查询 → 查询向量化 → 相似度搜索 → 时序聚合 → 排序     │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│                    返回视频片段                              │
└─────────────────────────────────────────────────────────────┘
```

**视频检索实现：**

```python
class VideoRetriever:
    def __init__(self, clip_model, vector_db):
        self.clip_model = clip_model
        self.vector_db = vector_db

    def index_video(self, video_path):
        """索引视频"""
        # 1. 抽取关键帧
        keyframes = self.extract_keyframes(video_path, fps=1)

        # 2. 语音转文字
        audio = self.extract_audio(video_path)
        transcript = self.speech_to_text(audio)

        # 3. 生成多模态embedding
        for i, frame in enumerate(keyframes):
            # 视觉embedding
            visual_emb = self.clip_model.encode_image(frame)

            # 对应时间段的文本embedding
            text_segment = self.get_transcript_segment(transcript, i)
            text_emb = self.clip_model.encode_text(text_segment)

            # 融合embedding
            combined_emb = self.fuse_embeddings(visual_emb, text_emb)

            # 存储
            self.vector_db.insert({
                "video_id": video_path,
                "timestamp": i,
                "embedding": combined_emb,
                "metadata": {"transcript": text_segment}
            })

    def search(self, query, top_k=5):
        """搜索视频片段"""
        # 文本查询向量化
        query_emb = self.clip_model.encode_text(query)

        # 向量搜索
        results = self.vector_db.search(query_emb, top_k=top_k)

        return results
```

#### 3.2.2 视频理解（Video Understanding）

**视频理解任务：**

| 任务 | 说明 | 技术方案 |
|------|------|----------|
| 场景分割 | 将视频分割成不同场景 | PySceneDetect + CLIP |
| 目标追踪 | 追踪视频中的物体 | YOLO + DeepSORT |
| 动作识别 | 识别视频中的动作 | VideoMAE / TimeSformer |
| 内容摘要 | 生成视频摘要 | 关键帧 + LLM |
| 视频问答 | 回答关于视频的问题 | Video-LLaVA / GPT-4V |

#### 3.2.3 视频生成（Video Generation）

**文生视频（Text-to-Video）技术：**

```
主流模型：
├── Sora (OpenAI) - 长视频生成，物理一致性好
├── Kling (快手) - 中文支持，质量高
├── Runway Gen-3 - 商用成熟，API可用
├── Pika - 短视频，动画风格
└── Stable Video Diffusion - 开源，可本地部署

核心技术：扩散模型（Diffusion Model）
├── 前向过程：逐步添加噪声，将视频变成纯噪声
└── 逆向过程：从噪声逐步去噪，生成视频

关键挑战：
├── 时序一致性：帧与帧之间的连贯性
├── 物理规律：符合现实世界的物理规则
├── 运动控制：精确控制物体的运动
└── 长视频生成：保持长时间的一致性
```

**扩散模型原理：**

```
┌─────────────────────────────────────────────────────────────┐
│                   扩散模型工作原理                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  前向扩散过程（训练时）：                                     │
│  原始视频 → 加噪声 → 加噪声 → ... → 纯噪声                    │
│  x₀        x₁        x₂            xₜ                      │
│                                                             │
│  逆向去噪过程（生成时）：                                     │
│  纯噪声 → 预测噪声 → 去噪 → ... → 生成视频                    │
│  xₜ      + 文本条件                x₀                       │
│                                                             │
│  关键组件：                                                  │
│  ├── U-Net：预测每一步的噪声                                 │
│  ├── CLIP：编码文本条件                                      │
│  ├── VAE：压缩和解压视频                                     │
│  └── 时序模块：保持帧间一致性                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 3.2.4 当前挑战与机遇

**视频生成面临的挑战：**

```
1. 时序一致性
   - 问题：物体在不同帧之间形状、颜色变化
   - 解决方向：3D感知、时序注意力机制

2. 物理规律
   - 问题：不符合重力、碰撞等物理规则
   - 解决方向：物理引擎集成、物理约束训练

3. 逻辑一致性
   - 问题：事件发展不符合常识
   - 解决方向：引入世界模型、因果推理

4. 算力瓶颈
   - 问题：生成高质量长视频需要巨大算力
   - 解决方向：模型压缩、分布式推理

5. 可控性
   - 问题：难以精确控制生成内容
   - 解决方向：ControlNet、运动轨迹引导
```

**行业机遇：**

```
应用场景：
├── 广告与营销：快速生成创意视频
├── 影视制作：预可视化、特效制作
├── 教育培训：自动生成教学视频
├── 游戏开发：过场动画、NPC行为
├── 电商展示：商品360度展示
└── 个人创作：降低视频创作门槛
```

---

## 本模块总结

### 核心知识点回顾

1. **提示工程**：掌握Prompt设计原则和高级技巧，是与大模型交互的基础
2. **上下文工程**：学会组织和管理长上下文，最大化信息利用效率
3. **RAG技术**：理解检索增强生成的价值和实现流程
4. **Agent可控性**：掌握控制幻觉、提升可控性的多种方法
5. **思维链与反思**：构建具有推理和自我改进能力的Agent
6. **多模态技术**：了解多模态Agent构建和视频AI的前沿进展

### 思考与实践

1. 设计一个包含RAG和工具调用的Agent，解决一个实际问题
2. 实现一个带有自我反思能力的代码生成Agent
3. 探索多模态大模型的API，构建一个图像问答应用

### 延伸阅读

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Documentation](https://python.langchain.com/)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

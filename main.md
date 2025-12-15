# AI大模型全栈开发课程大纲

## 模块1: AI大模型基础

### 理论知识
#### 从提示工程到RAG：构建大模型的知识与交互基础
**1. Prompt Engineering 与 Context Engineering**
- 提示工程基础
- 高级提示技巧
- 上下文工程（CE）的核心
- 长上下文管理

**2. RAG与私有知识库的构建与作用**
- RAG的核心价值
- 私有知识库构建三部曲
- 向量数据库（Vector DB）
- RAG的工作流

#### Agent：从可控性到自主反思
**1. Agent如何控制幻觉、提升任务可控性**
- Agent幻觉的成因：识别模型知识边界与推理链条的断裂
- 提升可控性1- 利用RAG或工具（Tool-use）进行事实校验
- 提升可控性2- 利用Function Calling和JSON模式约束Agent的行为与输出
- 提升可控性3- 设计Human-in-the-Loop的审批与干预节点

**2. 使用思维链构建有自主反思能力的Agent**
- 思维链（CoT）的原理
- 自主反思（Self-Reflection）的实现
- 案例框架（ReAct）
- 自我修正：设计批评家或评估提示来引导Agent自我校准

#### 多模态前沿：从Agent构建到视频AIGC
**1. 多模态Agent的构建要点**
- 多模态大模型（MLLM）的能力基座：理解GPT5、Gemini等模型的图文理解能力
- 核心构建点1：将视觉感知封装为Agent可以调用的工具（Tool）
- 核心构建点2：处理多模态输入（如图片、音频）的统一表征与分发
- 多模态Agent的应用：从看图说话到复杂的视觉问答与操作（VQA）

**2. 视频检索及视频生成的技术方案**
- 视频检索（Video Retrieval）：基于关键帧、音频（ASR）与时序的多模态向量化检索
- 视频理解（Video Understanding）：利用大模型进行场景分割、目标追踪与内容摘要
- 视频生成（Video Generation）：文生视频（T2V）模型（如Sora, Kling）的扩散模型原理
- 当前挑战与机遇：视频生成的时序一致性、逻辑性、物理规律与算力瓶颈

---

## 模块2: RAG

### 理论知识+案例详解+实操
#### Embeddings和向量数据库
**1. Embeddings模型及向量化**
- 什么是Embedding
- Word Embedding
- 余弦相似度计算
- Embedding模型的选择
- MTEB榜单
- 向量维度对模型性能的影响
- 神奇的“俄罗斯套娃”
- 如何选择适合的Embedding模型

**2. 向量数据库和向量检索**
- 什么是向量数据库
- FAISS, Elasticsearch, Milvus, Pinecone的特点
- 向量数据库与传统数据库的区别
- 如何将数据导入向量数据库
- Embedding与原数据导入Faiss
- 不同向量数据库的功能与性能比较

#### RAG技术与应用
**1. RAG检索增强生成的原理及流程**
- 大模型开发的三种范式
- 什么是RAG技术？它如何增强大模型的生成能力
- RAG的核心原理与流程
- NativeRAG
- CASE：DeepSeek +Faiss搭建本地知识库检索

**2. Query改写与知识库处理**
- Query改写
- Query联网搜索
- 知识库处理
- 场景1：知识库问题生成与检索优化
- 场景2：对话知识沉淀
- 场景3：知识库健康度检查
- 场景4：知识库版本管理与性能比较
- 如何提升RAG质量

#### RAG多模态数据处理
**1. 多模态数据处理：PDF、Word、网页等图文数据**
- PDF文档解析：挑战与策略
- Word文档解析
- 网页数据解析
- Qwen-Agent中的RAG

**2. 多模态数据处理：图片、视频等多媒体数据**
- 图像数据的多重表征
- 视频数据处理流程
- 语音转文本ASR
- GraphRAG使用
- 全局搜索
- 局部搜索

#### RAG调优
**1. 混合检索的适用场景及具体方法**
- 为什么需要混合检索
- 常见的混合检索方法
- 关键词（BM25）+ 向量检索（Embedding）
- 多路召回（分别检索）
- 引入重排序（Rerank）提升最终质量

**2. RAG系统的调试步骤及效果评估**
- 数据准备
- 检索阶段
- 生成阶段
- RAG效果评估：如何量化好与坏
- 持续优化的评估实践

### 项目实战
#### 企业知识库（企业RAG大赛冠军项目）
**1. 企业RAG大赛：搭建RAG知识库**
- RAG冠军方案（多路由+动态知识库）
- RAG比赛任务说明
- 基础RAG系统流程
- 解析模块、Docling优化、表格序列化
- 内容提取（ingestion）
- 检索（Retrieval）
- LLM重排序 (LLM reranking)
- 父页面检索
- 整合后的检索器
- 增强 (Augmentation)
- 生成 (Generation)
- 思维链、结构化输出、思维链+结构化输出
- 指令细化 (Instruction Refinement)
- 提示词创建、Prompt.py实现
- RAG系统调参

**2. 搭建自己的RAG系统**
- 选择适合的LLM和Embedding模型
- MinerU使用
- 更新中文知识库、设置相关的问题清单
- 针对开放式的问题，进行Prompt设置
- 搭建前端页面，比如使用streamlit

### 专项求职辅导
#### RAG相关简历+面试问题辅导
- 你能谈谈大模型应用开发的三种主要模式吗？
- 文档分块有哪些策略？你为什么选择用这个策略？
- 你能画一下RAG的系统架构图吗？并解释一下关键步骤。
- 文档分块有哪些策略？你为什么选择用这个策略？
- 你项目中用了哪个Embedding模型？为什么选它？
- 如果你的RAG效果很差，你会从哪几个方面去调试？
- 当用户的问题很模糊，或者依赖上一轮对话时，RAG怎么优化？
- 你只用了向量检索吗？它有什么缺点？什么是混合检索？
- 检索召回了20条文档，你怎么确保喂给LLM的是最好的3条？
- 系统上线后，你怎么维护和迭代你的知识库？
- 你如何评估一个RAG系统的好坏？

---

## 模块3：Agent

### 理论知识+案例详解+实操
#### Function Calling与MCP
**1. 使用Function Calling进行工具调用**
- 什么是Function Calling？
- Function Calling与MCP的区别
- 使用Qwen3完成天气调用Function Calling
- Qwen-Agent中的Function Calling
- 使用Function Calling完成数据库查询

**2. MCP与A2A的应用**
- MCP的核心概念 (MCP Host，MCP Client，MCP Server）
- MCP的使用场景
- CASE：旅游攻略MCP
- CASE：Fetch网页内容抓取
- CASE：Bing中文搜索
- CASE：搭建你的MCP服务
- 什么是Agent2Agent
- A2A与MCP的关系

#### 构建Agent的数据决策能力
**1. AI大赛：二手车价格预测**
- 使用Agent/LLM对二手车价格进行预测
- 为什么Agent的数据决策要用专业工具
- 分析式AI与生成式AI
- 十大经典机器学习算法

**2. 挑战Baseline**
- 数据探索
- 特征选择
- 模型训练与预测
- 特征工程

#### 构建Agent的搜索、感知与记忆能力
**1. Agent的信息抓取及搜索能力构建**
- RAG（检索增强生成）
- Qwen-Agent中的RAG能力
- Web搜索能力

**2. Agent Memory能力开发**
- 短时记忆（Short-term Memory）
- 长时记忆（Long-term Memory）
- 构建记忆流（Memory Stream）

#### Agent的能力优化与效果评估
**1. 使用用户使用数据提升Agent能力**
- 显式反馈
- 隐式反馈
- 利用反馈进行RAG或微调

**2. Agent智能体效果评估**
- RAG能力评估（大海捞针）
- 多跳推理评估（Multi-Hop）
- 业务指标评估

### 项目实战
#### OpenManus开发实战
**1. 深度解析OpenManus框架**
- OpenManus项目导论
- 核心流程：一次“手稿生成”的完整生命周期
- 核心模块Orchestrator
- 核心模块Agents
- 核心模块Memory
- 核心模块Tools
- 核心机制提示词工程

**2. 构建自己的AI写作助手**
- 本地运行OpenManus
- Agents的“角色”定制
- 集成企业知识库RAG
- 中文写作流

### 专项求职辅导
#### Agent相关简历+面试问题辅导
- 一分钟讲清楚Agent的定义
- 你如何处理Agent的幻觉问题？
- 在你的项目中，Agent的‘状态’是如何管理的？
- 你如何平衡Agent的自主性与可控性？
- 介绍一下你最复杂的那个Agent项目
- 如何回答“RAG的效果如何评估？”
- 为什么用LangGraph
- 如何处理Text-to-SQL
- 如何搭建一个RAG Agent，实现对本地知识库（如PDF）的问答
- 如何定义一个自定义工具
- 处理多文件时，如何解决上下文长度限制问题
- 如何评估检索效果
- 如何收集用户使用数据
- Agent设计哲学

---

## 模块4: 开发框架

### 理论知识+案例详解+实操
#### LangChain：多任务应用开发
**1. LangChain多任务应用开发**
- Models, Prompts, Memory, Indexes, Chains, Agents
- LangChain中的tools (serpapi, llm-math)
- LangChain中的Memory
- LECL构建任务链

**2. LangChain开发实操详解**
- CASE：动手搭建本地知识智能客服（理解ReAct）
- CASE：工具链组合设计（LangChain Agent）
- CASE：搭建故障诊断Agent（LangChain Agent）
- CASE：工具链组合设计（LCEL）

#### AI框架设计与选型
**1. 自研框架设计思路**
- 核心组件抽象：如何设计模块化、插件化与可扩展的框架结构
- 数据流与控制流：定义Agent、Tools与Models的标准化交互管线
- 状态管理与性能考量：异步处理、缓存机制与日志监控的设计

**2. 优秀开源开发框架详解**
- LlamaIndex深度解析
- AutoGen多智能体框架
- 框架选型对比：LangChain vs. LlamaIndex vs. AutoGen的适用场景与优劣

#### HuggingFace生态实战：从模型应用到高效微调
**1. HuggingFace模型库的使用**
- 核心组件（Transformers, Datasets, Tokenizers）
- Pipelines API：零代码/少代码调用模型的快捷方式
- Model Hub与Dataset Hub：模型与数据集的搜索、下载与版本控制

**2. 使用HuggingFace做模型微调**
- Trainer API：标准化的模型训练与评估高阶接口
- PEFT高效微调：LoRA与QLoRA的原理与代码实战
- TRLx：使用强化学习（RLHF/PPO）对齐语言模型

#### 神经网络基础与Tensorflow实战
**1. 神经网络基础**
- 神经网络结构
- 激活函数
- 损失函数
- 反向传播
- 梯度下降
- 优化方法（SGD、Adam）
- 使用numpy搭建神经网络

**2. TensorFlow实战**
- TensorFlow中的计算图与会话管理
- 分布式训练与模型并行
- TensorFlow Serving部署与推理
- 使用Keras构建简单神经网络
- 使用Keras进行二手车价格预测

#### Pytorch与视觉检测
**1. PyTorch的核心概念**
- PyTorch的张量与自动求导机制
- PyTorch的动态图与静态图

**2. PyTorch的分布式训练**
- 在多个GPU上进行训练
- 使用PyTorch Lightning简化模型训练

**3. 图像识别技术与缺陷检测**
- 传统图像识别模型
- 视觉检测方法
- 缺陷检测方案
- 卷积网络可视化

**4. 视觉检测模型YOLO**
- 从Yolov1到Yolov12
- ultralytics：基于pytorch的视觉检测工具
- Project：钢铁表面缺陷检测

### 专项求职辅导
#### 开发框架相关简历+面试问题辅导
- LangChain解决了什么问题？
- LangChain六大核心组件的职责与交互关系是什么
- 如何使用LangChain构建一个RAG系统
- LangChain中的Memory机制是如何实现的？
- LangChain vs. LlamaIndex：二者的核心定位有何不同？
- 什么时候应该选择LangChain，什么时候应该选择LlamaIndex？
- 如何看待AutoGen等多智能体框架？它与单Agent框架的核心区别是什么？
- 如果让你从零设计一个LLM应用框架，你会如何进行组件抽象？
- 如何设计一个可插拔的Tool/Plugin系统？
- HuggingFace Pipelines API的优势和局限性是什么？
- Tokenizers的作用是什么？为什么模型和Tokenizer必须匹配？
- 什么是全参数微调（Full Fine-Tuning）？它有什么缺点？
- 什么是PEFT（高效微调）？
- PyTorch的动态图机制（Eager Execution）是什么？它与静态图有何区别？
- TensorFlow 2.x的Keras API与TF 1.x的Session/Graph模式有何不同？
- 在LLM时代，你如何看待TensorFlow和PyTorch这两个框架的优劣和生态？

---

## 模块5: 模型训练与微调

### 理论知识+案例详解+实操
#### LLM微调原理
**1. LLM的微调原理**
- 高效微调的方法
- LoRA的数学原理
- LoRA算法的核心假设
- 矩阵分解与猜你喜欢
- SVD矩阵分解

**2. LLM微调的数据处理与显存评估方法**
- 微调数据准备
- 数据质量与数量要求
- 不同模型尺寸与场景的数据需求
- 硬件需求与显存计算
- 微调显存估算方法
- LoRA显存优化与计算示例
- 微调后的模型评估

#### 高质量微调数据工程与评估
**1. 微调数据的收集、清洗、标准**
- 数据收集策略
- 数据清洗核心流程
- 数据标注规范
- SFT vs. RLHF的数据差异

**2. 衡量数据质量和微调结果的关系**
- Garbage In, Garbage Out
- 自动化评估指标
- 基准测试（Benchmark）评估
- 人工评估的必要性

#### LLM模型蒸馏与微调实操
**1. LLM的模型蒸馏**
- 知识蒸馏的核心思想
- 蒸馏的目的与价值
- 经典蒸馏方法
- LLM时代的蒸馏挑战

**2. LLM模型微调与蒸馏实操**
- unsloth框架使用
- 教师-学生模型的选择
- 任务蒸馏演练
- 评估蒸馏效果

#### 视觉与多模态模型
**1. 多模态模型与视觉识别模型**
- 视觉识别（CV）的三大核心任务
- VLM在行业中的应用
- 视频理解SOTA
- 智能文档模型MinerU

**2. 训练Yolo目标检测模型**
- YOLO（You Only Look Once）的核心原理
- 目标检测数据集的准备与标注
- 训练自定义YOLO模型
- 评估模型性能指标

### 项目实战
#### AI质检
- AI在工业质检的价值
- 技术选型对决：YOLO vs. Qwen-VL
- 数据集分析 (EDA)：缺陷类别、图像特点、标注格式。
- 环境准备：Python, PyTorch, Ultralytics (YOLO), Transformers (Qwen-VL)
- 数据工程：从原始数据到YOLO训练集
- YOLO训练与调优
- 模型评估与缺陷分析
- Qwen-VL多模态大模型的探索性测试
- 使用Qwen-VL进行“零样本”缺陷检测
- 结果分析与局限性探讨
- YOLO与Qwen-VL的终极对决

### 专项求职辅导
#### 模型训练与微调相关简历+面试问题辅导
- 谈谈你对预训练和微调的理解
- 什么是全参数微调
- 为什么需要PEFT
- 请详细讲讲LoRA的原理
- 如果要你微调一个模型，你的技术选型是什么？
- 你的微调数据是怎么来的？是开源的、业务的、还是合成的？
- 你认为一条‘高质量’的微调数据应该符合什么标准？
- SFT（指令微调）的数据格式是怎么构建的？
- 你如何评估你微调后的模型效果？
- 自动化评估 (如BLEU, ROUGE) 和人工评估，你倾向用哪个？
- 你知道有哪些评估LLM的Benchmark吗？
- 为什么要用模型蒸馏？它的核心思想是什么？
- 你如何评估蒸馏的效果？
- 什么是多模态模型？
- CV的经典三大任务（分类、检测、分割）和多模态VQA（视觉问答）有什么不同？
- YOLO的数据集是怎么标注和准备的？
- 对于一个工业质检任务，你认为用YOLO和用Qwen-VL，它们各自的优缺点是什么？

---

## 模块6: 模型部署及高并发

### 理论知识+案例详解+实操
#### 企业级AI部署：从硬件选型到框架选择
**1. 企业级AI硬件的选型，规划与优化**
- GPU选型策略：对比H100, A100, L40S, 4090等主流GPU
- CPU与内存配比：分析CPU对Tokenization预处理、请求调度的影响，以及内存大小如何限制并发处理能力。
- 网络基础设施：探讨RoCE (RDMA) vs. TCP/IP在多节点/多卡（如张量并行）推理中的延迟和带宽瓶颈。
- 服务器与集群规划

**2. 部署框架Ollama, vLLM, SGLang的特点**
- Ollama：易用性、本地化部署优势
- vLLM：高吞吐推理引擎的核心优势（PagedAttention）
- SGLang：在复杂控制流（如RAG、CoT、Agents）任务中的高性能特性，及其与vLLM的差异。
- 选型对比

#### AI服务核心：高并发原理与性能监控调优
**1. 高并发的AI服务原理（KV Cache，vLLM及PagedAttention原理）**
- KV Cache瓶颈
- PagedAttention核心思想
- vLLM的实现
- Continuous Batching (连续批处理)
- vLLM如何利用PagedAttention实现请求的动态插入和退出，极大提升GPU利用率

**2. AI服务的性能监测及调优**
- 关键性能指标
- 监控工具栈
- vLLM性能调优
- 瓶颈定位：分析和识别推理服务中的常见瓶颈

#### SGLang深度优化：Radix缓存与复杂任务的极致吞吐
**1. SGLang的缓存优化及Radix Tree原理及运行时高级抽象**
- Radix Tree (基数树)原理
- SGLang的RadixAttention
- 运行时高级抽象 (Frontend/Backend)
- SGLang IR (中间表示)

**2. SGLang对于复杂AI任务中的极致延迟及吞吐优化**
- Token Healing技术
- 复杂控制流优化
- 多租户与智能调度
- SGLang与vLLM在执行中的差异

---

## 模块7: 低代码平台

### 理论知识+案例详解+实操
#### Coze工作原理与应用实例
**1. Coze工作原理与实例**
- 插件使用
- 工作流使用
- RAG知识库
- CASE：AI新闻Agent
- CASE：创建搜索新闻工作流
- CASE：weather_news工作流（基于意图识别）
- CASE：LLM联网搜索
- CASE：搭建古诗词Agent

**2. Coze的本地化部署及运行**
- 环境准备：Docker与基础环境配置
- Coze-studio的下载、配置与启动
- 接入本地大模型
- 本地部署Agent的调用

#### Dify本地化部署和应用
**1. Dify本地化部署**
- Dify开发平台
- Docker Compose部署
- 克隆Dify代码仓库
- 启动Dify服务
- 访问Dify
- 如何使用Dify

**2. Dify应用实战**
- CASE：LLM联网搜索
- CASE：搭建古诗词WorkFlow
- CASE：智能客服ChatFlow
- CASE：智能文档分析助手(MinerU+Dify)
- 如何应用Agent API（Coze, Dify）
- Coze API使用
- Cozepy工具
- Dify API使用

#### Agent调试、运维与系统集成
**1. 低代码Agent的调试、发布与调用**
- Agent工作流调试：单步执行与日志分析
- Agent的发布流程与版本管理
- Coze API调用实战（Cozepy工具使用）
- Dify API调用实战（及Python SDK使用）

**2. 低代码平台与企业内部系统的集成**
- 集成策略：API封装 vs. 数据库直连
- 将内部系统（如CRM）封装为Coze/Dify的插件（Tool）
- 通过工作流节点（如HTTP请求）调用企业内部API
- CASE：实现Agent查询内部订单系统并自动答复

---

## 模块8: 研发工程提效

### 理论知识+案例详解+实操
#### 智能编码革命：AI辅助编程
**1. AI辅助编程与Vibe Coding**
- AI辅助编程的核心理念
- Vibe Coding详解：什么是心流开发
- 如何通过AI高效完成代码生成、代码补全、代码审查与重构。
- CASE：多张Excel报表处理
- CASE：疫情实时监控大屏

**2. 国内外AI编程工具对比**
- 主流工具解析（Cursor）
- 国内外工具横向对比：GitHub Copilot, Cursor, Trae, CodeBuddy, 通义灵码
- 选型指南与最佳实践

#### AI赋能的智能测试与质量保障
**1. AI用例生成与智能测试**
- AI在软件测试领域的核心应用（用例生成、缺陷定位、回归测试等）。
- 基于需求文档（PRD）的用例生成
- 基于代码的单元测试生成
- UI自动化测试的AI演进

**2. AI赋能的测试流程闭环优化**
- AI驱动的缺陷管理
- 精准回归测试策略
- AI模拟用户异常行为，发现边缘缺陷。
- 构建AI驱动的CI/CD质量门禁

#### 从Text-to-SQL到数据智能
**1. Text-to-Sql智能查询引擎**
- Text-to-SQL技术原理
- 构建SQL Copilot的关键挑战
- CASE：搭建金融业务场景的SQL Copilot

**2. AI驱动的数据清洗与分析自动化**
- AI在数据清洗中的应用
- 自动化探索性数据分析（Auto-EDA）
- 数据报告与BI自动化

### 项目实战
#### ChatBI开发实战
- ChatBI的挑战与机遇
- 传统BI vs. 智能BI
- Vanna核心工作流
- 环境搭建与项目初始化
- Vanna的“知识”从哪里来
- 知识的存储与管理
- vn.ask()背后的RAG流程拆解
- 检索：Vanna如何找到相关知识
- 生成：LLM的SQL生成艺术
- SQL验证与修正
- 开发你的vanna应用
- 集成自定义大模型
- 使用本地向量存储
- 前端界面优化

### 专项求职辅导
#### 部署及提效相关简历+面试问题辅导
- 如何设计一个“够用”的AI硬件集群？
- “训练集群”和“推理集群”的硬件配置有何根本不同？
- 你用过哪些部署框架？它们各自解决了什么问题？优劣势是什么？
- LLM推理的主要瓶颈是什么？
- 什么是KV Cache
- 你的服务TFLOPS很高，但吞吐量上不去，可能是什么原因？
- 连续批处理（Continuous Batching）的原理是什么
- SGLang为什么在处理多轮对话或复杂Prompt时比vLLM更快
- Coze和Dify的架构有何不同？在企业中落地，你倾向于哪个？
- Cursor, Trae, 灵码这些工具有什么区别？
- 你如何使用AI编程工具提升编码效率？
- 如何利用AI优化测试流程？你有哪些实践经验？
- Text-to-SQL的核心难点是什么？
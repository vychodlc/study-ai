# 模块2: RAG技术与应用

## 课程概述

本模块深入讲解RAG（检索增强生成）技术，从Embeddings和向量数据库的基础知识开始，逐步深入到RAG的核心原理、多模态数据处理、调优策略，最后通过企业知识库项目实战，帮助学员掌握构建生产级RAG系统的完整能力。

---

## 第一章：Embeddings和向量数据库

### 1.1 Embeddings模型及向量化

#### 1.1.1 什么是Embedding

**Embedding的定义**

Embedding（嵌入）是一种将高维离散数据（如文本、图像）映射到低维连续向量空间的技术。通过Embedding，我们可以将语义信息编码为数学向量，使得计算机能够理解和处理语义相似性。

```
文本 → Embedding模型 → 向量（数组）

示例：
"人工智能" → [0.23, -0.45, 0.12, ..., 0.67]  # 768维或1024维向量
"机器学习" → [0.21, -0.42, 0.15, ..., 0.63]  # 语义相近，向量也相近
"今天天气很好" → [0.87, 0.12, -0.34, ..., 0.02]  # 语义不同，向量距离远
```

**Embedding的核心价值：**

| 价值 | 说明 |
|------|------|
| 语义表示 | 将文本的语义信息编码为数值向量 |
| 相似度计算 | 通过向量距离衡量语义相似性 |
| 降维压缩 | 将高维稀疏表示转为低维稠密表示 |
| 跨模态对齐 | 不同模态数据可以映射到同一空间 |

#### 1.1.2 Word Embedding的演进

**词嵌入技术的发展历程：**

```
┌─────────────────────────────────────────────────────────────┐
│                  词嵌入技术演进                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  One-Hot编码（传统方法）                                     │
│  └── 问题：维度灾难，无法表示语义相似性                        │
│                                                             │
│         ↓                                                   │
│                                                             │
│  Word2Vec (2013, Google)                                    │
│  ├── CBOW：根据上下文预测中心词                              │
│  ├── Skip-gram：根据中心词预测上下文                         │
│  └── 特点：静态词向量，无法处理一词多义                        │
│                                                             │
│         ↓                                                   │
│                                                             │
│  GloVe (2014, Stanford)                                     │
│  └── 结合全局统计信息和局部上下文                             │
│                                                             │
│         ↓                                                   │
│                                                             │
│  ELMo (2018, Allen AI)                                      │
│  └── 基于BiLSTM的上下文相关词向量                            │
│                                                             │
│         ↓                                                   │
│                                                             │
│  BERT Embeddings (2018, Google)                             │
│  └── 基于Transformer的双向上下文编码                         │
│                                                             │
│         ↓                                                   │
│                                                             │
│  Sentence Transformers (2019+)                              │
│  └── 专门优化的句子/文档级别Embedding                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**现代文本Embedding的特点：**

```python
from sentence_transformers import SentenceTransformer

# 加载预训练的Embedding模型
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# 生成句子级别的Embedding
sentences = [
    "机器学习是人工智能的一个分支",
    "深度学习需要大量的训练数据",
    "今天的股市行情很好"
]

# 获取向量表示
embeddings = model.encode(sentences)
print(f"向量维度: {embeddings.shape}")  # (3, 1024)
```

#### 1.1.3 余弦相似度计算

**余弦相似度公式：**

```
                    A · B           Σ(ai × bi)
cos(θ) = ──────────────────── = ─────────────────────
              ||A|| × ||B||      √Σai² × √Σbi²

范围：[-1, 1]
- 1：完全相同方向（最相似）
- 0：正交（不相关）
- -1：完全相反方向（最不相似）
```

**Python实现：**

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# 使用示例
embedding1 = model.encode("人工智能改变世界")
embedding2 = model.encode("AI技术正在改变我们的生活")
embedding3 = model.encode("今天吃什么")

sim_1_2 = calculate_similarity(embedding1, embedding2)
sim_1_3 = calculate_similarity(embedding1, embedding3)

print(f"语义相近的句子相似度: {sim_1_2:.4f}")  # 约 0.85
print(f"语义不同的句子相似度: {sim_1_3:.4f}")  # 约 0.15

# 批量计算相似度矩阵
all_embeddings = model.encode(["句子1", "句子2", "句子3"])
similarity_matrix = cosine_similarity(all_embeddings)
```

#### 1.1.4 Embedding模型的选择

**MTEB排行榜（Massive Text Embedding Benchmark）**

MTEB是评估文本Embedding模型最权威的基准测试，包含多个任务维度：

```
MTEB评估维度：
├── 检索（Retrieval）：文档检索能力
├── 重排序（Reranking）：结果排序能力
├── 聚类（Clustering）：语义聚类能力
├── 分类（Classification）：文本分类能力
├── 语义相似度（STS）：句子相似度评估
├── 配对分类（PairClassification）：句对关系判断
└── 摘要检索（Summarization）：摘要匹配能力
```

**主流中文Embedding模型对比：**

| 模型 | 维度 | 最大长度 | 特点 | 适用场景 |
|------|------|----------|------|----------|
| bge-large-zh-v1.5 | 1024 | 512 | MTEB中文榜首，综合性能优秀 | 通用场景 |
| m3e-large | 1024 | 512 | 中文优化，性价比高 | 中文RAG |
| text2vec-large | 1024 | 512 | 开源友好 | 研究实验 |
| text-embedding-ada-002 | 1536 | 8191 | OpenAI模型，长文本支持 | 多语言场景 |
| Cohere embed-v3 | 1024 | 512 | 多语言支持 | 跨语言检索 |

#### 1.1.5 向量维度对模型性能的影响

**维度选择的权衡：**

```
┌─────────────────────────────────────────────────────────────┐
│                   向量维度的影响                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  高维度（如1024、1536）                                      │
│  ├── 优点：表达能力强，语义区分度高                           │
│  ├── 缺点：存储成本高，检索速度慢                             │
│  └── 适用：高精度要求的场景                                  │
│                                                             │
│  低维度（如256、384）                                        │
│  ├── 优点：存储小，检索快                                    │
│  ├── 缺点：可能损失语义信息                                  │
│  └── 适用：大规模、低延迟场景                                │
│                                                             │
│  存储计算示例：                                              │
│  1000万文档 × 1024维 × 4字节(float32) ≈ 40GB                │
│  1000万文档 × 256维 × 4字节(float32) ≈ 10GB                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 1.1.6 俄罗斯套娃表示学习（Matryoshka Representation Learning）

**什么是MRL？**

MRL是一种创新的Embedding训练方法，使单个模型能够生成多种维度的向量，且较低维度的向量保留了核心语义信息。

```
┌─────────────────────────────────────────────────────────────┐
│                   俄罗斯套娃向量                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  完整向量 [1024维]: ████████████████████████████████        │
│                                                             │
│  截断到512维:      ████████████████                         │
│                    保留约98%的语义信息                       │
│                                                             │
│  截断到256维:      ████████                                 │
│                    保留约95%的语义信息                       │
│                                                             │
│  截断到64维:       ██                                       │
│                    保留约85%的语义信息                       │
│                                                             │
│  应用：可根据场景选择合适维度，平衡精度与效率                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**使用MRL模型：**

```python
from sentence_transformers import SentenceTransformer

# 支持MRL的模型
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5')

sentences = ["这是一个测试句子"]

# 获取不同维度的向量
embedding_1024 = model.encode(sentences)  # 完整1024维
embedding_256 = model.encode(sentences)[:, :256]  # 截断到256维

# 两者都可以用于检索，256维版本更快但精度略低
```

#### 1.1.7 如何选择适合的Embedding模型

**选择决策流程：**

```
开始
  │
  ├─→ 语言要求？
  │     ├── 纯中文 → bge-large-zh / m3e-large
  │     ├── 纯英文 → bge-large-en / e5-large
  │     └── 多语言 → multilingual-e5 / Cohere embed
  │
  ├─→ 文本长度？
  │     ├── 短文本（<512） → 常规模型
  │     └── 长文本（>512） → 长上下文模型（jina-embeddings-v2）
  │
  ├─→ 部署环境？
  │     ├── 云端API → OpenAI / Cohere / 智谱
  │     └── 本地部署 → 开源模型（bge / m3e）
  │
  ├─→ 性能要求？
  │     ├── 高精度优先 → 大模型（large）
  │     └── 速度优先 → 小模型（base / small）
  │
  └─→ 特殊需求？
        ├── 代码检索 → CodeBERT / UniXcoder
        └── 专业领域 → 领域微调模型
```

---

### 1.2 向量数据库和向量检索

#### 1.2.1 什么是向量数据库

**向量数据库的定义：**

向量数据库是专门设计用于存储、索引和查询高维向量数据的数据库系统。它支持基于向量相似度的快速检索，是RAG系统的核心基础设施。

```
┌─────────────────────────────────────────────────────────────┐
│                    向量数据库架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  向量存储   │    │   索引层    │    │  元数据存储  │     │
│  │  (Vectors)  │ ←→ │  (Index)   │ ←→ │ (Metadata)  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         ↑                  ↑                  ↑            │
│         └──────────────────┼──────────────────┘            │
│                            │                               │
│                     ┌──────────────┐                       │
│                     │   查询引擎    │                       │
│                     │ (Query Engine)│                       │
│                     └──────────────┘                       │
│                            ↑                               │
│                     ┌──────────────┐                       │
│                     │    API层     │                       │
│                     └──────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 1.2.2 主流向量数据库对比

**FAISS**

```python
# FAISS - Facebook AI相似度搜索
# 特点：高性能、纯内存、支持GPU加速

import faiss
import numpy as np

# 创建索引
dimension = 1024
index = faiss.IndexFlatL2(dimension)  # L2距离

# 添加向量
vectors = np.random.random((10000, dimension)).astype('float32')
index.add(vectors)

# 搜索
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=5)

# 保存和加载
faiss.write_index(index, "index.faiss")
index = faiss.read_index("index.faiss")
```

**Milvus**

```python
# Milvus - 云原生分布式向量数据库
# 特点：可扩展、高可用、支持多种索引

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# 连接
connections.connect("default", host="localhost", port="19530")

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
]
schema = CollectionSchema(fields, "文档集合")

# 创建集合
collection = Collection("documents", schema)

# 创建索引
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index("embedding", index_params)

# 插入数据
data = [
    [1, 2, 3],  # ids
    [[0.1]*1024, [0.2]*1024, [0.3]*1024],  # embeddings
    ["文本1", "文本2", "文本3"]  # texts
]
collection.insert(data)

# 搜索
collection.load()
results = collection.search(
    data=[[0.1]*1024],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=5
)
```

**Elasticsearch向量搜索**

```python
# Elasticsearch 8.x 内置向量搜索
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

# 创建索引
index_mapping = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}
es.indices.create(index="documents", body=index_mapping)

# 向量搜索
query = {
    "knn": {
        "field": "embedding",
        "query_vector": [0.1] * 1024,
        "k": 5,
        "num_candidates": 100
    }
}
results = es.search(index="documents", body=query)
```

**主流向量数据库对比表：**

| 特性 | FAISS | Milvus | Pinecone | Chroma | Elasticsearch |
|------|-------|--------|----------|--------|---------------|
| 部署方式 | 库 | 自托管/云 | 全托管 | 自托管 | 自托管/云 |
| 分布式 | ❌ | ✅ | ✅ | ❌ | ✅ |
| 元数据过滤 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 混合搜索 | ❌ | ✅ | ✅ | ✅ | ✅ |
| GPU加速 | ✅ | ✅ | - | ❌ | ❌ |
| 学习曲线 | 中 | 高 | 低 | 低 | 中 |
| 适用规模 | 中小 | 大规模 | 大规模 | 小规模 | 中大规模 |

#### 1.2.3 向量数据库与传统数据库的区别

```
┌─────────────────────────────────────────────────────────────┐
│              向量数据库 vs 传统数据库                         │
├─────────────────┬─────────────────┬─────────────────────────┤
│     维度        │   传统数据库     │      向量数据库          │
├─────────────────┼─────────────────┼─────────────────────────┤
│  数据模型       │ 行/列/文档      │ 高维向量 + 元数据        │
│  查询方式       │ 精确匹配/范围    │ 相似度搜索              │
│  索引类型       │ B-tree/Hash    │ HNSW/IVF/PQ            │
│  返回结果       │ 精确匹配        │ Top-K近似结果           │
│  典型查询       │ WHERE id = 1   │ 找最相似的5个向量        │
│  数据类型       │ 结构化数据      │ 非结构化数据的向量表示   │
│  主要用途       │ 事务处理        │ 语义搜索/推荐系统       │
└─────────────────┴─────────────────┴─────────────────────────┘
```

#### 1.2.4 向量索引类型详解

**常用索引类型：**

```
┌─────────────────────────────────────────────────────────────┐
│                    向量索引类型                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Flat（暴力搜索）                                         │
│     ├── 原理：遍历所有向量计算距离                            │
│     ├── 优点：100%召回率，结果精确                           │
│     ├── 缺点：搜索慢，O(n)复杂度                             │
│     └── 适用：小规模数据（<10万）                            │
│                                                             │
│  2. IVF（倒排文件索引）                                      │
│     ├── 原理：先聚类，搜索时只遍历相关簇                      │
│     ├── 优点：速度快，可调节精度                             │
│     ├── 缺点：需要训练，可能丢失结果                          │
│     ├── 参数：nlist(簇数量), nprobe(搜索簇数)                │
│     └── 适用：中等规模（10万-1000万）                        │
│                                                             │
│  3. HNSW（分层可导航小世界图）                                │
│     ├── 原理：构建多层图结构，贪婪搜索                        │
│     ├── 优点：高召回率，搜索快                               │
│     ├── 缺点：内存占用大，构建慢                             │
│     ├── 参数：M(连接数), efConstruction(构建质量)            │
│     └── 适用：对召回率要求高的场景                           │
│                                                             │
│  4. PQ（乘积量化）                                           │
│     ├── 原理：向量压缩，用短编码近似原始向量                   │
│     ├── 优点：大幅降低内存占用                               │
│     ├── 缺点：精度损失                                      │
│     └── 适用：超大规模、内存受限                             │
│                                                             │
│  5. IVF_PQ（组合索引）                                       │
│     ├── 原理：IVF + PQ的组合                                │
│     └── 适用：超大规模数据的平衡方案                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**FAISS索引选择示例：**

```python
import faiss

dimension = 1024
n_vectors = 1000000

# 1. Flat索引（精确）
index_flat = faiss.IndexFlatL2(dimension)

# 2. IVF索引（快速）
nlist = 100  # 聚类数量
quantizer = faiss.IndexFlatL2(dimension)
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index_ivf.train(training_vectors)  # 需要训练
index_ivf.nprobe = 10  # 搜索时检查的簇数量

# 3. HNSW索引（高召回）
index_hnsw = faiss.IndexHNSWFlat(dimension, 32)  # M=32

# 4. IVF_PQ索引（压缩）
m = 8  # 子向量数量
index_ivfpq = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
index_ivfpq.train(training_vectors)
```

---

## 第二章：RAG技术与应用

### 2.1 RAG检索增强生成的原理及流程

#### 2.1.1 大模型开发的三种范式

```
┌─────────────────────────────────────────────────────────────┐
│                  大模型应用开发范式                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  范式1：直接提示（Prompting）                                │
│  ├── 方式：设计好Prompt，直接调用模型                         │
│  ├── 优点：简单快速，无需额外数据                             │
│  ├── 缺点：受限于模型知识，无法更新                           │
│  └── 适用：简单任务，模型已知领域                             │
│                                                             │
│  范式2：RAG（检索增强生成）                                   │
│  ├── 方式：检索相关知识 + 模型生成                            │
│  ├── 优点：知识可更新，可追溯，成本低                         │
│  ├── 缺点：依赖检索质量，上下文有限                           │
│  └── 适用：知识密集型任务，企业知识库                         │
│                                                             │
│  范式3：微调（Fine-tuning）                                  │
│  ├── 方式：用特定数据训练模型                                │
│  ├── 优点：深度定制，风格一致                                │
│  ├── 缺点：成本高，需要大量数据，难以更新                      │
│  └── 适用：特定领域/风格，稳定任务                            │
│                                                             │
│  组合使用：RAG + 微调 = 最佳效果                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.1.2 RAG的核心原理与流程

**Native RAG架构：**

```
┌─────────────────────────────────────────────────────────────┐
│                    Native RAG 流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    离线索引阶段                      │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │                                                     │    │
│  │  文档 → 加载 → 分块 → Embedding → 存入向量库          │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    在线查询阶段                      │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │                                                     │    │
│  │  用户问题                                           │    │
│  │      ↓                                              │    │
│  │  Query Embedding                                    │    │
│  │      ↓                                              │    │
│  │  向量检索 → Top-K文档块                              │    │
│  │      ↓                                              │    │
│  │  构建Prompt = 系统提示 + 检索文档 + 用户问题          │    │
│  │      ↓                                              │    │
│  │  LLM生成回答                                        │    │
│  │      ↓                                              │    │
│  │  返回用户                                           │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.1.3 CASE：DeepSeek + FAISS搭建本地知识库检索

**完整实现代码：**

```python
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class SimpleRAG:
    def __init__(self, embedding_model_name="BAAI/bge-large-zh-v1.5"):
        # 初始化Embedding模型
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = 1024

        # 初始化FAISS索引
        self.index = faiss.IndexFlatIP(self.dimension)  # 内积相似度

        # 存储文档原文
        self.documents = []

        # 初始化DeepSeek客户端
        self.llm_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )

    def add_documents(self, documents: list):
        """添加文档到知识库"""
        # 生成向量
        embeddings = self.embedding_model.encode(
            documents,
            normalize_embeddings=True
        )

        # 添加到索引
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)

        print(f"已添加 {len(documents)} 个文档，总计 {len(self.documents)} 个")

    def retrieve(self, query: str, top_k: int = 3):
        """检索相关文档"""
        # 查询向量化
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        ).astype('float32')

        # 检索
        scores, indices = self.index.search(query_embedding, top_k)

        # 返回结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "score": float(scores[0][i])
                })

        return results

    def generate(self, query: str, context: str) -> str:
        """调用LLM生成回答"""
        prompt = f"""你是一个专业的知识问答助手。请根据以下参考资料回答用户的问题。

## 参考资料
{context}

## 回答要求
1. 仅基于参考资料中的信息回答
2. 如果资料中没有相关信息，请明确说明"根据提供的资料无法回答"
3. 回答要准确、简洁

## 用户问题
{query}

## 你的回答："""

        response = self.llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        return response.choices[0].message.content

    def query(self, question: str, top_k: int = 3) -> str:
        """完整的RAG查询流程"""
        # 1. 检索相关文档
        retrieved = self.retrieve(question, top_k)

        # 2. 构建上下文
        context = "\n\n".join([
            f"[文档{i+1}] (相关度: {r['score']:.2f})\n{r['document']}"
            for i, r in enumerate(retrieved)
        ])

        # 3. 生成回答
        answer = self.generate(question, context)

        return answer


# 使用示例
if __name__ == "__main__":
    # 初始化RAG系统
    rag = SimpleRAG()

    # 添加示例文档
    documents = [
        "Python是一种高级编程语言，由Guido van Rossum于1989年发明。它以简洁清晰的语法著称。",
        "机器学习是人工智能的一个分支，它使计算机能够从数据中学习而无需明确编程。",
        "深度学习是机器学习的子集，使用多层神经网络来处理复杂的模式识别任务。",
        "RAG（检索增强生成）是一种结合信息检索和文本生成的技术，可以提高大模型的准确性。",
        "FAISS是Facebook开源的高效相似度搜索库，广泛用于向量检索场景。",
    ]
    rag.add_documents(documents)

    # 查询
    question = "什么是RAG技术？"
    answer = rag.query(question)
    print(f"问题: {question}")
    print(f"回答: {answer}")
```

---

### 2.2 Query改写与知识库处理

#### 2.2.1 Query改写

**为什么需要Query改写？**

用户的原始查询往往不够精确或完整，通过改写可以提高检索效果。

```
┌─────────────────────────────────────────────────────────────┐
│                    Query改写策略                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Query扩展（Query Expansion）                            │
│     原始："RAG效果不好"                                      │
│     扩展："RAG检索效果差 检索增强生成优化 RAG调优方法"         │
│                                                             │
│  2. Query分解（Query Decomposition）                        │
│     原始："比较FAISS和Milvus的性能和易用性"                   │
│     分解：                                                   │
│     - "FAISS的性能特点是什么"                                │
│     - "Milvus的性能特点是什么"                               │
│     - "FAISS的使用难度如何"                                  │
│     - "Milvus的使用难度如何"                                 │
│                                                             │
│  3. HyDE（假设性文档嵌入）                                   │
│     原始查询 → LLM生成假设答案 → 用假设答案检索               │
│     优点：假设答案与真实文档更相似                            │
│                                                             │
│  4. 多轮对话Query改写                                        │
│     上文："介绍一下FAISS"                                    │
│     当前："它支持GPU吗"                                      │
│     改写："FAISS支持GPU加速吗"                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Query改写实现：**

```python
class QueryRewriter:
    def __init__(self, llm_client):
        self.llm = llm_client

    def expand_query(self, query: str) -> list:
        """扩展查询"""
        prompt = f"""请将以下查询扩展为3个语义相关但表述不同的查询，用于提高搜索召回率。

原始查询：{query}

请返回3个扩展查询，每行一个："""

        response = self.llm.generate(prompt)
        expanded = response.strip().split('\n')
        return [query] + expanded

    def decompose_query(self, query: str) -> list:
        """分解复杂查询"""
        prompt = f"""请将以下复杂查询分解为多个简单的子查询。

原始查询：{query}

分解后的子查询（每行一个）："""

        response = self.llm.generate(prompt)
        sub_queries = response.strip().split('\n')
        return sub_queries

    def hyde(self, query: str) -> str:
        """HyDE：生成假设性文档"""
        prompt = f"""请根据以下问题，写一段可能包含答案的文档内容。

问题：{query}

假设性文档内容："""

        hypothetical_doc = self.llm.generate(prompt)
        return hypothetical_doc

    def rewrite_with_context(self, query: str, chat_history: list) -> str:
        """结合对话历史改写查询"""
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in chat_history[-4:]  # 最近4轮
        ])

        prompt = f"""根据对话历史，将用户的最新问题改写为独立完整的查询。

对话历史：
{history_text}

最新问题：{query}

改写后的独立查询："""

        rewritten = self.llm.generate(prompt)
        return rewritten.strip()
```

#### 2.2.2 知识库处理

**场景1：知识库问题生成与检索优化**

```python
class KnowledgeBaseOptimizer:
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model

    def generate_qa_pairs(self, document: str) -> list:
        """为文档生成问答对，提高检索命中率"""
        prompt = f"""请根据以下文档内容，生成5个用户可能会问的问题。

文档内容：
{document}

生成的问题（每行一个）："""

        response = self.llm.generate(prompt)
        questions = response.strip().split('\n')

        # 返回问答对
        qa_pairs = [{"question": q, "answer": document} for q in questions]
        return qa_pairs

    def index_with_qa(self, documents: list):
        """用问题作为索引键"""
        all_entries = []
        for doc in documents:
            # 生成问答对
            qa_pairs = self.generate_qa_pairs(doc)
            for qa in qa_pairs:
                all_entries.append({
                    "index_text": qa["question"],  # 用问题做检索
                    "content": qa["answer"]  # 返回文档
                })
        return all_entries
```

**场景2：知识库健康度检查**

```python
class KnowledgeBaseHealthChecker:
    def check_health(self, documents: list, embeddings: np.ndarray):
        """检查知识库健康度"""
        report = {
            "total_documents": len(documents),
            "duplicate_ratio": self.check_duplicates(embeddings),
            "coverage_analysis": self.analyze_coverage(documents),
            "quality_issues": self.check_quality(documents)
        }
        return report

    def check_duplicates(self, embeddings: np.ndarray, threshold=0.95):
        """检测重复文档"""
        similarity_matrix = cosine_similarity(embeddings)
        duplicates = []
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix)):
                if similarity_matrix[i][j] > threshold:
                    duplicates.append((i, j, similarity_matrix[i][j]))
        return len(duplicates) / len(embeddings)

    def analyze_coverage(self, documents: list):
        """分析主题覆盖"""
        # 使用聚类分析主题分布
        # 检测是否有主题缺失
        pass

    def check_quality(self, documents: list):
        """检查文档质量问题"""
        issues = []
        for i, doc in enumerate(documents):
            if len(doc) < 50:
                issues.append({"index": i, "issue": "文档过短"})
            if len(doc) > 2000:
                issues.append({"index": i, "issue": "文档过长，考虑分块"})
        return issues
```

---

### 2.3 如何提升RAG质量

**RAG质量提升策略矩阵：**

```
┌─────────────────────────────────────────────────────────────┐
│                   RAG质量提升策略                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  数据准备阶段                        │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  1. 文档预处理                                      │    │
│  │     - 清洗：去除噪音、乱码、无关内容                  │    │
│  │     - 格式统一：统一编码、格式                       │    │
│  │     - 元数据提取：标题、日期、来源等                  │    │
│  │                                                     │    │
│  │  2. 智能分块                                        │    │
│  │     - 语义分块：按段落、章节分块                     │    │
│  │     - 递归分块：大块套小块                           │    │
│  │     - 重叠分块：保持上下文连贯                       │    │
│  │     - 分块大小：通常200-500 tokens                  │    │
│  │                                                     │    │
│  │  3. 数据增强                                        │    │
│  │     - 生成问答对                                    │    │
│  │     - 生成摘要作为索引                              │    │
│  │     - 关键词提取                                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   检索阶段                          │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  1. 多路召回                                        │    │
│  │     - 向量检索 + 关键词检索                          │    │
│  │     - 多个Embedding模型                             │    │
│  │                                                     │    │
│  │  2. Query改写                                       │    │
│  │     - 扩展、分解、HyDE                              │    │
│  │                                                     │    │
│  │  3. 重排序（Rerank）                                 │    │
│  │     - 使用Cross-encoder重排                         │    │
│  │     - LLM重排序                                     │    │
│  │                                                     │    │
│  │  4. 过滤与后处理                                    │    │
│  │     - 相似度阈值过滤                                 │    │
│  │     - 去重、合并相关块                              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   生成阶段                          │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  1. Prompt优化                                      │    │
│  │     - 清晰的指令                                    │    │
│  │     - 合理的上下文组织                              │    │
│  │     - 输出格式约束                                  │    │
│  │                                                     │    │
│  │  2. 答案质量控制                                    │    │
│  │     - 要求引用来源                                  │    │
│  │     - 不确定时明确表示                              │    │
│  │     - 后置验证                                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 第三章：RAG多模态数据处理

### 3.1 多模态数据处理：PDF、Word、网页

#### 3.1.1 PDF文档解析

**PDF解析的挑战：**

```
PDF解析难点：
├── 布局复杂：多列、表格、图文混排
├── 格式多样：扫描件、文字型、混合型
├── 特殊元素：页眉页脚、水印、批注
├── 编码问题：中文乱码、特殊字符
└── 表格提取：跨页表格、合并单元格
```

**PDF解析方案对比：**

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| PyPDF2 | 轻量，仅文字 | 简单文字PDF |
| pdfplumber | 支持表格 | 需要表格提取 |
| PyMuPDF | 速度快，功能全 | 通用场景 |
| Unstructured | AI辅助解析 | 复杂布局 |
| MinerU | 多模态，准确度高 | 生产级应用 |
| Docling | IBM开源，表格优秀 | 企业文档 |

**使用PyMuPDF解析PDF：**

```python
import fitz  # PyMuPDF

def parse_pdf(pdf_path: str) -> list:
    """解析PDF文档"""
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc):
        # 提取文本
        text = page.get_text("text")

        # 提取图片
        images = []
        for img_index, img in enumerate(page.get_images()):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            images.append(pix.tobytes())

        # 提取表格（使用pdfplumber更准确）
        # tables = extract_tables(page)

        pages.append({
            "page_num": page_num + 1,
            "text": text,
            "images": images
        })

    doc.close()
    return pages


def smart_chunk_pdf(pages: list, chunk_size=500, overlap=50):
    """智能分块PDF内容"""
    chunks = []

    for page in pages:
        text = page["text"]
        page_num = page["page_num"]

        # 按段落分割
        paragraphs = text.split('\n\n')

        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": {"page": page_num}
                    })
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": {"page": page_num}
            })

    return chunks
```

#### 3.1.2 Word文档解析

```python
from docx import Document

def parse_docx(docx_path: str) -> dict:
    """解析Word文档"""
    doc = Document(docx_path)

    # 提取段落
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]

    # 提取表格
    tables = []
    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text for cell in row.cells]
            table_data.append(row_data)
        tables.append(table_data)

    return {
        "paragraphs": paragraphs,
        "tables": tables
    }
```

#### 3.1.3 网页数据解析

```python
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def parse_webpage(url: str) -> dict:
    """解析网页内容"""
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, 'html.parser')

    # 移除脚本和样式
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()

    # 提取标题
    title = soup.title.string if soup.title else ""

    # 提取正文
    main_content = soup.find('main') or soup.find('article') or soup.body
    text = main_content.get_text(separator='\n', strip=True)

    # 提取链接
    links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]

    return {
        "title": title,
        "content": text,
        "links": links,
        "url": url
    }
```

### 3.2 多模态数据处理：图片、视频

#### 3.2.1 图像数据的多重表征

```python
from PIL import Image
import base64
from io import BytesIO

class ImageProcessor:
    def __init__(self, vision_model, clip_model, ocr_model):
        self.vision_model = vision_model
        self.clip_model = clip_model
        self.ocr_model = ocr_model

    def process_image(self, image_path: str) -> dict:
        """多维度处理图像"""
        image = Image.open(image_path)

        # 1. 视觉描述
        description = self.vision_model.describe(image)

        # 2. CLIP向量
        clip_embedding = self.clip_model.encode(image)

        # 3. OCR文字提取
        ocr_text = self.ocr_model.extract(image)

        # 4. 组合表征
        combined_text = f"图像描述：{description}\n提取文字：{ocr_text}"

        return {
            "description": description,
            "clip_embedding": clip_embedding,
            "ocr_text": ocr_text,
            "combined_text": combined_text
        }
```

#### 3.2.2 GraphRAG使用

**GraphRAG概述：**

GraphRAG是微软开源的一种增强RAG方法，通过构建知识图谱来增强检索和生成能力。

```
GraphRAG工作流程：
1. 文档 → 实体提取 → 关系抽取 → 构建知识图谱
2. 知识图谱 → 社区检测 → 生成社区摘要
3. 查询时：
   - 局部搜索：从具体实体出发，遍历相关节点
   - 全局搜索：基于社区摘要回答宏观问题
```

```python
# GraphRAG配置示例
# settings.yaml

indexing:
  model: deepseek-chat
  chunk_size: 300
  chunk_overlap: 100

query:
  local:
    text_unit_prop: 0.5
    community_prop: 0.1
  global:
    map_max_tokens: 500
    reduce_max_tokens: 2000

# 使用命令
# graphrag index --root ./ragtest
# graphrag query --root ./ragtest --method local "你的问题"
# graphrag query --root ./ragtest --method global "宏观问题"
```

---

## 第四章：RAG调优

### 4.1 混合检索

#### 4.1.1 为什么需要混合检索

```
向量检索的局限：
├── 对专有名词、缩写不敏感
├── 可能遗漏精确匹配
├── 对数字、日期等效果差
└── 语义理解有时过于"发散"

关键词检索的局限：
├── 无法理解同义词
├── 忽略语义相似性
├── 对长尾查询效果差
└── 对拼写错误敏感

混合检索 = 向量检索 + 关键词检索
取长补短，提高召回率和准确率
```

#### 4.1.2 混合检索实现

```python
from rank_bm25 import BM25Okapi
import jieba
import numpy as np

class HybridRetriever:
    def __init__(self, embedding_model, documents):
        self.embedding_model = embedding_model
        self.documents = documents

        # 初始化向量检索
        self.embeddings = embedding_model.encode(documents)
        self.vector_index = self.build_vector_index(self.embeddings)

        # 初始化BM25检索
        tokenized_docs = [list(jieba.cut(doc)) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, top_k: int = 10, alpha: float = 0.5):
        """
        混合检索

        Args:
            query: 查询文本
            top_k: 返回数量
            alpha: 向量检索权重，0-1之间
        """
        # 向量检索
        query_embedding = self.embedding_model.encode([query])
        vector_scores = self.vector_search(query_embedding, top_k * 2)

        # BM25检索
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # 分数归一化
        vector_scores_norm = self.normalize(vector_scores)
        bm25_scores_norm = self.normalize(bm25_scores)

        # 融合分数
        combined_scores = alpha * vector_scores_norm + (1 - alpha) * bm25_scores_norm

        # 排序返回
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        results = [(self.documents[i], combined_scores[i]) for i in top_indices]

        return results

    def normalize(self, scores):
        """Min-Max归一化"""
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
```

#### 4.1.3 Rerank重排序

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list, top_k: int = 5):
        """使用Cross-Encoder重排序"""
        # 构建query-document对
        pairs = [[query, doc] for doc in documents]

        # 计算相关性分数
        scores = self.model.predict(pairs)

        # 排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]


# 完整的检索流程
class AdvancedRAG:
    def __init__(self):
        self.hybrid_retriever = HybridRetriever(...)
        self.reranker = Reranker()

    def retrieve(self, query: str, top_k: int = 5):
        # 1. 混合检索，多召回
        candidates = self.hybrid_retriever.search(query, top_k=20)

        # 2. 重排序，精排
        docs = [doc for doc, score in candidates]
        reranked = self.reranker.rerank(query, docs, top_k=top_k)

        return reranked
```

### 4.2 RAG效果评估

#### 4.2.1 评估指标

```
┌─────────────────────────────────────────────────────────────┐
│                   RAG评估指标体系                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  检索阶段指标：                                              │
│  ├── 召回率(Recall@K)：相关文档被检索到的比例                 │
│  ├── 精确率(Precision@K)：检索结果中相关文档的比例            │
│  ├── MRR：第一个相关文档的排名倒数                           │
│  └── NDCG：考虑排名的归一化折扣累积增益                       │
│                                                             │
│  生成阶段指标：                                              │
│  ├── 忠实度(Faithfulness)：答案是否基于检索内容               │
│  ├── 相关性(Relevance)：答案与问题的相关程度                  │
│  ├── 正确性(Correctness)：答案是否正确                       │
│  └── 流畅度(Fluency)：答案是否通顺                           │
│                                                             │
│  端到端指标：                                                │
│  ├── 答案准确率                                             │
│  ├── 引用准确率                                             │
│  └── 用户满意度                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 4.2.2 评估工具：RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# 准备评估数据
eval_data = {
    "question": ["什么是RAG？", "FAISS有什么特点？"],
    "answer": ["RAG是...", "FAISS是..."],
    "contexts": [["文档1", "文档2"], ["文档3", "文档4"]],
    "ground_truth": ["RAG的正确定义...", "FAISS的正确特点..."]
}

# 执行评估
result = evaluate(
    eval_data,
    metrics=[
        faithfulness,  # 答案忠实于上下文
        answer_relevancy,  # 答案与问题相关
        context_precision,  # 检索的精确率
        context_recall,  # 检索的召回率
    ]
)

print(result)
```

---

## 第五章：项目实战——企业知识库

### 5.1 RAG冠军方案解析

**冠军方案核心架构：多路由 + 动态知识库**

```
┌─────────────────────────────────────────────────────────────┐
│                   RAG冠军方案架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户Query                                                   │
│      ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              意图识别 & 路由                        │    │
│  │  ├── 简单问答 → 直接检索                            │    │
│  │  ├── 复杂推理 → 多跳检索                            │    │
│  │  ├── 对比问题 → 多文档聚合                          │    │
│  │  └── 计算问题 → 工具调用                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              智能检索层                             │    │
│  │  ├── Docling解析优化                                │    │
│  │  ├── 表格序列化                                     │    │
│  │  ├── 父页面检索                                     │    │
│  │  └── LLM重排序                                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              生成优化层                             │    │
│  │  ├── 思维链推理                                     │    │
│  │  ├── 结构化输出                                     │    │
│  │  └── 指令细化                                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                 │
│  最终答案                                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 关键技术实现

**1. Docling解析优化**

```python
from docling.document_converter import DocumentConverter

def parse_with_docling(pdf_path):
    """使用Docling进行高质量PDF解析"""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    # 获取Markdown格式输出
    markdown = result.document.export_to_markdown()

    # 获取表格
    tables = result.document.tables

    return markdown, tables
```

**2. 父页面检索**

```python
class ParentDocumentRetriever:
    """
    检索时使用小块，返回时使用大块（父文档）
    提高检索精度的同时保持上下文完整性
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.child_chunks = []  # 小块，用于检索
        self.parent_docs = []   # 大块，用于返回
        self.child_to_parent = {}  # 映射关系

    def add_document(self, doc, parent_chunk_size=2000, child_chunk_size=200):
        parent_id = len(self.parent_docs)
        # 父文档
        self.parent_docs.append(doc)

        # 子文档
        child_chunks = self.chunk(doc, child_chunk_size)
        for chunk in child_chunks:
            child_id = len(self.child_chunks)
            self.child_chunks.append(chunk)
            self.child_to_parent[child_id] = parent_id

    def search(self, query, top_k=3):
        # 用子块检索
        child_results = self.vector_search(query, top_k * 3)

        # 映射到父文档，去重
        parent_ids = list(set([
            self.child_to_parent[child_id]
            for child_id, score in child_results
        ]))[:top_k]

        return [self.parent_docs[pid] for pid in parent_ids]
```

**3. LLM重排序**

```python
def llm_rerank(query: str, documents: list, llm) -> list:
    """使用LLM进行重排序"""
    prompt = f"""请对以下文档按照与查询的相关性进行排序。

查询：{query}

文档列表：
{chr(10).join([f"[{i}] {doc[:200]}..." for i, doc in enumerate(documents)])}

请返回按相关性从高到低排序的文档编号，格式如：0, 2, 1, 3
"""
    response = llm.generate(prompt)
    # 解析排序结果
    order = [int(x.strip()) for x in response.split(',')]
    return [documents[i] for i in order]
```

---

## 专项求职辅导

### RAG相关面试问题精讲

**1. 你能画一下RAG的系统架构图吗？**

```
核心组件：
1. 文档处理管线：加载 → 解析 → 分块 → 向量化
2. 向量存储：向量数据库 + 元数据存储
3. 检索引擎：Query处理 → 向量检索 → 重排序
4. 生成引擎：Prompt构建 → LLM调用 → 后处理
5. 评估监控：效果评估 + 日志监控

关键设计点：
- 分块策略的选择
- 多路召回的设计
- 重排序的引入
- Prompt模板的优化
```

**2. 文档分块有哪些策略？**

```
1. 固定大小分块
   - 按字符数/token数切分
   - 简单但可能破坏语义

2. 语义分块
   - 按段落、章节分块
   - 保持语义完整性

3. 递归分块
   - 先大块再小块
   - 配合父文档检索

4. 重叠分块
   - 块之间有重叠
   - 保持上下文连贯

选择依据：文档类型、查询模式、模型上下文长度
```

**3. 如果你的RAG效果很差，你会从哪几个方面去调试？**

```
调试路径：
1. 检索问题？
   - 检查召回率：相关文档是否被检索到
   - 检查排序：相关文档排名是否靠前
   - 解决方案：改进分块、调整Embedding、引入重排序

2. 分块问题？
   - 检查分块是否破坏语义
   - 检查分块大小是否合适
   - 解决方案：调整分块策略、增加重叠

3. Query问题？
   - 用户Query是否模糊
   - 解决方案：Query改写、扩展

4. 生成问题？
   - 检索到了但答案不对
   - 解决方案：优化Prompt、调整temperature
```

---

## 本模块总结

### 核心技能清单

1. **Embedding技术**：理解原理，会选择模型，会评估效果
2. **向量数据库**：掌握FAISS/Milvus使用，理解索引类型
3. **RAG流程**：完整掌握索引和查询流程
4. **多模态处理**：会处理PDF、Word、网页、图片
5. **调优技巧**：Query改写、混合检索、重排序
6. **效果评估**：掌握评估指标和工具

### 实践建议

1. 从简单RAG开始，逐步增加复杂度
2. 重视数据质量，"Garbage In, Garbage Out"
3. 建立评估体系，量化优化效果
4. 关注用户反馈，持续迭代改进

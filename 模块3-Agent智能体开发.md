# 模块3: Agent智能体开发

## 课程概述

本模块深入讲解AI Agent（智能体）的设计与开发。从Function Calling和MCP协议入手，逐步构建Agent的数据决策能力、搜索感知能力和记忆能力，并通过OpenManus项目实战，掌握企业级Agent系统的完整开发流程。

---

## 第一章：Function Calling与MCP

### 1.1 使用Function Calling进行工具调用

#### 1.1.1 什么是Function Calling？

**Function Calling的定义：**

Function Calling（函数调用）是一种让大语言模型能够调用外部工具和API的机制。通过Function Calling，LLM可以：

1. 识别用户意图需要调用哪个工具
2. 从对话中提取工具所需的参数
3. 返回结构化的函数调用请求
4. 基于函数执行结果继续对话

```
┌─────────────────────────────────────────────────────────────┐
│                  Function Calling工作流程                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户："北京今天天气怎么样？"                                 │
│              ↓                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  LLM分析：需要调用天气查询工具                       │    │
│  │  提取参数：city="北京", date="今天"                  │    │
│  └─────────────────────────────────────────────────────┘    │
│              ↓                                              │
│  返回函数调用请求：                                          │
│  {                                                          │
│    "name": "get_weather",                                   │
│    "arguments": {"city": "北京", "date": "2024-01-15"}     │
│  }                                                          │
│              ↓                                              │
│  应用程序执行函数，获取结果                                   │
│              ↓                                              │
│  将结果返回给LLM                                             │
│              ↓                                              │
│  LLM生成最终回答："北京今天晴，气温-2°C到8°C..."             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Function Calling的核心价值：**

| 价值 | 说明 |
|------|------|
| 扩展能力边界 | 让LLM能够获取实时信息、执行操作 |
| 结构化输出 | 保证输出格式符合预期 |
| 可控性 | 明确定义LLM能做什么、不能做什么 |
| 可组合性 | 多个工具可以组合使用 |

#### 1.1.2 Function Calling与MCP的区别

```
┌─────────────────────────────────────────────────────────────┐
│              Function Calling vs MCP                        │
├─────────────────┬─────────────────┬─────────────────────────┤
│     维度        │ Function Calling │         MCP            │
├─────────────────┼─────────────────┼─────────────────────────┤
│  定义者        │ OpenAI等各厂商   │ Anthropic提出的协议      │
│  标准化程度    │ 各家实现不同     │ 统一的协议规范          │
│  工具发现      │ 需要预定义       │ 动态发现服务能力        │
│  运行时        │ 应用内执行       │ 独立的MCP Server        │
│  复用性        │ 需要重复实现     │ Server可跨应用复用      │
│  适用场景      │ 简单工具调用     │ 复杂工具生态            │
└─────────────────┴─────────────────┴─────────────────────────┘

MCP = Model Context Protocol（模型上下文协议）
是一种标准化的工具通信协议，让AI应用可以连接各种工具服务
```

#### 1.1.3 使用Qwen3完成天气Function Calling

```python
from openai import OpenAI
import json

# 初始化客户端（以通义千问为例）
client = OpenAI(
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海"
                    },
                    "date": {
                        "type": "string",
                        "description": "日期，格式：YYYY-MM-DD，默认今天"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 模拟天气API
def get_weather(city: str, date: str = None) -> dict:
    """模拟天气查询"""
    weather_data = {
        "北京": {"temp": "5°C", "condition": "晴", "wind": "北风3级"},
        "上海": {"temp": "12°C", "condition": "多云", "wind": "东风2级"},
        "广州": {"temp": "18°C", "condition": "小雨", "wind": "南风1级"},
    }
    return weather_data.get(city, {"error": "未找到该城市天气信息"})

# 执行Function Calling
def chat_with_tools(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    # 第一轮：让模型决定是否调用工具
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message

    # 检查是否需要调用工具
    if assistant_message.tool_calls:
        messages.append(assistant_message)

        # 执行工具调用
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # 调用对应函数
            if function_name == "get_weather":
                result = get_weather(**arguments)

            # 将结果添加到消息中
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

        # 第二轮：让模型基于工具结果生成回答
        final_response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages
        )

        return final_response.choices[0].message.content
    else:
        return assistant_message.content

# 使用示例
answer = chat_with_tools("北京今天天气怎么样？")
print(answer)
# 输出：北京今天天气晴朗，气温5°C，北风3级。
```

#### 1.1.4 使用Function Calling完成数据库查询

```python
import sqlite3
import json
from openai import OpenAI

# 定义SQL查询工具
sql_tools = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "执行SQL查询获取数据库信息。可以查询用户表(users)、订单表(orders)、产品表(products)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "要执行的SQL SELECT查询语句"
                    }
                },
                "required": ["sql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_schema",
            "description": "获取数据库表的结构信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "表名：users, orders, products"
                    }
                },
                "required": ["table_name"]
            }
        }
    }
]

class DatabaseAgent:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.client = OpenAI(...)

        # 表结构信息
        self.schemas = {
            "users": "id INT, name TEXT, email TEXT, created_at DATETIME",
            "orders": "id INT, user_id INT, product_id INT, amount DECIMAL, order_date DATETIME",
            "products": "id INT, name TEXT, price DECIMAL, stock INT"
        }

    def get_table_schema(self, table_name: str) -> str:
        return self.schemas.get(table_name, "表不存在")

    def query_database(self, sql: str) -> str:
        """执行SQL查询（只允许SELECT）"""
        # 安全检查：只允许SELECT语句
        if not sql.strip().upper().startswith("SELECT"):
            return "错误：只允许SELECT查询"

        try:
            cursor = self.conn.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            result = {"columns": columns, "data": rows}
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"查询错误：{str(e)}"

    def chat(self, user_question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """你是一个数据库查询助手。根据用户的问题：
1. 先了解表结构（如需要）
2. 构造正确的SQL查询
3. 解读查询结果并回答用户

数据库包含：users(用户表), orders(订单表), products(产品表)"""
            },
            {"role": "user", "content": user_question}
        ]

        # 多轮工具调用
        while True:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=sql_tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            if not message.tool_calls:
                return message.content

            messages.append(message)

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if func_name == "query_database":
                    result = self.query_database(args["sql"])
                elif func_name == "get_table_schema":
                    result = self.get_table_schema(args["table_name"])

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })


# 使用示例
agent = DatabaseAgent("company.db")
answer = agent.chat("最近一个月销售额最高的产品是什么？")
print(answer)
```

---

### 1.2 MCP与A2A的应用

#### 1.2.1 MCP的核心概念

**MCP架构：**

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP架构                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                       ┌─────────────┐     │
│  │  MCP Host   │ ←─── MCP协议 ───→    │ MCP Server  │     │
│  │ (AI应用)    │                       │ (工具服务)   │     │
│  └─────────────┘                       └─────────────┘     │
│        ↑                                     ↑             │
│        │                                     │             │
│  ┌─────────────┐                       ┌─────────────┐     │
│  │ MCP Client  │                       │   Tools     │     │
│  │ (协议实现)   │                       │ Resources   │     │
│  └─────────────┘                       │ Prompts     │     │
│                                        └─────────────┘     │
│                                                             │
│  核心组件说明：                                              │
│  ├── MCP Host：运行AI应用的环境（如Claude Desktop）          │
│  ├── MCP Client：实现MCP协议的客户端                        │
│  ├── MCP Server：提供工具能力的服务                         │
│  └── 三种能力类型：                                         │
│      ├── Tools：可执行的工具/函数                           │
│      ├── Resources：可读取的资源/数据                       │
│      └── Prompts：预定义的提示模板                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 1.2.2 MCP的使用场景

```
MCP应用场景：
├── 文件系统访问
│   └── 读写本地文件、目录操作
├── 数据库连接
│   └── 查询MySQL、PostgreSQL、SQLite
├── Web内容抓取
│   └── 获取网页内容、API调用
├── 代码执行
│   └── 运行Python、JavaScript代码
├── 搜索服务
│   └── 接入Google、Bing搜索
└── 企业系统集成
    └── 连接CRM、ERP等内部系统
```

#### 1.2.3 CASE：搭建MCP服务

**1. 创建一个旅游攻略MCP Server**

```python
# travel_mcp_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 创建MCP Server
server = Server("travel-guide")

# 模拟旅游数据
TRAVEL_DATA = {
    "北京": {
        "景点": ["故宫", "长城", "颐和园", "天坛"],
        "美食": ["烤鸭", "炸酱面", "豆汁", "卤煮"],
        "最佳季节": "春季(4-5月)和秋季(9-10月)"
    },
    "上海": {
        "景点": ["外滩", "东方明珠", "豫园", "迪士尼"],
        "美食": ["小笼包", "生煎", "本帮菜", "蟹壳黄"],
        "最佳季节": "春季(3-5月)和秋季(9-11月)"
    }
}

@server.list_tools()
async def list_tools():
    """列出可用工具"""
    return [
        Tool(
            name="get_travel_guide",
            description="获取指定城市的旅游攻略，包括景点、美食和最佳旅游季节",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        ),
        Tool(
            name="search_attractions",
            description="搜索特定类型的景点",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["历史文化", "自然风光", "主题乐园"]
                    }
                },
                "required": ["city"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """执行工具调用"""
    if name == "get_travel_guide":
        city = arguments["city"]
        if city in TRAVEL_DATA:
            data = TRAVEL_DATA[city]
            result = f"""
{city}旅游攻略：

🏛 必游景点：
{chr(10).join(['  • ' + s for s in data['景点']])}

🍜 特色美食：
{chr(10).join(['  • ' + f for f in data['美食']])}

📅 最佳旅游季节：{data['最佳季节']}
"""
            return [TextContent(type="text", text=result)]
        else:
            return [TextContent(type="text", text=f"暂无{city}的旅游信息")]

    elif name == "search_attractions":
        # 实现景点搜索逻辑
        pass

# 运行服务
async def main():
    async with stdio_server() as streams:
        await server.run(
            streams[0],  # stdin
            streams[1],  # stdout
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**2. 配置MCP Server（Claude Desktop）**

```json
// ~/.config/claude/claude_desktop_config.json (macOS)
{
  "mcpServers": {
    "travel-guide": {
      "command": "python",
      "args": ["/path/to/travel_mcp_server.py"],
      "env": {}
    },
    "fetch": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-fetch"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-filesystem", "/Users/me/documents"]
    }
  }
}
```

#### 1.2.4 什么是Agent2Agent (A2A)

**A2A协议概述：**

```
A2A (Agent-to-Agent) 是Google DeepMind提出的Agent间通信协议

┌─────────────────────────────────────────────────────────────┐
│                    A2A vs MCP                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  MCP：Agent ←→ 工具服务                                     │
│  └── 让Agent能够调用外部工具                                 │
│                                                             │
│  A2A：Agent ←→ Agent                                        │
│  └── 让多个Agent能够协作完成复杂任务                          │
│                                                             │
│  关系：                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   用户请求                          │    │
│  │                      ↓                              │    │
│  │              ┌─────────────┐                        │    │
│  │              │ 主Agent     │                        │    │
│  │              └──────┬──────┘                        │    │
│  │           ┌────────┼────────┐                       │    │
│  │           ↓        ↓        ↓                       │    │
│  │    ┌─────────┐┌─────────┐┌─────────┐               │    │
│  │    │Agent A  ││Agent B  ││Agent C  │  ← A2A通信    │    │
│  │    └────┬────┘└────┬────┘└────┬────┘               │    │
│  │         ↓          ↓          ↓                     │    │
│  │    ┌─────────┐┌─────────┐┌─────────┐               │    │
│  │    │MCP工具  ││MCP工具  ││MCP工具  │  ← MCP调用    │    │
│  │    └─────────┘└─────────┘└─────────┘               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 第二章：构建Agent的数据决策能力

### 2.1 AI大赛：二手车价格预测

#### 2.1.1 为什么Agent的数据决策要用专业工具

**分析式AI vs 生成式AI：**

```
┌─────────────────────────────────────────────────────────────┐
│              分析式AI vs 生成式AI                           │
├─────────────────┬─────────────────┬─────────────────────────┤
│     维度        │   分析式AI      │      生成式AI           │
├─────────────────┼─────────────────┼─────────────────────────┤
│  核心任务       │ 预测、分类、回归 │ 生成文本、图像、代码    │
│  数据需求       │ 结构化数据      │ 非结构化数据            │
│  代表算法       │ XGBoost, RF等   │ GPT, BERT等            │
│  输出类型       │ 数值/类别       │ 自然语言/媒体          │
│  可解释性       │ 较高            │ 较低                   │
│  计算效率       │ 高效            │ 资源密集               │
├─────────────────┴─────────────────┴─────────────────────────┤
│                                                             │
│  最佳实践：让LLM Agent调用专业ML工具进行数据决策              │
│  ├── LLM负责：理解需求、选择方法、解释结果                    │
│  └── ML工具负责：实际的预测和分类计算                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.1.2 十大经典机器学习算法

```
┌─────────────────────────────────────────────────────────────┐
│                 十大经典机器学习算法                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  监督学习 - 回归：                                           │
│  1. 线性回归 (Linear Regression)                            │
│     └── 预测连续值，简单可解释                               │
│  2. 决策树回归 (Decision Tree Regressor)                    │
│     └── 非线性关系，特征重要性                               │
│  3. 随机森林 (Random Forest)                                │
│     └── 集成学习，抗过拟合                                   │
│  4. XGBoost / LightGBM                                      │
│     └── 梯度提升，竞赛利器                                   │
│                                                             │
│  监督学习 - 分类：                                           │
│  5. 逻辑回归 (Logistic Regression)                          │
│     └── 二分类基线，概率输出                                 │
│  6. 支持向量机 (SVM)                                        │
│     └── 小样本，高维空间                                     │
│  7. K近邻 (KNN)                                             │
│     └── 简单直观，无需训练                                   │
│  8. 朴素贝叶斯 (Naive Bayes)                                │
│     └── 文本分类，快速                                      │
│                                                             │
│  无监督学习：                                                │
│  9. K-Means聚类                                             │
│     └── 数据分组，客户细分                                   │
│  10. 主成分分析 (PCA)                                        │
│      └── 降维，特征压缩                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 挑战Baseline

#### 2.2.1 数据探索

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv('used_cars.csv')

# 基本信息
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"\n数据类型:\n{df.dtypes}")
print(f"\n缺失值:\n{df.isnull().sum()}")

# 统计描述
print(f"\n数值列统计:\n{df.describe()}")

# 目标变量分布
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
df['price'].hist(bins=50)
plt.title('价格分布')

plt.subplot(1, 2, 2)
np.log1p(df['price']).hist(bins=50)
plt.title('对数价格分布')
plt.tight_layout()
plt.show()

# 相关性分析
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('特征相关性热力图')
plt.show()
```

#### 2.2.2 特征工程

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def create_features(self, df):
        """特征工程"""
        df = df.copy()

        # 1. 车龄特征
        current_year = 2024
        df['car_age'] = current_year - df['year']
        df['age_squared'] = df['car_age'] ** 2

        # 2. 里程相关特征
        df['km_per_year'] = df['mileage'] / (df['car_age'] + 1)

        # 3. 品牌价值（可以用历史数据计算品牌均价）
        brand_avg_price = df.groupby('brand')['price'].transform('mean')
        df['brand_price_ratio'] = df['price'] / brand_avg_price

        # 4. 类别特征编码
        categorical_cols = ['brand', 'model', 'fuel_type', 'transmission']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))

        return df

    def prepare_data(self, df, target_col='price'):
        """准备训练数据"""
        df = self.create_features(df)

        # 选择特征列
        feature_cols = [col for col in df.columns
                       if col not in [target_col, 'brand', 'model', 'fuel_type', 'transmission']
                       and df[col].dtype in ['int64', 'float64']]

        X = df[feature_cols]
        y = df[target_col]

        # 处理缺失值
        X = X.fillna(X.median())

        return X, y
```

#### 2.2.3 模型训练与预测

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

class CarPricePredictor:
    def __init__(self):
        self.models = {
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1),
            'lgb': lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
        }
        self.best_model = None

    def train_and_evaluate(self, X_train, X_val, y_train, y_val):
        """训练并评估所有模型"""
        results = {}

        for name, model in self.models.items():
            print(f"训练 {name}...")
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_val)

            # 评估
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            results[name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model': model
            }

            print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")

        # 选择最佳模型
        best_name = min(results, key=lambda x: results[x]['rmse'])
        self.best_model = results[best_name]['model']
        print(f"\n最佳模型: {best_name}")

        return results

    def predict(self, X):
        """使用最佳模型预测"""
        return self.best_model.predict(X)

    def get_feature_importance(self, feature_names):
        """获取特征重要性"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            return sorted(zip(feature_names, importance),
                         key=lambda x: x[1], reverse=True)
        return None


# 使用示例
fe = FeatureEngineer()
X, y = fe.prepare_data(df)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

predictor = CarPricePredictor()
results = predictor.train_and_evaluate(X_train, X_val, y_train, y_val)

# 查看特征重要性
importance = predictor.get_feature_importance(X.columns.tolist())
print("\n特征重要性Top10:")
for feat, imp in importance[:10]:
    print(f"  {feat}: {imp:.4f}")
```

---

## 第三章：构建Agent的搜索、感知与记忆能力

### 3.1 Agent的信息抓取及搜索能力构建

#### 3.1.1 RAG能力集成

```python
class SearchableAgent:
    """具有RAG能力的Agent"""

    def __init__(self, llm, embedding_model, vector_db):
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_db = vector_db

    def search_knowledge_base(self, query: str, top_k: int = 5) -> list:
        """从知识库检索相关信息"""
        query_embedding = self.embedding_model.encode(query)
        results = self.vector_db.search(query_embedding, top_k=top_k)
        return results

    def answer_with_rag(self, question: str) -> str:
        """使用RAG回答问题"""
        # 检索相关文档
        docs = self.search_knowledge_base(question)
        context = "\n\n".join([doc.content for doc in docs])

        # 构建提示
        prompt = f"""请根据以下资料回答问题。

参考资料：
{context}

问题：{question}

回答："""

        return self.llm.generate(prompt)
```

#### 3.1.2 Web搜索能力

```python
import requests
from bs4 import BeautifulSoup

class WebSearchAgent:
    """具有Web搜索能力的Agent"""

    def __init__(self, llm, search_api_key=None):
        self.llm = llm
        self.search_api_key = search_api_key

    def search_web(self, query: str, num_results: int = 5) -> list:
        """执行网络搜索"""
        # 使用搜索API（以SerpAPI为例）
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.search_api_key,
            "num": num_results
        }
        response = requests.get(url, params=params)
        results = response.json().get("organic_results", [])

        return [
            {"title": r["title"], "snippet": r["snippet"], "link": r["link"]}
            for r in results
        ]

    def fetch_webpage(self, url: str) -> str:
        """抓取网页内容"""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            # 移除脚本和样式
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()

            return soup.get_text(separator='\n', strip=True)[:5000]
        except Exception as e:
            return f"抓取失败: {str(e)}"

    def search_and_answer(self, question: str) -> str:
        """搜索并回答问题"""
        # 1. 搜索
        search_results = self.search_web(question)

        # 2. 抓取Top结果的内容
        contents = []
        for result in search_results[:3]:
            content = self.fetch_webpage(result["link"])
            contents.append(f"来源: {result['title']}\n{content[:1000]}")

        context = "\n\n---\n\n".join(contents)

        # 3. 生成回答
        prompt = f"""请根据以下搜索结果回答问题。

搜索结果：
{context}

问题：{question}

请综合以上信息给出准确的回答，并标注信息来源："""

        return self.llm.generate(prompt)
```

### 3.2 Agent Memory能力开发

#### 3.2.1 短时记忆（Short-term Memory）

```python
from collections import deque
from typing import List, Dict

class ShortTermMemory:
    """短时记忆：维护对话历史"""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: deque = deque(maxlen=max_turns * 2)

    def add(self, role: str, content: str):
        """添加消息"""
        self.history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict]:
        """获取对话历史"""
        return list(self.history)

    def get_context_string(self) -> str:
        """获取上下文字符串"""
        return "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.history
        ])

    def clear(self):
        """清空记忆"""
        self.history.clear()

    def summarize(self, llm) -> str:
        """让LLM总结对话历史"""
        if len(self.history) == 0:
            return ""

        prompt = f"""请简洁总结以下对话的要点：

{self.get_context_string()}

总结："""
        return llm.generate(prompt)
```

#### 3.2.2 长时记忆（Long-term Memory）

```python
import json
from datetime import datetime

class LongTermMemory:
    """长时记忆：持久化存储重要信息"""

    def __init__(self, embedding_model, vector_db, user_id: str):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.user_id = user_id

    def store(self, content: str, memory_type: str = "general"):
        """存储记忆"""
        embedding = self.embedding_model.encode(content)

        memory_entry = {
            "user_id": self.user_id,
            "content": content,
            "type": memory_type,
            "timestamp": datetime.now().isoformat(),
            "embedding": embedding.tolist()
        }

        self.vector_db.insert(memory_entry)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关记忆"""
        query_embedding = self.embedding_model.encode(query)

        results = self.vector_db.search(
            query_embedding,
            filter={"user_id": self.user_id},
            top_k=top_k
        )

        return results

    def get_user_profile(self) -> Dict:
        """获取用户画像"""
        # 检索用户相关的所有记忆
        all_memories = self.vector_db.get_all(
            filter={"user_id": self.user_id}
        )

        # 提取关键信息形成用户画像
        profile = {
            "preferences": [],
            "interests": [],
            "past_interactions": len(all_memories)
        }

        # 可以用LLM来分析记忆并生成画像
        return profile
```

#### 3.2.3 构建记忆流（Memory Stream）

```python
class MemoryStream:
    """
    记忆流：结合短时和长时记忆的统一管理
    灵感来自Stanford的Generative Agents论文
    """

    def __init__(self, llm, embedding_model, vector_db, user_id: str):
        self.llm = llm
        self.short_term = ShortTermMemory(max_turns=10)
        self.long_term = LongTermMemory(embedding_model, vector_db, user_id)

    def add_observation(self, content: str, importance: float = None):
        """添加观察/事件"""
        # 添加到短时记忆
        self.short_term.add("observation", content)

        # 评估重要性
        if importance is None:
            importance = self._evaluate_importance(content)

        # 重要的记忆存入长时记忆
        if importance > 0.7:
            self.long_term.store(content, memory_type="important")

    def _evaluate_importance(self, content: str) -> float:
        """评估记忆的重要性 (0-1)"""
        prompt = f"""请评估以下信息的重要性，从1到10打分：
信息：{content}
重要性评分（只输出数字）："""

        try:
            score = float(self.llm.generate(prompt).strip())
            return min(max(score / 10, 0), 1)
        except:
            return 0.5

    def reflect(self) -> str:
        """反思：从记忆中提取高层次的洞察"""
        # 获取最近的记忆
        recent = self.short_term.get_context_string()

        # 获取相关的长时记忆
        long_term_relevant = self.long_term.retrieve(recent, top_k=5)
        long_term_context = "\n".join([m["content"] for m in long_term_relevant])

        prompt = f"""基于以下记忆，提取3个高层次的洞察或模式：

近期记忆：
{recent}

历史记忆：
{long_term_context}

洞察："""

        insights = self.llm.generate(prompt)

        # 将洞察也存入长时记忆
        self.long_term.store(insights, memory_type="reflection")

        return insights

    def retrieve_context(self, query: str) -> str:
        """为回答问题检索相关上下文"""
        # 短时记忆
        short_context = self.short_term.get_context_string()

        # 长时记忆
        long_memories = self.long_term.retrieve(query, top_k=3)
        long_context = "\n".join([m["content"] for m in long_memories])

        return f"""
对话历史：
{short_context}

相关记忆：
{long_context}
"""
```

---

## 第四章：Agent的能力优化与效果评估

### 4.1 使用用户反馈提升Agent能力

#### 4.1.1 显式反馈收集

```python
class FeedbackCollector:
    """收集用户显式反馈"""

    def __init__(self, db):
        self.db = db

    def collect_rating(self, session_id: str, response_id: str,
                       rating: int, comment: str = None):
        """收集评分反馈"""
        feedback = {
            "session_id": session_id,
            "response_id": response_id,
            "rating": rating,  # 1-5星
            "comment": comment,
            "timestamp": datetime.now().isoformat(),
            "feedback_type": "rating"
        }
        self.db.insert("feedback", feedback)

    def collect_correction(self, session_id: str, original_response: str,
                          corrected_response: str):
        """收集纠正反馈"""
        feedback = {
            "session_id": session_id,
            "original": original_response,
            "corrected": corrected_response,
            "timestamp": datetime.now().isoformat(),
            "feedback_type": "correction"
        }
        self.db.insert("feedback", feedback)

    def analyze_feedback(self) -> Dict:
        """分析反馈数据"""
        all_feedback = self.db.query("feedback")

        ratings = [f["rating"] for f in all_feedback if f["feedback_type"] == "rating"]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0

        # 统计常见问题
        corrections = [f for f in all_feedback if f["feedback_type"] == "correction"]

        return {
            "total_feedback": len(all_feedback),
            "average_rating": avg_rating,
            "correction_count": len(corrections)
        }
```

#### 4.1.2 隐式反馈收集

```python
class ImplicitFeedbackTracker:
    """追踪用户隐式反馈"""

    def __init__(self, db):
        self.db = db

    def track_interaction(self, session_id: str, event_type: str, data: Dict):
        """记录交互事件"""
        event = {
            "session_id": session_id,
            "event_type": event_type,  # click, copy, regenerate, abandon
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        self.db.insert("interactions", event)

    def analyze_behavior(self, session_id: str = None) -> Dict:
        """分析用户行为"""
        query = {"session_id": session_id} if session_id else {}
        interactions = self.db.query("interactions", query)

        # 分析指标
        regenerate_rate = len([i for i in interactions if i["event_type"] == "regenerate"]) / len(interactions)
        copy_rate = len([i for i in interactions if i["event_type"] == "copy"]) / len(interactions)
        abandon_rate = len([i for i in interactions if i["event_type"] == "abandon"]) / len(interactions)

        return {
            "regenerate_rate": regenerate_rate,  # 重新生成率（高=不满意）
            "copy_rate": copy_rate,  # 复制率（高=有用）
            "abandon_rate": abandon_rate  # 放弃率（高=无效）
        }
```

#### 4.1.3 利用反馈进行优化

```python
class FeedbackOptimizer:
    """基于反馈优化Agent"""

    def __init__(self, llm, feedback_db, knowledge_db):
        self.llm = llm
        self.feedback_db = feedback_db
        self.knowledge_db = knowledge_db

    def generate_training_data(self) -> List[Dict]:
        """从反馈生成训练数据"""
        corrections = self.feedback_db.query(
            "feedback",
            {"feedback_type": "correction"}
        )

        training_data = []
        for c in corrections:
            training_data.append({
                "instruction": c.get("original_query", ""),
                "input": "",
                "output": c["corrected"]
            })

        return training_data

    def update_knowledge_base(self):
        """根据反馈更新知识库"""
        # 获取低评分的回答
        low_rated = self.feedback_db.query(
            "feedback",
            {"feedback_type": "rating", "rating": {"$lt": 3}}
        )

        for feedback in low_rated:
            if feedback.get("comment"):
                # 分析问题并更新知识库
                analysis = self.llm.generate(f"""
分析以下负面反馈，识别需要改进的知识点：

原回答：{feedback.get('original_response')}
用户评论：{feedback['comment']}

需要补充的知识点：""")

                # 将新知识添加到知识库
                self.knowledge_db.add_document(analysis)

    def generate_prompt_improvements(self) -> str:
        """生成提示词改进建议"""
        all_feedback = self.feedback_db.query("feedback")

        prompt = f"""分析以下用户反馈，提出系统提示词的改进建议：

反馈数据：
{json.dumps(all_feedback[:50], ensure_ascii=False, indent=2)}

改进建议："""

        return self.llm.generate(prompt)
```

### 4.2 Agent智能体效果评估

#### 4.2.1 RAG能力评估（大海捞针）

```python
class NeedleInHaystackEvaluator:
    """
    大海捞针测试：评估Agent在长上下文中定位信息的能力
    """

    def __init__(self, agent):
        self.agent = agent

    def generate_haystack(self, length: int, needle: str, position: float) -> str:
        """
        生成测试数据

        Args:
            length: 总长度（tokens）
            needle: 要隐藏的关键信息
            position: 关键信息的位置（0-1）
        """
        # 生成填充文本
        filler = "这是一段普通的填充文本。" * (length // 10)

        # 在指定位置插入关键信息
        insert_pos = int(len(filler) * position)
        haystack = filler[:insert_pos] + f" {needle} " + filler[insert_pos:]

        return haystack

    def evaluate(self, context_lengths: List[int], positions: List[float]) -> Dict:
        """执行评估"""
        needle = "秘密代码是ABC123"
        question = "文档中提到的秘密代码是什么？"

        results = {}

        for length in context_lengths:
            results[length] = {}
            for position in positions:
                haystack = self.generate_haystack(length, needle, position)

                # 测试Agent能否找到needle
                response = self.agent.answer(question, context=haystack)

                # 检查是否正确找到
                success = "ABC123" in response

                results[length][position] = {
                    "success": success,
                    "response": response[:200]
                }

        return results

    def visualize_results(self, results: Dict):
        """可视化结果"""
        import matplotlib.pyplot as plt
        import numpy as np

        lengths = sorted(results.keys())
        positions = sorted(results[lengths[0]].keys())

        # 创建热力图数据
        data = np.zeros((len(lengths), len(positions)))
        for i, length in enumerate(lengths):
            for j, position in enumerate(positions):
                data[i][j] = 1 if results[length][position]["success"] else 0

        plt.figure(figsize=(10, 6))
        plt.imshow(data, cmap='RdYlGn', aspect='auto')
        plt.colorbar(label='Success')
        plt.xlabel('Needle Position')
        plt.ylabel('Context Length')
        plt.xticks(range(len(positions)), [f"{p:.0%}" for p in positions])
        plt.yticks(range(len(lengths)), lengths)
        plt.title('Needle in a Haystack Evaluation')
        plt.show()
```

#### 4.2.2 多跳推理评估

```python
class MultiHopEvaluator:
    """多跳推理能力评估"""

    def __init__(self, agent):
        self.agent = agent

    def create_multi_hop_question(self, hops: int = 2) -> Dict:
        """创建多跳问题"""
        # 示例：2跳问题
        if hops == 2:
            facts = [
                "张三在阿里巴巴工作",
                "阿里巴巴的总部在杭州",
            ]
            question = "张三工作的公司总部在哪个城市？"
            answer = "杭州"
            reasoning_chain = ["张三在阿里巴巴工作", "阿里巴巴总部在杭州", "所以张三工作的公司总部在杭州"]

        elif hops == 3:
            facts = [
                "李四是张三的同事",
                "张三在阿里巴巴工作",
                "阿里巴巴的创始人是马云",
            ]
            question = "李四同事所在公司的创始人是谁？"
            answer = "马云"
            reasoning_chain = [
                "李四的同事是张三",
                "张三在阿里巴巴工作",
                "阿里巴巴的创始人是马云"
            ]

        return {
            "facts": facts,
            "question": question,
            "answer": answer,
            "reasoning_chain": reasoning_chain
        }

    def evaluate(self, max_hops: int = 3) -> Dict:
        """评估多跳推理能力"""
        results = {}

        for hops in range(2, max_hops + 1):
            test_case = self.create_multi_hop_question(hops)

            # 提供facts作为上下文
            context = "\n".join(test_case["facts"])

            response = self.agent.answer(
                test_case["question"],
                context=context
            )

            # 检查答案正确性
            correct = test_case["answer"].lower() in response.lower()

            # 检查推理链
            reasoning_found = all(
                step.lower() in response.lower()
                for step in test_case["reasoning_chain"]
            )

            results[f"{hops}-hop"] = {
                "correct": correct,
                "reasoning_chain_found": reasoning_found,
                "response": response
            }

        return results
```

---

## 第五章：项目实战——OpenManus开发

### 5.1 深度解析OpenManus框架

#### 5.1.1 OpenManus项目导论

**OpenManus是什么？**

OpenManus是一个开源的AI写作Agent框架，可以自动完成从构思到成稿的完整写作流程。

```
OpenManus核心特点：
├── 多Agent协作：不同Agent负责不同写作阶段
├── 记忆管理：维护写作上下文和知识库
├── 工具集成：支持搜索、RAG等能力
└── 可扩展性：易于定制和扩展
```

#### 5.1.2 核心流程

```
┌─────────────────────────────────────────────────────────────┐
│                OpenManus写作流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户输入主题                                                │
│       ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Orchestrator（编排器）                             │    │
│  │  分析任务，制定计划，协调各Agent                     │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │Research │→│ Outline │→│ Writer  │→│ Editor  │        │
│  │ Agent   │  │ Agent   │  │ Agent   │  │ Agent   │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│       ↓            ↓            ↓            ↓             │
│    搜索资料      生成大纲      撰写内容      润色修改          │
│       ↓            ↓            ↓            ↓             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Memory（记忆）                      │    │
│  │  存储中间结果、上下文信息、知识片段                   │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Tools（工具）                       │    │
│  │  Web搜索、知识库检索、文件操作等                     │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                     │
│  最终文章输出                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 5.1.3 核心模块详解

**1. Orchestrator（编排器）**

```python
class Orchestrator:
    """编排器：协调多个Agent完成复杂任务"""

    def __init__(self, agents: Dict, memory: Memory):
        self.agents = agents
        self.memory = memory

    async def run(self, task: str) -> str:
        """执行任务"""
        # 1. 分析任务，制定计划
        plan = await self.create_plan(task)

        # 2. 按计划执行
        results = {}
        for step in plan.steps:
            agent = self.agents[step.agent_name]

            # 获取上下文
            context = self.memory.get_context(step.context_keys)

            # 执行步骤
            result = await agent.execute(step.instruction, context)

            # 存储结果
            self.memory.store(step.output_key, result)
            results[step.output_key] = result

        # 3. 整合最终结果
        final_output = self.compose_output(results, plan)
        return final_output

    async def create_plan(self, task: str) -> Plan:
        """创建执行计划"""
        prompt = f"""请为以下任务制定执行计划：

任务：{task}

可用的Agent：
- research: 搜索和收集资料
- outline: 生成文章大纲
- writer: 撰写内容
- editor: 编辑润色

请返回JSON格式的执行计划..."""

        plan_json = await self.llm.generate(prompt)
        return Plan.from_json(plan_json)
```

**2. Agents（智能体）**

```python
class BaseAgent:
    """Agent基类"""

    def __init__(self, llm, tools: List = None):
        self.llm = llm
        self.tools = tools or []
        self.system_prompt = ""

    async def execute(self, instruction: str, context: str = "") -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"上下文：\n{context}\n\n任务：{instruction}"}
        ]
        return await self.llm.generate(messages)


class ResearchAgent(BaseAgent):
    """研究Agent：负责搜索和收集资料"""

    def __init__(self, llm, search_tool, rag_tool):
        super().__init__(llm, [search_tool, rag_tool])
        self.system_prompt = """你是一个专业的研究助手。
你的任务是搜索和收集与主题相关的资料。
请使用提供的搜索工具获取信息，并整理成结构化的研究笔记。"""

    async def execute(self, topic: str, context: str = "") -> str:
        # 1. 生成搜索查询
        queries = await self.generate_search_queries(topic)

        # 2. 执行搜索
        all_results = []
        for query in queries:
            results = await self.tools[0].search(query)
            all_results.extend(results)

        # 3. 从知识库补充
        rag_results = await self.tools[1].retrieve(topic)
        all_results.extend(rag_results)

        # 4. 整理成研究笔记
        notes = await self.compile_notes(topic, all_results)
        return notes


class WriterAgent(BaseAgent):
    """写作Agent：负责撰写内容"""

    def __init__(self, llm):
        super().__init__(llm)
        self.system_prompt = """你是一位专业作家。
请根据提供的大纲和研究资料，撰写高质量的文章内容。
要求：
1. 内容准确，有理有据
2. 结构清晰，逻辑流畅
3. 语言生动，易于理解"""

    async def execute(self, instruction: str, context: str) -> str:
        # context包含大纲和研究资料
        return await super().execute(instruction, context)
```

**3. Memory（记忆）**

```python
class Memory:
    """写作流程的记忆管理"""

    def __init__(self):
        self.storage = {}
        self.history = []

    def store(self, key: str, value: any):
        """存储信息"""
        self.storage[key] = value
        self.history.append({
            "action": "store",
            "key": key,
            "timestamp": datetime.now().isoformat()
        })

    def get(self, key: str) -> any:
        """获取信息"""
        return self.storage.get(key)

    def get_context(self, keys: List[str]) -> str:
        """获取指定键的上下文"""
        context_parts = []
        for key in keys:
            if key in self.storage:
                context_parts.append(f"## {key}\n{self.storage[key]}")
        return "\n\n".join(context_parts)

    def get_full_context(self) -> str:
        """获取完整上下文"""
        return self.get_context(list(self.storage.keys()))
```

### 5.2 构建自己的AI写作助手

#### 5.2.1 完整实现

```python
import asyncio
from typing import Dict, List

class AIWritingAssistant:
    """AI写作助手完整实现"""

    def __init__(self, llm, embedding_model, vector_db, search_api):
        # 初始化工具
        self.search_tool = WebSearchTool(search_api)
        self.rag_tool = RAGTool(embedding_model, vector_db)

        # 初始化Agents
        self.agents = {
            "research": ResearchAgent(llm, self.search_tool, self.rag_tool),
            "outline": OutlineAgent(llm),
            "writer": WriterAgent(llm),
            "editor": EditorAgent(llm)
        }

        # 初始化记忆
        self.memory = Memory()

        # 初始化编排器
        self.orchestrator = Orchestrator(self.agents, self.memory)

    async def write_article(self, topic: str, style: str = "专业",
                           length: str = "中等") -> str:
        """
        生成文章

        Args:
            topic: 文章主题
            style: 写作风格（专业/通俗/学术）
            length: 文章长度（短/中等/长）
        """
        # 存储初始参数
        self.memory.store("topic", topic)
        self.memory.store("style", style)
        self.memory.store("length", length)

        # 1. 研究阶段
        print("📚 正在搜集资料...")
        research_notes = await self.agents["research"].execute(
            f"搜索并整理关于'{topic}'的相关资料"
        )
        self.memory.store("research", research_notes)

        # 2. 大纲阶段
        print("📝 正在生成大纲...")
        outline = await self.agents["outline"].execute(
            f"为'{topic}'生成文章大纲，风格：{style}，长度：{length}",
            context=research_notes
        )
        self.memory.store("outline", outline)

        # 3. 写作阶段
        print("✍️ 正在撰写文章...")
        draft = await self.agents["writer"].execute(
            "根据大纲和研究资料撰写完整文章",
            context=self.memory.get_context(["research", "outline"])
        )
        self.memory.store("draft", draft)

        # 4. 编辑阶段
        print("🔍 正在润色修改...")
        final_article = await self.agents["editor"].execute(
            f"润色修改文章，确保{style}风格，检查语法和逻辑",
            context=draft
        )
        self.memory.store("final", final_article)

        print("✅ 文章生成完成！")
        return final_article

    async def iterative_improve(self, feedback: str) -> str:
        """根据反馈迭代改进"""
        current = self.memory.get("final") or self.memory.get("draft")

        improved = await self.agents["editor"].execute(
            f"根据以下反馈修改文章：\n{feedback}",
            context=current
        )

        self.memory.store("final", improved)
        return improved


# 使用示例
async def main():
    # 初始化（需要配置API keys）
    assistant = AIWritingAssistant(
        llm=LLM(...),
        embedding_model=EmbeddingModel(...),
        vector_db=VectorDB(...),
        search_api="..."
    )

    # 生成文章
    article = await assistant.write_article(
        topic="人工智能在医疗领域的应用",
        style="通俗易懂",
        length="中等"
    )

    print(article)

    # 根据反馈改进
    improved = await assistant.iterative_improve(
        "请增加更多实际案例，并加强结论部分"
    )

    print(improved)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 专项求职辅导

### Agent相关面试问题精讲

**1. 一分钟讲清楚Agent的定义**

```
Agent = LLM + 记忆 + 工具 + 规划

核心要素：
1. 感知：理解环境和用户输入
2. 规划：制定实现目标的计划
3. 行动：调用工具执行操作
4. 记忆：保持上下文和学习

与普通LLM调用的区别：
- LLM：单轮问答
- Agent：多步推理，自主决策，工具使用
```

**2. 你如何处理Agent的幻觉问题？**

```
多层次防幻觉策略：

1. RAG增强
   - 用检索到的真实信息作为依据
   - 要求Agent引用来源

2. 工具验证
   - 涉及事实时，调用工具验证
   - 计算用计算器，查询用数据库

3. 结构化输出
   - 用JSON Schema约束输出
   - 强制要求特定字段

4. Human-in-the-Loop
   - 高风险决策需要人工审批
   - 设置确认环节

5. 自我检验
   - 让Agent检查自己的输出
   - 多次采样取一致结果
```

**3. 在你的项目中，Agent的'状态'是如何管理的？**

```
状态管理策略：

1. 短时状态（会话级）
   - 存储在内存中
   - 对话历史、当前任务状态

2. 长时状态（持久化）
   - 存储在数据库/向量库
   - 用户偏好、历史知识

3. 状态机设计
   - 明确的状态定义（思考、行动、等待）
   - 状态转换规则

4. 快照与恢复
   - 支持状态序列化
   - 可以从中间状态恢复
```

**4. 如何平衡Agent的自主性与可控性？**

```
可控性设计：

1. 权限分级
   - 低风险：自动执行
   - 中风险：执行后通知
   - 高风险：需要审批

2. 行为边界
   - 明确定义Agent能做什么、不能做什么
   - 使用白名单而非黑名单

3. 中间输出
   - 展示思考过程
   - 允许用户干预

4. 回滚机制
   - 操作可撤销
   - 保留操作日志

5. 监控告警
   - 异常行为检测
   - 实时通知
```

---

## 本模块总结

### 核心能力清单

1. **Function Calling**：掌握工具调用的设计和实现
2. **MCP协议**：理解并能搭建MCP服务
3. **数据决策**：能将ML工具集成到Agent中
4. **搜索能力**：实现Web搜索和RAG集成
5. **记忆系统**：设计短时和长时记忆
6. **效果评估**：掌握Agent评估方法

### 实践建议

1. 从简单的Function Calling开始，逐步增加复杂度
2. 重视Agent的可控性设计
3. 建立完善的评估体系
4. 多做实际项目，积累经验

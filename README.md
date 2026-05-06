# DS-Free-API (Python)

将 DeepSeek 网页端对话反代为标准 **OpenAI** 与 **Anthropic** 兼容 API。

支持 Chat Completions、Messages、流式返回、工具调用、图片/文件上传（OCR 文字识别）、联网搜索、多轮对话。

基于 FastAPI + httpx + Pydantic v2 + wasmtime PoW 求解。

---

## 特性

- **双协议兼容** — 同时提供 OpenAI `/v1/chat/completions` 和 Anthropic `/v1/messages` 端点
- **AWS WAF 绕过** — Playwright 自动通过 AWS WAF Bot Control JS Challenge，支持多账号并发登录
- **联网搜索** — 默认开启，DeepSeek 实时搜索，支持 `web_search_options` 参数
- **多轮对话** — ChatML 格式完整历史传递，确保上下文完整
- **图片/文件上传** — 支持 base64/URL 图片上传，DeepSeek OCR 文字识别
- **超多账号管理** — 账号池 + per-model-type 索引 + 熔断机制 + 后台健康监控自动恢复
- **高并发** — per-model-type 锁 + 请求排队等待 + httpx 200 连接池 + uvicorn backlog 2048
- **快速启动** — 缓存恢复 + 并行验证(8并发) + 并行创建 session + PoW WASM 实例池化
- **流式性能** — StreamState list+join O(1) + SSE 增量扫描 + SSE 反缓冲 + 双流架构
- **工具调用** — 支持 function calling / tool_use，自动解析 `<tool_calls>` 标签
- **Docker 支持** — 提供 Dockerfile，一键部署

---

## 快速开始

### 1. 安装

```bash
# 需要 Python >= 3.11
pip install -e .

# 安装 Playwright 浏览器（用于绕过 AWS WAF）
playwright install chromium
```

### 2. 配置账号

```bash
cp accounts.example.txt accounts.txt
```

编辑 `accounts.txt`，每行一个账号：

```
# 邮箱格式：email,password
your@email.com,your_password

# 手机号格式：mobile,password 或 mobile,area_code,password
# 13800138000,+86,password3
```

也支持环境变量：`DS_ACCOUNTS=email1,pass1;email2,pass2`

### 3. 启动

```bash
# 默认读取当前目录 config.toml
ds-free-api

# 指定配置路径
ds-free-api -c /path/to/config.toml
```

启动后访问 `http://127.0.0.1:5317`。

### Docker 启动

```bash
docker build -t ds-free-api .
docker run -d -p 5317:5317 \
  -v $(pwd)/accounts.txt:/app/accounts.txt \
  -v $(pwd)/config.toml:/app/config.toml \
  ds-free-api
```

---

## 配置

复制 `config.example.toml` 为 `config.toml`：

```toml
[server]
host = "127.0.0.1"
port = 5317
# accounts_file = "accounts.txt"

[deepseek]
# model_types = ["default", "expert"]
```

也支持在 `config.toml` 中直接定义账号（与 txt 文件二选一，txt 优先）：

```toml
[[accounts]]
email = "user@example.com"
password = "pass1"
```

---

## 模型

| 模型 ID | 内部类型 | 说明 |
|---------|---------|------|
| `deepseek-v4-flash` | default | 快速模式（默认） |
| `deepseek-v4-pro` | expert | 专家模式（深度思考） |

Anthropic 端点对外暴露标准 `claude-*` 模型 ID（`claude-sonnet-4-20250514`、`claude-opus-4-20250514`），内部自动映射为 deepseek 模型

---

## API 端点

### OpenAI 兼容

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 |
| GET | `/v1` | API 可达验证（Cherry Studio 等） |
| POST | `/v1/chat/completions` | Chat Completions（流式/非流式/工具调用/图片） |
| GET | `/v1/models` | 模型列表 |
| GET | `/v1/models/{id}` | 模型详情 |

### Anthropic 兼容

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/messages` | Messages API |
| POST | `/anthropic/v1/messages` | Messages API（备用路径） |
| GET | `/anthropic/v1/models` | 模型列表 |
| GET | `/anthropic/v1/models/{id}` | 模型详情 |

---

## 使用示例

### OpenAI 流式对话

```bash
curl http://127.0.0.1:5317/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "stream": true,
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### 联网搜索

```bash
curl http://127.0.0.1:5317/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "今天有什么新闻？"}],
    "web_search_options": {"search_context_size": "medium"}
  }'
```

### 图片上传（OCR）

```bash
curl http://127.0.0.1:5317/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "图片里有什么文字？"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
      ]
    }]
  }'
```

### Anthropic Messages

```bash
curl http://127.0.0.1:5317/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### CherryStudio / Open WebUI 配置

- **API Base URL**: `http://127.0.0.1:5317/v1`
- **API Key**: 任意值（如 `sk-xxx`）
- **Model**: `deepseek-v4-flash` 或 `deepseek-v4-pro`

---

## 架构

```
请求 → FastAPI handlers (app.py → handlers.py)
         │  字节级流式检测 / 统一错误映射 / SSE 反缓冲
         │
         ├─ OpenAI Adapter
         │   request.py  — 解析+校验+prompt构建+模型解析+图片提取
         │   completions.py — 对话编排（completion优先/edit fallback/图片并行/PoW并行）
         │   response.py — StreamState状态机 + stream_response(高性能) + stream_response_dual(结构化)
         │   adapter.py  — 统一入口
         │
         └─ Anthropic Compat
             request.py  — Anthropic→OpenAI 请求映射（模型/工具/thinking/web_search）
             response.py — OpenAI→Anthropic 响应映射（零二次解析，直接消费 chunk dict）
             compat.py   — 统一入口
                    ↓
              ds_core/
              ├─ accounts.py  — 账号池（per-type索引/熔断/健康监控/缓存/排队等待）
              ├─ client.py    — HTTP 客户端（智能重试/auth缓存/200连接池/流空闲超时/WAF token自动刷新）
              ├─ completions.py — 对话编排（completion+fallback/图片并行上传/完整历史ChatML）
              ├─ pow.py       — PoW 求解器（WASM实例池化/动态符号探测）
              └─ waf_bypass.py — AWS WAF 绕过（Playwright headless=False + 反检测脚本）
```

### 请求链路

**OpenAI 非流式**: `handlers → adapter.chat_completions → parse_request → v0_chat → aggregate_response → Response`

**OpenAI 流式**: `handlers → adapter.chat_completions_stream → parse_request → v0_chat → stream_response → StreamingResponse`

**Anthropic 非流式**: `handlers → compat.messages → to_openai_request → adapter.chat_completions → from_chat_completion_bytes`

**Anthropic 流式**: `handlers → compat.messages_stream → to_openai_request → adapter.chat_completions_stream_dual → stream_response_dual → from_chat_completion_stream_structured → StreamingResponse`

---

## 优化清单

| 优化项 | 说明 |
|--------|------|
| PoW 实例池化 | 预创建4个 WASM Store+Instance，solve 直接取用 ~10x |
| StreamState list+join | string += O(n²) → list.append O(1)，长回复毫秒级 |
| 双流架构 | `stream_response`(高性能bytes) + `stream_response_dual`(bytes+dict)，Anthropic 零二次解析 |
| SSE 反缓冲 | 移除 GZipMiddleware，SSE 流跳过压缩+添加 X-Accel-Buffering:no，确保逐 chunk 到达 |
| 字节级流式检测 | `b'"stream":true' in body.replace(b' ',b'')` 替代 json.loads，零解析开销 |
| 手动 SSE 序列化 | `_chunk_to_json_data` 手动构建 dict，避免 model_dump_json 开销 |
| per-model-type 索引 | `_by_type` dict 直接定位候选，100账号 O(100)→O(k) |
| 请求排队等待 | 无账号时排队30s而非立即报错，Event 通知释放 |
| 并行验证缓存 | 8并发验证缓存账号，启动速度 ~8x |
| 并行创建 session | asyncio.gather 并行创建所有 model_type session |
| 图片并行上传 | 多图 asyncio.gather 并行上传，与 PoW 计算并行 |
| 渐进式文件轮询 | 0.5s→2.5s 渐进间隔，首次检测快3x |
| auth headers 缓存 | per-token dict 缓存，避免每次请求创建新 dict |
| SSE 增量扫描 | 手动扫描替代 split("\n")，减少内存分配 |
| 缓存异步写入 | run_in_executor 不阻塞事件循环 |
| 条件 Gzip 压缩 | 非 SSE JSON 响应 >1KB 自动压缩，SSE 流不压缩 |
| 请求体限制 | 10MB 上限防 OOM |
| 后台健康监控 | 每60s 自动恢复不健康账号 |
| 熔断机制 | 失败3次进入5分钟冷却，选择时跳过 |
| 连接池调优 | httpx 200连接/100 keepalive，uvicorn backlog 2048 |
| 智能重试 | 指数退避+抖动，login 遇到 202/405 自动刷新 WAF token 重试 |
| AWS WAF 绕过 | Playwright headless=False + 反检测脚本，多账号并发登录自动续期 WAF token |
| 流空闲超时 | 60s 无数据自动断开，防止挂起 |
| 完整历史模式 | ChatML 格式传递完整对话历史，不依赖 parent_message_id |
| 联网搜索 | 默认开启，`web_search_options` 可选配置，有文件附件时自动禁用 |

---

## 许可证

Apache License 2.0

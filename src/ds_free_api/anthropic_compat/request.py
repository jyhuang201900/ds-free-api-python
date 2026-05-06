"""Anthropic 请求映射 —— 将 Anthropic Messages 请求映射为 OpenAI ChatCompletion 请求"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger("ds_free_api.anthropic_compat.request")


class AnthropicCompatError(Exception):
    """Anthropic 兼容层错误"""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.message = message
        super().__init__(message)

    @classmethod
    def bad_request(cls, msg: str) -> AnthropicCompatError:
        return cls("bad_request", msg)

    @classmethod
    def overloaded(cls) -> AnthropicCompatError:
        return cls("overloaded", "service overloaded")

    @classmethod
    def internal(cls, msg: str) -> AnthropicCompatError:
        return cls("internal", msg)

    def status_code(self) -> int:
        return {"bad_request": 400, "overloaded": 429, "internal": 500}.get(self.kind, 500)


# ============================================================================
# Anthropic 请求类型
# ============================================================================


class MessagesRequest(BaseModel):
    """POST /v1/messages 请求体"""

    model: str
    messages: list[dict]
    max_tokens: int
    system: Any = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tools: list[dict] | None = None
    tool_choice: Any = None
    thinking: Any = None
    metadata: dict | None = None
    web_search_options: dict | None = None

    model_config = {"extra": "allow"}


# 以下类型仅用于映射函数内部，不再用于 Pydantic 验证


def _parse_content(value: Any) -> tuple[str | None, list[dict] | None]:
    """解析 Anthropic content 字段 → (text, blocks)"""
    if isinstance(value, str):
        return (value, None)
    if isinstance(value, list):
        return (None, value)
    return (None, None)


def _parse_system(value: Any) -> tuple[str | None, list[dict] | None]:
    """解析 Anthropic system 字段 → (text, blocks)"""
    if isinstance(value, str):
        return (value, None)
    if isinstance(value, list):
        return (None, value)
    return (None, None)


def _parse_tool_choice(value: Any) -> tuple[str, str | None, bool]:
    """解析 Anthropic tool_choice → (type, name, disable_parallel)"""
    if isinstance(value, str):
        return (value, None, False)
    if isinstance(value, dict):
        return (
            value.get("type", "auto"),
            value.get("name"),
            value.get("disable_parallel_tool_use", False),
        )
    return ("auto", None, False)


def _parse_thinking(value: Any) -> tuple[str, int | None]:
    """解析 Anthropic thinking → (type, budget_tokens)"""
    if isinstance(value, dict):
        return (value.get("type", "disabled"), value.get("budget_tokens"))
    return ("disabled", None)


# ============================================================================
# 映射函数
# ============================================================================


def to_openai_request(body: bytes) -> tuple[bytes, str]:
    """将 Anthropic Messages 请求 JSON 映射为 OpenAI ChatCompletion 请求 JSON

    返回 (openai_body, original_model_id)，original_model_id 保留原始 Anthropic 模型名，
    供响应层回写，使客户端看到标准 claude-* 模型 ID。
    """
    try:
        raw = json.loads(body)
    except json.JSONDecodeError as e:
        raise AnthropicCompatError.bad_request(f"bad request: {e}")

    req = MessagesRequest.model_validate(raw, strict=False)

    original_model = req.model  # 保留原始模型 ID

    openai: dict[str, Any] = {}
    # 模型 ID 映射：Anthropic 客户端可能发送 claude-* 模型名
    openai["model"] = _map_model_id(req.model)
    openai["max_tokens"] = req.max_tokens

    # messages
    messages = []
    if req.system:
        messages.append(_system_to_openai(req.system))
    for msg in req.messages:
        messages.extend(_message_param_to_openai(msg))
    openai["messages"] = messages

    if req.stream:
        openai["stream"] = True

    if req.stop_sequences:
        openai["stop"] = req.stop_sequences

    if req.temperature is not None:
        openai["temperature"] = req.temperature

    if req.top_p is not None:
        openai["top_p"] = req.top_p

    # tools
    parallel_disabled = False
    if req.tools:
        openai_tools = _tools_to_openai(req.tools)
        if openai_tools:
            openai["tools"] = openai_tools

    # tool_choice
    if req.tool_choice:
        tc_type, tc_name, tc_disable = _parse_tool_choice(req.tool_choice)
        parallel_disabled = tc_disable
        tc_value = _tool_choice_to_openai_raw(tc_type, tc_name)
        if tc_value is not None:
            openai["tool_choice"] = tc_value
        elif tc_type == "none":
            # tool_choice=none: 移除 tools，不传 tool_choice
            openai.pop("tools", None)

    if parallel_disabled:
        openai["parallel_tool_calls"] = False

    # thinking → reasoning_effort
    if req.thinking:
        think_type, _ = _parse_thinking(req.thinking)
        if think_type in ("enabled", "adaptive"):
            openai["reasoning_effort"] = "high"
        else:
            openai["reasoning_effort"] = "minimal"

    # web_search_options：仅在用户显式提供时传递
    if req.web_search_options:
        openai["web_search_options"] = req.web_search_options

    return json.dumps(openai).encode(), original_model


def _system_to_openai(system: Any) -> dict:
    text, blocks = _parse_system(system)
    if not text and blocks:
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        text = "\n".join(texts)
    if not text:
        text = ""
    return {"role": "system", "content": text}


def _message_param_to_openai(msg: dict) -> list[dict]:
    role = msg.get("role", "user")
    content = msg.get("content")
    text, blocks = _parse_content(content)

    # 纯文本
    if text is not None:
        return [{"role": role, "content": text}]

    blocks = blocks or []

    if role == "assistant":
        return _assistant_blocks_to_openai(blocks)
    elif role == "user":
        return _user_blocks_to_openai(blocks)
    else:
        t = _extract_text(blocks)
        return [{"role": role, "content": t}]


def _assistant_blocks_to_openai(blocks: list[dict]) -> list[dict]:
    texts = []
    tool_calls = []

    for block in blocks:
        btype = block.get("type", "")
        if btype == "text":
            texts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                },
            })
        # thinking / redacted_thinking / other → 跳过

    msg: dict[str, Any] = {"role": "assistant"}
    msg["content"] = "\n".join(texts) if texts else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return [msg]


def _user_blocks_to_openai(blocks: list[dict]) -> list[dict]:
    text_parts = []
    image_parts = []
    tool_results = []

    for block in blocks:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "image":
            # 转为 OpenAI image_url 格式（保留 base64 供上传）
            source = block.get("source", {})
            if source.get("type") == "base64":
                url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                image_parts.append(url)
            elif source.get("type") == "url":
                image_parts.append(source.get("url", ""))
            else:
                text_parts.append("[image]")
        elif btype == "document":
            # DeepSeek 不支持 PDF/文档，转为占位符
            source = block.get("source", {})
            if source.get("type") == "base64":
                text_parts.append("[document]")
            elif source.get("type") == "url":
                text_parts.append(f"[document: {source.get('url', '')}]")
            elif source.get("type") == "content":
                # 内联文本内容
                content = source.get("content", "")
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    text_parts.append(_extract_text(content))
            else:
                text_parts.append("[document]")
        elif btype == "tool_result":
            tool_use_id = block.get("tool_use_id", "")
            result_content = block.get("content")
            if isinstance(result_content, str):
                text = result_content
            elif isinstance(result_content, list):
                text = _extract_text(result_content)
            else:
                text = ""
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": text,
            })

    result = []

    if text_parts or image_parts:
        if not image_parts:
            result.append({"role": "user", "content": "\n".join(text_parts)})
        else:
            # 多模态：文本 + 图片
            parts = []
            for t in text_parts:
                parts.append({"type": "text", "text": t})
            for url in image_parts:
                parts.append({"type": "image_url", "image_url": {"url": url}})
            result.append({"role": "user", "content": parts})

    result.extend(tool_results)
    return result


def _extract_text(blocks: list[dict]) -> str:
    return "\n".join(
        b.get("text", "") for b in blocks if b.get("type") == "text"
    )


def _tools_to_openai(tools: list[dict]) -> list[dict]:
    result = []
    for tool in tools:
        ttype = tool.get("type", "")
        if ttype in ("custom", "") or ttype is None:
            name = tool.get("name", "")
            if not name:
                continue
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        # 服务器工具（bash / web_search 等）忽略
    return result


def _tool_choice_to_openai_raw(tc_type: str, tc_name: str | None) -> Any:
    if tc_type == "auto":
        return "auto"
    elif tc_type in ("any", "required"):
        return "required"
    elif tc_type == "tool" and tc_name:
        return {"type": "function", "function": {"name": tc_name}}
    elif tc_type == "none":
        # OpenAI 不支持 "none"，通过不传 tool_choice + 不传 tools 实现
        return None
    return "auto"


# Anthropic 模型 ID → deepseek 模型 ID 映射
# 让使用 claude-* 模型名的客户端能正常工作
_ANTHROPIC_MODEL_MAP: dict[str, str] = {
    # Claude 3.5 Sonnet → deepseek-v4-flash (主力对话模型)
    "claude-3-5-sonnet-latest": "deepseek-v4-flash",
    "claude-3-5-sonnet-20241022": "deepseek-v4-flash",
    "claude-3-5-sonnet-20240620": "deepseek-v4-flash",
    "claude-3.5-sonnet": "deepseek-v4-flash",
    # Claude 3 Opus → deepseek-v4-pro (推理模型)
    "claude-3-opus-latest": "deepseek-v4-pro",
    "claude-3-opus-20240229": "deepseek-v4-pro",
    "claude-3-opus": "deepseek-v4-pro",
    # Claude 3 Haiku → deepseek-v4-flash (快速模型)
    "claude-3-haiku-20240307": "deepseek-v4-flash",
    "claude-3-haiku": "deepseek-v4-flash",
    # Claude 3.5 Haiku → deepseek-v4-flash
    "claude-3-5-haiku-latest": "deepseek-v4-flash",
    "claude-3-5-haiku-20241022": "deepseek-v4-flash",
    "claude-3.5-haiku": "deepseek-v4-flash",
    # Claude 4 Sonnet → deepseek-v4-flash (Claude Code 默认)
    "claude-sonnet-4-20250514": "deepseek-v4-flash",
    "claude-4-sonnet": "deepseek-v4-flash",
    # Claude 4 Opus → deepseek-v4-pro
    "claude-opus-4-20250514": "deepseek-v4-pro",
    "claude-4-opus": "deepseek-v4-pro",
    # Claude 3.7 Sonnet → deepseek-v4-flash
    "claude-3-7-sonnet-latest": "deepseek-v4-flash",
    "claude-3-7-sonnet-20250219": "deepseek-v4-flash",
    "claude-3.7-sonnet": "deepseek-v4-flash",
    # Claude Sonnet 4 → deepseek-v4-flash
    # Claude Opus 4 → deepseek-v4-pro
    # Claude Sonnet 4.5 → deepseek-v4-flash
    "claude-sonnet-4-5-20250514": "deepseek-v4-flash",
    # Claude Sonnet 4.6 → deepseek-v4-flash
    "claude-sonnet-4-6-20250603": "deepseek-v4-flash",
    # Claude Opus 4.7 → deepseek-v4-pro
    "claude-opus-4-7-20250603": "deepseek-v4-pro",
    # Claude Haiku 4.5 → deepseek-v4-flash
    "claude-haiku-4-5-20250514": "deepseek-v4-flash",
}


def _map_model_id(model: str) -> str:
    """将 Anthropic 模型 ID 映射为 deepseek 模型 ID

    - 精确匹配已知 claude-* 模型名
    - 以 claude- 开头的未知模型默认映射为 deepseek-chat
    - 其他模型名原样传递（如已经是 deepseek-*）
    """
    if model in _ANTHROPIC_MODEL_MAP:
        return _ANTHROPIC_MODEL_MAP[model]
    if model.startswith("claude-"):
        return "deepseek-v4-flash"
    return model

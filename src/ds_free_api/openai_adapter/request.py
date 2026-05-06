"""OpenAI 请求解析 —— 将 OpenAI ChatCompletion 请求降级为 ds_core::ChatRequest

完整历史模式：
- 将所有 messages 转换为 ChatML 格式的完整对话历史
- 确保上下文完整传递给 DeepSeek，不依赖 parent_message_id
- 支持 system/user/assistant/tool 角色和 tool_calls
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any, TypedDict

import tiktoken

from ..ds_core.completions import ChatRequest
from .types import (
    ChatCompletionRequest,
    FunctionDefinition,
    NamedFunction,
    Tool,
    ToolChoice,
)

logger = logging.getLogger("ds_free_api.openai_adapter.request")

# 缓存 tiktoken 编码器，避免每次请求重复加载
_tiktoken_enc: tiktoken.Encoding | None = None


def _get_tiktoken_enc() -> tiktoken.Encoding | None:
    """获取 tiktoken 编码器（sync lazy init，线程安全）"""
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass
    return _tiktoken_enc


@dataclass
class AdapterRequest:
    """解析并降级后的请求上下文"""

    model: str
    ds_req: ChatRequest
    stream: bool = False
    include_usage: bool = False
    include_obfuscation: bool = False
    stop: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    image_attachments: list[dict] = field(default_factory=list)  # [{filename, content_type, data}]


async def parse(body: bytes, registry: dict[str, str], *, _validate_model: bool = True) -> AdapterRequest:
    """解析 JSON 请求体（优化性能：使用快速验证模式）"""
    try:
        raw = json.loads(body)
    except json.JSONDecodeError as e:
        raise BadRequestError(f"bad request: {e}")

    # 使用 Python 模式验证（比 JSON 模式更快，因为跳过了类型转换）
    req = ChatCompletionRequest.model_validate(raw, strict=False)

    # 兼容旧版 functions / function_call → tools / tool_choice
    if (not req.tools or len(req.tools) == 0) and req.functions:
        req.tools = [
            Tool(type="function", function=f)
            for f in req.functions
        ]

    if req.tool_choice is None and req.function_call is not None:
        # function_call → tool_choice 映射
        if isinstance(req.function_call, str):
            req.tool_choice = ToolChoice(mode=req.function_call)
        elif isinstance(req.function_call, dict):
            name = req.function_call.get("name", "")
            req.tool_choice = ToolChoice(
                type="function",
                function=NamedFunction(name=name),
            )

    # 校验
    _validate(req, registry)

    # 构建 prompt
    tool_ctx = _extract_tools(req)
    prompt = _build_prompt(req, tool_ctx)

    # 模型解析
    model_res = _resolve_model(
        registry, req.model,
        req.reasoning_effort,
        req.web_search_options,
    )

    # 计算 prompt tokens
    prompt_tokens = 0
    enc = _get_tiktoken_enc()
    if enc:
        try:
            prompt_tokens = len(enc.encode(prompt))
        except Exception:
            pass

    # stop 序列
    stop: list[str] = []
    if req.stop:
        stop = req.stop.items

    # stream options
    include_usage = False
    include_obfuscation = False
    if req.stream_options:
        include_usage = req.stream_options.include_usage
        include_obfuscation = req.stream_options.include_obfuscation

    return AdapterRequest(
        model=req.model,
        ds_req=ChatRequest(
            prompt=prompt,
            thinking_enabled=model_res["thinking_enabled"],
            search_enabled=model_res["search_enabled"],
            model_type=model_res["model_type"],
            ref_file_ids=[],  # 由 completions 层上传图片后填充
        ),
        stream=req.stream,
        include_usage=include_usage,
        include_obfuscation=include_obfuscation,
        stop=stop,
        prompt_tokens=prompt_tokens,
        image_attachments=_extract_images(req),
    )


class BadRequestError(Exception):
    """请求格式错误"""

    pass


class ToolContext(TypedDict):
    tools: list[dict]
    tool_choice_mode: str
    named_tool: str | None
    parallel_tool_calls: bool | None


# ============================================================================
# 校验
# ============================================================================


def _validate(req: ChatCompletionRequest, registry: dict[str, str] | None = None) -> None:
    """校验请求有效性（单次遍历优化）"""
    # 基础校验
    if not req.model:
        raise BadRequestError("model 不能为空")
    if not req.messages:
        raise BadRequestError("messages 不能为空")

    # 模型校验：使用 registry 动态校验（如果提供）
    if registry is not None:
        if req.model.lower() not in registry:
            raise BadRequestError(f"不支持的模型: {req.model}")

    # 消息长度限制（防止滥用）
    if len(req.messages) > 500:
        raise BadRequestError("消息数量不能超过 500")

    # 单次遍历：恶意检测 + tool 校验
    for msg in req.messages:
        content = msg.content.get_text() if msg.content else ""

        if len(content) > 100000:
            raise BadRequestError("单条消息过长（超过 100k 字符）")
        if content.count("{{") > 10 or content.count("}}") > 10:
            raise BadRequestError("消息内容包含过多模板语法")
        if msg.role == "tool" and not msg.tool_call_id:
            raise BadRequestError("tool 消息必须提供 tool_call_id")


# ============================================================================
# 工具提取
# ============================================================================


def _extract_tools(req: ChatCompletionRequest) -> ToolContext:
    """提取工具上下文"""
    if not req.tools:
        return {"tools": [], "tool_choice_mode": "none", "named_tool": None, "parallel_tool_calls": None}

    # tool_choice=none 时忽略工具
    choice_mode = "auto"
    if req.tool_choice:
        if isinstance(req.tool_choice.mode, str):
            choice_mode = req.tool_choice.mode
        elif isinstance(req.tool_choice.type, str):
            choice_mode = req.tool_choice.type

    if choice_mode == "none":
        return {"tools": [], "tool_choice_mode": "none", "named_tool": None, "parallel_tool_calls": None}

    tools_list = []
    for t in req.tools:
        if t.function and t.function.name:
            tools_list.append({
                "name": t.function.name,
                "description": t.function.description or "",
                "parameters": t.function.parameters,
                "type": t.type,
            })

    named_tool = None
    if req.tool_choice and req.tool_choice.function:
        named_tool = req.tool_choice.function.name

    return {
        "tools": tools_list,
        "tool_choice_mode": choice_mode,
        "named_tool": named_tool,
        "parallel_tool_calls": req.parallel_tool_calls,
    }


# ============================================================================
# Prompt 构建（多轮对话模式）
# ============================================================================


# DeepSeek token 限制
MAX_PROMPT_TOKENS = 80000  # 80k 安全阈值（DeepSeek 限制 100k）


def _build_prompt(req: ChatCompletionRequest, tool_ctx: dict) -> str:
    """构建 prompt（完整历史模式 + 智能截断）

    将 OpenAI messages 转换为 ChatML 格式的完整对话历史，
    当超过 token 限制时，从旧到新截断历史消息。
    """
    # 构建 tool reminder 块
    tool_reminder = ""
    tools = tool_ctx.get("tools", [])
    if tools:
        tool_lines = ["你可以使用以下工具："]
        for t in tools:
            desc = t.get("description", "")
            name = t.get("name", "")
            params = t.get("parameters", {})
            if desc:
                tool_lines.append(f"- {name}: {desc}")
            else:
                tool_lines.append(f"- {name}")
            if params and isinstance(params, dict) and params.get("properties"):
                tool_lines.append(f"  参数: {params}")

        # tool_choice 指令
        choice_mode = tool_ctx.get("tool_choice_mode", "auto")
        if choice_mode in ("required", "any"):
            tool_lines.append("注意：你必须调用一个或多个工具")
        if tool_ctx.get("parallel_tool_calls") is False:
            tool_lines.append("注意：一次只能调用一个工具")
        named = tool_ctx.get("named_tool")
        if named:
            tool_lines.append(f"注意：你必须调用 '{named}' 工具")

        tool_lines.append("调用工具时，请使用 <tool_calls>[{...}]</tool_calls> 格式")

        # response_format 处理
        if req.response_format:
            fmt_type = req.response_format.type
            if fmt_type == "json_object":
                tool_lines.append("请直接输出合法的 JSON 对象")
            elif fmt_type == "json_schema" and req.response_format.json_schema:
                tool_lines.append(f"请按 JSON Schema 输出: {req.response_format.json_schema}")

        tool_reminder = "\n".join(tool_lines)

    # 分离 system 消息和对话消息（合并多个 system）
    system_parts = []
    conversation_msgs = []
    for msg in req.messages:
        if msg.role == "system":
            system_parts.append(_get_message_text(msg))
        else:
            conversation_msgs.append(msg)
    system_content = "\n\n".join(p for p in system_parts if p)

    # 构建 system 部分（固定，不截断）
    system_part = ""
    if system_content or tool_reminder:
        content = system_content
        if tool_reminder:
            content = f"{content}\n\n{tool_reminder}" if content else tool_reminder
        system_part = f"<|im_start|>system\n{content}<|im_end|>"

    # 计算固定部分的 token 数
    enc = _get_tiktoken_enc()
    fixed_tokens = 0
    if enc:
        fixed_tokens = len(enc.encode(system_part)) if system_part else 0
        fixed_tokens += 10  # <|im_start|>assistant 预留

    # 从新到旧遍历，保留能放下的消息
    kept_parts: list[str] = []
    current_tokens = fixed_tokens

    # 倒序遍历（从最新消息开始）
    for msg in reversed(conversation_msgs):
        role = msg.role
        content = _get_message_text(msg)

        if role == "user":
            part = f"<|im_start|>user\n{content}<|im_end|>"
        elif role == "assistant":
            if msg.tool_calls:
                tool_calls_str = _format_tool_calls(msg.tool_calls)
                if content:
                    content = f"{content}\n{tool_calls_str}"
                else:
                    content = tool_calls_str
            part = f"<|im_start|>assistant\n{content}<|im_end|>"
        elif role == "tool":
            tool_call_id = msg.tool_call_id or "unknown"
            part = f"<|im_start|>tool\ntool_call_id: {tool_call_id}\n{content}<|im_end|>"
        else:
            continue

        # 计算 token 数
        part_tokens = len(enc.encode(part)) if enc else len(content) // 2

        # 检查是否超限
        if current_tokens + part_tokens > MAX_PROMPT_TOKENS:
            # 超限，停止添加更旧的消息
            logger.debug(f"历史截断: 保留 {len(kept_parts)} 条消息，{current_tokens} tokens")
            break

        kept_parts.append(part)
        current_tokens += part_tokens

    # 反转回正序
    kept_parts.reverse()

    # 确保至少有一条对话消息（避免空历史导致模型异常）
    if not kept_parts and conversation_msgs:
        # 强制保留最后一条消息
        last_msg = conversation_msgs[-1]
        role = last_msg.role
        content = _get_message_text(last_msg)
        if role == "user":
            kept_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            if last_msg.tool_calls:
                tool_calls_str = _format_tool_calls(last_msg.tool_calls)
                content = f"{content}\n{tool_calls_str}" if content else tool_calls_str
            kept_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif role == "tool":
            tool_call_id = last_msg.tool_call_id or "unknown"
            kept_parts.append(f"<|im_start|>tool\ntool_call_id: {tool_call_id}\n{content}<|im_end|>")

    # 组装最终 prompt
    parts: list[str] = []
    if system_part:
        parts.append(system_part)
    parts.extend(kept_parts)
    parts.append("<|im_start|>assistant")

    final_prompt = "\n".join(parts)

    # 日志：最终 token 数
    if enc:
        final_tokens = len(enc.encode(final_prompt))
        logger.debug(f"prompt 构建: {final_tokens} tokens, {len(kept_parts)} 条消息")

    return final_prompt


def _format_tool_calls(tool_calls: list) -> str:
    """将 tool_calls 转换为 <tool_calls> 格式"""
    calls = []
    for tc in tool_calls:
        if tc.function:
            call_dict = {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
            calls.append(call_dict)
    return f"<tool_calls>{json.dumps(calls, ensure_ascii=False)}</tool_calls>"


def _get_message_text(msg) -> str:
    """获取消息文本内容（图片以占位符替代）"""
    if msg.content is None:
        return ""
    if isinstance(msg.content, str):
        return msg.content
    # 多模态内容：提取文本 + 图片占位
    if msg.content.parts:
        parts = []
        for p in msg.content.parts:
            if p.text:
                parts.append(p.text)
            elif p.image_url:
                parts.append("[image]")
        return "\n".join(parts)
    return msg.content.get_text()


def _extract_images(req) -> list[dict]:
    """从请求中提取图片附件，返回 [{filename, content_type, data}]"""
    images = []
    idx = 0
    for msg in req.messages:
        if msg.content is None or isinstance(msg.content, str):
            continue
        if msg.content.parts:
            for p in msg.content.parts:
                if p.image_url and p.image_url.url:
                    url = p.image_url.url
                    # data:image/png;base64,xxxxx
                    if url.startswith("data:"):
                        # 解析 data URI
                        try:
                            header, encoded = url.split(",", 1)
                            # header: data:image/png;base64
                            mime = header.split(":")[1].split(";")[0]
                            ext = mime.split("/")[-1] if "/" in mime else "png"
                            images.append({
                                "filename": f"image_{idx}.{ext}",
                                "content_type": mime,
                                "data": base64.b64decode(encoded),
                            })
                            idx += 1
                        except Exception:
                            pass
                    # http/https URL — 下载图片
                    elif url.startswith("http"):
                        images.append({
                            "filename": f"image_{idx}.jpg",
                            "content_type": "image/jpeg",
                            "data": None,  # 标记需要下载
                            "url": url,
                        })
                        idx += 1
    return images


# ============================================================================
# 模型解析
# ============================================================================


def _resolve_model(
    registry: dict[str, str],
    model: str,
    reasoning_effort: str | None,
    web_search_options: Any | None,
) -> dict:
    """解析模型类型和能力标志"""
    model_lower = model.lower()
    model_type = registry.get(model_lower)
    if model_type is None:
        raise BadRequestError(f"不支持的模型: {model}")

    # thinking 默认开启
    thinking_enabled = True
    if reasoning_effort == "none":
        thinking_enabled = False
    elif reasoning_effort and reasoning_effort in ("minimal", "low", "medium", "high", "xhigh"):
        thinking_enabled = True

    # search 默认开启
    search_enabled = True
    if web_search_options is not None:
        # 支持 search_context_size="none" 显式关闭搜索
        ctx_size = None
        if isinstance(web_search_options, dict):
            ctx_size = web_search_options.get("search_context_size")
        elif hasattr(web_search_options, "search_context_size"):
            ctx_size = web_search_options.search_context_size
        if ctx_size == "none":
            search_enabled = False

    return {
        "model_type": model_type,
        "thinking_enabled": thinking_enabled,
        "search_enabled": search_enabled,
    }

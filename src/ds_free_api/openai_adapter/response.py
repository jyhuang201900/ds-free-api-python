"""OpenAI 响应转换 —— 将 DeepSeek SSE 流映射为 OpenAI 响应格式

数据流：sse_parser -> state -> converter -> tool_parser
"""

from __future__ import annotations

import base64
import itertools
import json
import logging
import os
import time
from typing import Any, AsyncIterator

from ..ds_core.completions import CoreError
from .types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    ChunkChoice,
    Delta,
    MessageResponse,
    ToolCall,
    FunctionCall,
    Usage,
)

logger = logging.getLogger("ds_free_api.openai_adapter.response")

FINISH_STOP = "stop"
FINISH_TOOL_CALLS = "tool_calls"

# 全局 ID 计数器
_chatcmpl_counter = itertools.count(1)


def _next_chatcmpl_id() -> str:
    return f"chatcmpl-{next(_chatcmpl_counter):016x}"


def _now_secs() -> int:
    return int(time.time())


# Obfuscation
OBFUSCATION_TARGET_LEN = 512
OBFUSCATION_MIN_PAD = 16


def _random_padding(length: int) -> str:
    if length == 0:
        return ""
    byte_len = (length * 3 + 3) // 4
    random_bytes = os.urandom(byte_len)
    s = base64.b64encode(random_bytes).decode()
    return s[:length]


class AdapterError(Exception):
    """适配器错误"""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.message = message
        super().__init__(message)

    @classmethod
    def bad_request(cls, msg: str) -> AdapterError:
        return cls("bad_request", msg)

    @classmethod
    def overloaded(cls) -> AdapterError:
        return cls("overloaded", "service overloaded")

    @classmethod
    def provider_error(cls, msg: str) -> AdapterError:
        return cls("provider_error", msg)

    @classmethod
    def internal(cls, msg: str) -> AdapterError:
        return cls("internal", msg)

    def status_code(self) -> int:
        return {
            "bad_request": 400,
            "overloaded": 429,
            "provider_error": 502,
            "internal": 500,
        }.get(self.kind, 500)

    @classmethod
    def from_core_error(cls, e: CoreError) -> AdapterError:
        mapping = {
            "overloaded": cls.overloaded(),
            "provider": cls.provider_error(e.message),
            "stream": cls.internal(e.message),
            "pow": cls.internal(f"proof of work failed: {e.message}"),
        }
        return mapping.get(e.kind, cls.internal(e.message))


# ============================================================================
# SSE 解析
# ============================================================================


def parse_sse_events(raw: bytes) -> list[dict]:
    """解析 SSE 事件流为事件列表（增量扫描，避免 split 创建大列表）"""
    text = raw.decode("utf-8", errors="replace")
    events = []
    pos = 0
    text_len = len(text)
    prefix = "data: "
    prefix_len = 6

    while pos < text_len:
        # 跳过空白
        while pos < text_len and text[pos] in " \t\r\n":
            pos += 1
        if pos >= text_len:
            break

        # 检查 data: 前缀
        if text[pos:pos + prefix_len] == prefix:
            pos += prefix_len
            # 找行尾
            end = pos
            while end < text_len and text[end] != '\n' and text[end] != '\r':
                end += 1
            data_str = text[pos:end]
            pos = end

            if data_str == "[DONE]":
                continue
            try:
                events.append(json.loads(data_str))
            except json.JSONDecodeError:
                continue
        else:
            # 跳过非 data: 行
            while pos < text_len and text[pos] != '\n' and text[pos] != '\r':
                pos += 1

    return events

# ============================================================================
# DeepSeek SSE 状态机
# ============================================================================


class StreamState:
    """跟踪 DeepSeek SSE 流的状态

    对齐 Rust 版 DsState 逻辑：
    - 维护 current_path：有 p 字段时更新，后续纯 v 事件复用
    - response/fragments/-1/content 路径一律 APPEND（push_str）
    - 无 p 无 o 的纯 v 事件：用 current_path + APPEND 处理
    """

    def __init__(self):
        self._content_parts: list[str] = []  # list append O(1) 替代 string += O(n²)
        self._reasoning_parts: list[str] = []
        self.completion_tokens = 0
        self.finished = False
        self.current_path: str | None = None
        self._fragment_type: str = "RESPONSE"  # 当前 fragment 类型：THINK / RESPONSE / TOOL_SEARCH / TOOL_OPEN
        self.message_id: int | None = None  # 从 SSE 提取的 response message_id
        self.request_message_id: int | None = None  # 用户消息 ID

    @property
    def content(self) -> str:
        """惰性拼接：仅在访问时 join"""
        return "".join(self._content_parts)

    @property
    def reasoning(self) -> str:
        """惰性拼接：仅在访问时 join"""
        return "".join(self._reasoning_parts)

    def process_event(self, event: dict) -> None:
        """处理单个 SSE 事件"""
        # 处理 ready 事件：顶层 response_message_id
        if "response_message_id" in event:
            self.message_id = event["response_message_id"]
            self.request_message_id = event.get("request_message_id")
            return

        has_p = "p" in event
        op = event.get("o")
        value = event.get("v")

        if value is None:
            return

        # 有 p 字段时更新 current_path
        if has_p:
            self.current_path = event.get("p")

        # 有 p 或有 o → 走 apply_path
        if has_p or op is not None:
            if self.current_path:
                self._apply_path(self.current_path, op, value)

        # 无 p 无 o 但有 current_path → 视为 APPEND
        elif self.current_path is not None:
            self._apply_path(self.current_path, "APPEND", value)

        # 无 p 无 o 且无 current_path → 初始 snapshot
        elif isinstance(value, dict):
            resp = value.get("response", {})
            # 提取 message_id（真正的多轮对话关键）
            if resp.get("message_id"):
                self.message_id = resp["message_id"]
            fragments = resp.get("fragments", [])
            for frag in fragments:
                frag_type = frag.get("type", "")
                content = frag.get("content", "")
                if frag_type in ("THINK", "RESPONSE", "TOOL_SEARCH", "TOOL_OPEN"):
                    self._fragment_type = frag_type
                if content and frag_type == "THINK":
                    self._reasoning_parts.append(content)
                elif content and frag_type == "RESPONSE":
                    self._content_parts.append(content)

    def _apply_path(self, path: str, op: str | None, value: Any) -> None:
        """处理路径操作，对齐 Rust apply_path"""
        if path == "response/status":
            if value == "FINISHED":
                self.finished = True

        elif path in ("response/accumulated_token_usage", "accumulated_token_usage"):
            if isinstance(value, (int, float)):
                self.completion_tokens = int(value)

        elif path == "response/fragments/-1/content":
            # 一律 APPEND（push_str），不区分 APPEND/SET
            if isinstance(value, str):
                ft = self._fragment_type
                if ft == "THINK":
                    self._reasoning_parts.append(value)
                elif ft == "RESPONSE":
                    self._content_parts.append(value)

        elif path == "response/fragments" and op == "APPEND":
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        frag_type = item.get("type", "")
                        content = item.get("content", "")
                        if frag_type in ("THINK", "RESPONSE", "TOOL_SEARCH", "TOOL_OPEN"):
                            self._fragment_type = frag_type
                        if content and frag_type == "THINK":
                            self._reasoning_parts.append(content)
                        elif content and frag_type == "RESPONSE":
                            self._content_parts.append(content)

        elif path == "response" and op == "BATCH":
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        sub_p = item.get("p", "")
                        sub_v = item.get("v")
                        if sub_p == "accumulated_token_usage" and isinstance(sub_v, (int, float)):
                            self.completion_tokens = int(sub_v)
                        elif sub_p == "quasi_status" and sub_v == "FINISHED":
                            self.finished = True

    @property
    def finish_reason(self) -> str | None:
        """获取 finish_reason：finished=True 返回 stop，否则 None"""
        return FINISH_STOP if self.finished else None


# ============================================================================
# Tool 调用解析
# ============================================================================


def parse_tool_calls(content: str) -> tuple[list[ToolCall], str] | None:
    """从内容中解析所有 <tool_calls>...</tool_calls> 标签

    返回 (tool_calls, remaining_content) 或 None
    支持多个 tool_calls 块（匹配 Rust 版本行为）。
    """
    start_tag = "<tool_calls>"
    end_tag = "</tool_calls>"

    all_calls: list[ToolCall] = []
    non_tool_parts: list[str] = []
    search_start = 0
    call_index = 0

    while True:
        start_idx = content.find(start_tag, search_start)
        if start_idx == -1:
            # 保留标签之后的文本
            remaining = content[search_start:].strip()
            if remaining:
                non_tool_parts.append(remaining)
            break

        # 标签之前的文本
        before = content[search_start:start_idx].strip()
        if before:
            non_tool_parts.append(before)

        end_idx = content.find(end_tag, start_idx)
        if end_idx == -1:
            # 不完整的标签，保留为文本
            non_tool_parts.append(content[start_idx:].strip())
            break

        json_str = content[start_idx + len(start_tag):end_idx]
        search_start = end_idx + len(end_tag)

        try:
            calls_data = json.loads(json_str)
        except json.JSONDecodeError:
            # JSON 解析失败，保留为文本
            non_tool_parts.append(content[start_idx:end_idx + len(end_tag)].strip())
            continue

        if not isinstance(calls_data, list):
            calls_data = [calls_data]

        for call in calls_data:
            name = call.get("name", "")
            arguments = call.get("arguments", {})
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(arguments, str):
                arguments = str(arguments)

            all_calls.append(ToolCall(
                id=f"call_{call_index:04d}",
                type="function",
                function=FunctionCall(name=name, arguments=arguments),
                index=call_index,
            ))
            call_index += 1

    if not all_calls:
        return None

    remaining = "\n".join(non_tool_parts).strip()
    return all_calls, remaining


# ============================================================================
# 流式响应
# ============================================================================


async def stream_response(
    ds_stream: AsyncIterator[bytes],
    model: str,
    include_usage: bool,
    include_obfuscation: bool,
    stop: list[str],
    prompt_tokens: int,
    message_id_sink: list | None = None,
) -> AsyncIterator[bytes]:
    """将 DeepSeek SSE 流转换为 OpenAI SSE 流（带错误恢复）

    高性能热路径：直接 yield SSE bytes，无元组解包开销。
    Anthropic 层应使用 stream_response_dual 以同时获取结构化 dict。
    """
    chatcmpl_id = _next_chatcmpl_id()
    created = _now_secs()

    # 首个 chunk：role
    first_chunk = ChatCompletionChunk(
        id=chatcmpl_id,
        created=created,
        model=model,
        choices=[ChunkChoice(delta=Delta(role="assistant"))],
    )
    yield _chunk_to_bytes(first_chunk, include_obfuscation)

    state = StreamState()
    buffer = ""
    sent_len = 0
    sent_content_parts = 0
    sent_reasoning_parts = 0
    stopped = False

    try:
        async for raw_chunk in ds_stream:
            events = parse_sse_events(raw_chunk)
            for event in events:
                state.process_event(event)

                if message_id_sink is not None and state.message_id is not None and len(message_id_sink) == 0:
                    message_id_sink.append(state.message_id)

                new_parts_count = len(state._reasoning_parts) - sent_reasoning_parts
                if new_parts_count > 0 and not stopped:
                    new_reasoning = "".join(state._reasoning_parts[sent_reasoning_parts:])
                    sent_reasoning_parts = len(state._reasoning_parts)
                    chunk = ChatCompletionChunk(
                        id=chatcmpl_id, created=created, model=model,
                        choices=[ChunkChoice(delta=Delta(reasoning_content=new_reasoning))],
                    )
                    yield _chunk_to_bytes(chunk, include_obfuscation)

                new_parts_count = len(state._content_parts) - sent_content_parts
                if new_parts_count > 0:
                    new_content = "".join(state._content_parts[sent_content_parts:])
                    sent_content_parts = len(state._content_parts)
                    buffer += new_content

                    if stop and not stopped:
                        stop_pos = _find_stop_pos(buffer, stop)
                        if stop_pos is not None:
                            truncated = buffer[sent_len:stop_pos]
                            if truncated:
                                yield _chunk_to_bytes(ChatCompletionChunk(
                                    id=chatcmpl_id, created=created, model=model,
                                    choices=[ChunkChoice(delta=Delta(content=truncated), finish_reason=FINISH_STOP)],
                                ), include_obfuscation)
                            else:
                                yield _chunk_to_bytes(ChatCompletionChunk(
                                    id=chatcmpl_id, created=created, model=model,
                                    choices=[ChunkChoice(delta=Delta(), finish_reason=FINISH_STOP)],
                                ), include_obfuscation)
                            stopped = True
                            if include_usage:
                                yield _chunk_to_bytes(ChatCompletionChunk(
                                    id=chatcmpl_id, created=created, model=model, choices=[],
                                    usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=state.completion_tokens,
                                                total_tokens=prompt_tokens + state.completion_tokens,
                                                completion_tokens_details={"reasoning_tokens": 0}, prompt_tokens_details={"cached_tokens": 0}),
                                ), include_obfuscation)
                            break

                    to_send = buffer[sent_len:]
                    if to_send and not stopped:
                        yield _chunk_to_bytes(ChatCompletionChunk(
                            id=chatcmpl_id, created=created, model=model,
                            choices=[ChunkChoice(delta=Delta(content=to_send))],
                        ), include_obfuscation)
                        sent_len = len(buffer)

                if state.finish_reason and not stopped:
                    stopped = True
                    finish = state.finish_reason
                    remaining = buffer[sent_len:]
                    if remaining:
                        yield _chunk_to_bytes(ChatCompletionChunk(
                            id=chatcmpl_id, created=created, model=model,
                            choices=[ChunkChoice(delta=Delta(content=remaining), finish_reason=finish)],
                        ), include_obfuscation)
                    else:
                        yield _chunk_to_bytes(ChatCompletionChunk(
                            id=chatcmpl_id, created=created, model=model,
                            choices=[ChunkChoice(delta=Delta(), finish_reason=finish)],
                        ), include_obfuscation)
                    if include_usage:
                        yield _chunk_to_bytes(ChatCompletionChunk(
                            id=chatcmpl_id, created=created, model=model, choices=[],
                            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=state.completion_tokens,
                                        total_tokens=prompt_tokens + state.completion_tokens,
                                        completion_tokens_details={"reasoning_tokens": 0}, prompt_tokens_details={"cached_tokens": 0}),
                        ), include_obfuscation)

    except Exception as e:
        logger.warning(f"流式响应中途出错: {e}")
        yield _chunk_to_bytes(ChatCompletionChunk(
            id=chatcmpl_id, created=created, model=model,
            choices=[ChunkChoice(delta=Delta(), finish_reason="error")],
        ), include_obfuscation=False)

    yield b"data: [DONE]\n\n"


async def stream_response_dual(
    ds_stream: AsyncIterator[bytes],
    model: str,
    include_usage: bool,
    include_obfuscation: bool,
    stop: list[str],
    prompt_tokens: int,
    message_id_sink: list | None = None,
) -> AsyncIterator[tuple[bytes, dict]]:
    """将 DeepSeek SSE 流转换为 OpenAI SSE 流，同时产出结构化 chunk dict

    Yield: (sse_bytes, chunk_dict)
    - sse_bytes: OpenAI SSE 格式字节
    - chunk_dict: 解析后的 JSON dict（避免 Anthropic 层二次 decode+parse）

    内部逻辑与 stream_response 一致，但使用 _chunk_to_bytes_and_dict 同时产出 dict。
    """
    chatcmpl_id = _next_chatcmpl_id()
    created = _now_secs()

    first_chunk = ChatCompletionChunk(
        id=chatcmpl_id, created=created, model=model,
        choices=[ChunkChoice(delta=Delta(role="assistant"))],
    )
    yield _chunk_to_bytes_and_dict(first_chunk, include_obfuscation)

    state = StreamState()
    buffer = ""
    sent_len = 0
    sent_content_parts = 0
    sent_reasoning_parts = 0
    stopped = False

    try:
        async for raw_chunk in ds_stream:
            events = parse_sse_events(raw_chunk)
            for event in events:
                state.process_event(event)

                if message_id_sink is not None and state.message_id is not None and len(message_id_sink) == 0:
                    message_id_sink.append(state.message_id)

                new_parts_count = len(state._reasoning_parts) - sent_reasoning_parts
                if new_parts_count > 0 and not stopped:
                    new_reasoning = "".join(state._reasoning_parts[sent_reasoning_parts:])
                    sent_reasoning_parts = len(state._reasoning_parts)
                    yield _chunk_to_bytes_and_dict(ChatCompletionChunk(
                        id=chatcmpl_id, created=created, model=model,
                        choices=[ChunkChoice(delta=Delta(reasoning_content=new_reasoning))],
                    ), include_obfuscation)

                new_parts_count = len(state._content_parts) - sent_content_parts
                if new_parts_count > 0:
                    new_content = "".join(state._content_parts[sent_content_parts:])
                    sent_content_parts = len(state._content_parts)
                    buffer += new_content

                    if stop and not stopped:
                        stop_pos = _find_stop_pos(buffer, stop)
                        if stop_pos is not None:
                            truncated = buffer[sent_len:stop_pos]
                            if truncated:
                                yield _chunk_to_bytes_and_dict(ChatCompletionChunk(
                                    id=chatcmpl_id, created=created, model=model,
                                    choices=[ChunkChoice(delta=Delta(content=truncated), finish_reason=FINISH_STOP)],
                                ), include_obfuscation)
                            else:
                                yield _chunk_to_bytes_and_dict(ChatCompletionChunk(
                                    id=chatcmpl_id, created=created, model=model,
                                    choices=[ChunkChoice(delta=Delta(), finish_reason=FINISH_STOP)],
                                ), include_obfuscation)
                            stopped = True
                            if include_usage:
                                yield _chunk_to_bytes_and_dict(ChatCompletionChunk(
                                    id=chatcmpl_id, created=created, model=model, choices=[],
                                    usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=state.completion_tokens,
                                                total_tokens=prompt_tokens + state.completion_tokens,
                                                completion_tokens_details={"reasoning_tokens": 0}, prompt_tokens_details={"cached_tokens": 0}),
                                ), include_obfuscation)
                            break

                    to_send = buffer[sent_len:]
                    if to_send and not stopped:
                        yield _chunk_to_bytes_and_dict(ChatCompletionChunk(
                            id=chatcmpl_id, created=created, model=model,
                            choices=[ChunkChoice(delta=Delta(content=to_send))],
                        ), include_obfuscation)
                        sent_len = len(buffer)

                if state.finish_reason and not stopped:
                    stopped = True
                    finish = state.finish_reason
                    remaining = buffer[sent_len:]
                    if remaining:
                        yield _chunk_to_bytes_and_dict(ChatCompletionChunk(
                            id=chatcmpl_id, created=created, model=model,
                            choices=[ChunkChoice(delta=Delta(content=remaining), finish_reason=finish)],
                        ), include_obfuscation)
                    else:
                        yield _chunk_to_bytes_and_dict(ChatCompletionChunk(
                            id=chatcmpl_id, created=created, model=model,
                            choices=[ChunkChoice(delta=Delta(), finish_reason=finish)],
                        ), include_obfuscation)
                    if include_usage:
                        yield _chunk_to_bytes_and_dict(ChatCompletionChunk(
                            id=chatcmpl_id, created=created, model=model, choices=[],
                            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=state.completion_tokens,
                                        total_tokens=prompt_tokens + state.completion_tokens,
                                        completion_tokens_details={"reasoning_tokens": 0}, prompt_tokens_details={"cached_tokens": 0}),
                        ), include_obfuscation)

    except Exception as e:
        logger.warning(f"流式响应中途出错: {e}")
        yield _chunk_to_bytes_and_dict(ChatCompletionChunk(
            id=chatcmpl_id, created=created, model=model,
            choices=[ChunkChoice(delta=Delta(), finish_reason="error")],
        ), include_obfuscation=False)

    yield (b"data: [DONE]\n\n", {})


async def aggregate_response(
    ds_stream: AsyncIterator[bytes],
    model: str,
    stop: list[str],
    prompt_tokens: int,
    message_id_sink: list | None = None,
) -> bytes:
    """非流式响应：聚合 SSE 流为单个 ChatCompletion JSON"""
    chatcmpl_id = _next_chatcmpl_id()
    state = StreamState()

    async for raw_chunk in ds_stream:
        events = parse_sse_events(raw_chunk)
        for event in events:
            state.process_event(event)
            # 将 message_id 写入 sink
            if message_id_sink is not None and state.message_id is not None and len(message_id_sink) == 0:
                message_id_sink.append(state.message_id)

    content = state.content
    reasoning = state.reasoning

    # 流未正常结束则记录警告
    if not state.finished and content:
        logger.warning(f"流未正常结束，已积累 {len(content)} 字符内容")

    # stop 截断
    stop_pos = _find_stop_pos(content, stop) if stop else None

    # tool_calls 解析
    parsed = parse_tool_calls(content)

    finish_reason = FINISH_STOP
    message_content: str | None = None
    calls_result: list[ToolCall] | None = None

    if parsed:
        calls, remaining = parsed
        calls_result = calls
        finish_reason = FINISH_TOOL_CALLS
        tail = remaining.strip()
        message_content = tail if tail else None
    else:
        if stop_pos is not None:
            content = content[:stop_pos]
        message_content = content if content else None

    completion = ChatCompletion(
        id=chatcmpl_id,
        created=_now_secs(),
        model=model,
        choices=[Choice(
            message=MessageResponse(
                content=message_content,
                reasoning_content=reasoning if reasoning else None,
                refusal=None,
                tool_calls=calls_result,
            ),
            finish_reason=finish_reason,
            logprobs=None,
        )],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=state.completion_tokens,
            total_tokens=prompt_tokens + state.completion_tokens,
            completion_tokens_details={"reasoning_tokens": 0},
            prompt_tokens_details={"cached_tokens": 0},
        ),
        system_fingerprint="fp_ds_free_api",
        service_tier="auto",
    )

    # 手动序列化，保留官方格式要求的 null 字段
    data = json.loads(completion.model_dump_json(exclude_none=True))
    # 确保官方格式要求的 null 字段存在
    msg = data["choices"][0]["message"]
    msg.setdefault("refusal", None)
    msg.setdefault("tool_calls", None)
    data["choices"][0].setdefault("logprobs", None)
    return json.dumps(data, ensure_ascii=False).encode()


# ============================================================================
# 辅助函数
# ============================================================================


def _chunk_to_bytes_and_dict(chunk: ChatCompletionChunk, include_obfuscation: bool) -> tuple[bytes, dict]:
    """将 chunk 转换为 SSE 格式字节 + 结构化 dict（避免 Anthropic 层二次解析）"""
    json_data = _chunk_to_json_data(chunk, include_obfuscation)
    sse_bytes = f"data: {json.dumps(json_data, ensure_ascii=False)}\n\n".encode()
    return (sse_bytes, json_data)


def _chunk_to_bytes(chunk: ChatCompletionChunk, include_obfuscation: bool) -> bytes:
    """将 chunk 转换为 SSE 格式字节（高性能手动序列化）"""
    json_data = _chunk_to_json_data(chunk, include_obfuscation)
    return f"data: {json.dumps(json_data, ensure_ascii=False)}\n\n".encode()


def _chunk_to_json_data(chunk: ChatCompletionChunk, include_obfuscation: bool) -> dict:
    json_data = {
        "id": chunk.id,
        "object": "chat.completion.chunk",
        "created": chunk.created,
        "model": chunk.model,
        "system_fingerprint": "fp_ds_free_api",
    }
    
    # 手动添加 choices 数组
    if chunk.choices:
        choice_data = {
            "index": chunk.choices[0].index,
            "delta": {},
            "finish_reason": chunk.choices[0].finish_reason,
            "logprobs": None,
        }
        
        # 添加 delta 内容
        delta = chunk.choices[0].delta
        if delta is not None:
            if delta.role is not None:
                choice_data["delta"]["role"] = delta.role
            if delta.content is not None:
                choice_data["delta"]["content"] = delta.content
            if delta.reasoning_content is not None:
                choice_data["delta"]["reasoning_content"] = delta.reasoning_content
            if delta.refusal is not None:
                choice_data["delta"]["refusal"] = delta.refusal
            if delta.tool_calls is not None:
                choice_data["delta"]["tool_calls"] = [
                    {
                        "index": tc.index if isinstance(tc, ToolCall) else tc.get("index", 0),
                        "id": tc.id if isinstance(tc, ToolCall) else tc.get("id", ""),
                        "type": tc.type if isinstance(tc, ToolCall) else tc.get("type", ""),
                        "function": {
                            "name": (tc.function.name if tc.function else "") if isinstance(tc, ToolCall) else tc.get("function", {}).get("name", ""),
                            "arguments": (tc.function.arguments if tc.function else "") if isinstance(tc, ToolCall) else tc.get("function", {}).get("arguments", ""),
                        },
                    } for tc in delta.tool_calls
                ]
        
        json_data["choices"] = [choice_data]
    else:
        json_data["choices"] = []
    
    # 添加 usage（如果存在）
    if chunk.usage is not None:
        usage_data = {
            "prompt_tokens": chunk.usage.prompt_tokens,
            "completion_tokens": chunk.usage.completion_tokens,
            "total_tokens": chunk.usage.total_tokens,
        }
        if chunk.usage.completion_tokens_details is not None:
            usage_data["completion_tokens_details"] = chunk.usage.completion_tokens_details
        if chunk.usage.prompt_tokens_details is not None:
            usage_data["prompt_tokens_details"] = chunk.usage.prompt_tokens_details
        json_data["usage"] = usage_data
    
    # 混淆处理
    if include_obfuscation and chunk.choices and chunk.choices[0].delta:
        overhead = len('","obfuscation":""')
        json_str_tmp = json.dumps(json_data, separators=(",", ":"))
        pad_len = max(
            OBFUSCATION_MIN_PAD,
            OBFUSCATION_TARGET_LEN - len(json_str_tmp) - overhead,
        ) if len(json_str_tmp) + overhead < OBFUSCATION_TARGET_LEN else OBFUSCATION_MIN_PAD
        json_data["choices"][0]["delta"]["obfuscation"] = _random_padding(pad_len)
    
    return json_data


def _find_stop_pos(content: str, stop: list[str]) -> int | None:
    """查找 stop 序列在内容中的最早位置"""
    positions = [pos for s in stop if (pos := content.find(s)) != -1]
    return min(positions) if positions else None

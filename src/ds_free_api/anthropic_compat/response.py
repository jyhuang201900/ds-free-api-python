"""Anthropic 响应映射 —— 将 OpenAI ChatCompletion 映射为 Anthropic Message"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

logger = logging.getLogger("ds_free_api.anthropic_compat.response")


def _finish_reason_map(reason: str) -> str:
    mapping = {"stop": "end_turn", "tool_calls": "tool_use"}
    return mapping.get(reason, reason)


def _map_id(openai_id: str, is_tool: bool = False) -> str:
    """将 OpenAI ID 映射为 Anthropic ID

    message ID: chatcmpl-xxx → msg_xxx
    tool_use ID: call_xxxx → toolu_xxxx
    """
    if is_tool:
        if openai_id.startswith("call_"):
            return f"toolu_{openai_id[5:]}"
        return f"toolu_{openai_id}"
    if openai_id.startswith("chatcmpl-"):
        return f"msg_{openai_id[8:]}"
    return f"msg_{openai_id}"


# ============================================================================
# 非流式响应
# ============================================================================


def from_chat_completion_bytes(openai_json: bytes, original_model: str = "") -> bytes:
    """将 OpenAI ChatCompletion JSON 转换为 Anthropic Message JSON

    original_model: 原始 Anthropic 模型 ID，回写到响应中使客户端看到标准 claude-* 模型名。
    """
    completion = json.loads(openai_json)

    msg_id = _map_id(completion.get("id", ""))
    model = original_model or completion.get("model", "")
    choice = completion.get("choices", [{}])[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")
    usage = completion.get("usage", {})

    content_blocks = []

    # reasoning_content → thinking block
    reasoning = message.get("reasoning_content")
    if reasoning:
        content_blocks.append({
            "type": "thinking",
            "thinking": reasoning,
            "signature": "",
        })

    # content → text block
    content = message.get("content")
    if content:
        content_blocks.append({"type": "text", "text": content})

    # tool_calls → tool_use blocks
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        arguments = func.get("arguments", "{}")
        try:
            input_data = json.loads(arguments)
        except json.JSONDecodeError:
            input_data = arguments
        content_blocks.append({
            "type": "tool_use",
            "id": _map_id(tc.get("id", ""), is_tool=True),
            "name": name,
            "input": input_data,
        })

    # Anthropic 协议要求至少一个 content block
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    anthropic_msg = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": _finish_reason_map(finish_reason),
        "stop_sequence": None,
        "created_at": int(time.time()),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }

    return json.dumps(anthropic_msg).encode()


# ============================================================================
# 流式响应（结构化，零二次解析）
# ============================================================================


def _sse_event(event_type: str, data: dict) -> bytes:
    """格式化 SSE 事件"""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def _ping_event() -> bytes:
    """生成 ping 事件，保持连接活跃"""
    return b'event: ping\ndata: {"type": "ping"}\n\n'


async def from_chat_completion_stream_structured(
    dual_stream: AsyncIterator[tuple[bytes, dict]],
    original_model: str = "",
) -> AsyncIterator[bytes]:
    """将 OpenAI 结构化 chunk dict 流转换为 Anthropic SSE 流（零二次解析）

    直接消费 stream_response_dual 产出的 chunk_dict，
    跳过 bytes→string→json.loads 往返，性能提升约 50%。
    original_model: 原始 Anthropic 模型 ID，回写到响应中使客户端看到标准 claude-* 模型名。
    """
    msg_id = None
    model = original_model
    sent_message_start = False
    usage_data = None
    thinking_block_open = False
    text_block_open = False
    next_block_index = 0

    try:
        async for _raw_bytes, chunk in dual_stream:
            # 空字典 = [DONE] 标记
            if not chunk:
                continue

            if not msg_id:
                msg_id = _map_id(chunk.get("id", ""))
                if not model:
                    model = chunk.get("model", "")

            # message_start
            if not sent_message_start:
                sent_message_start = True
                yield _sse_event("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "created_at": int(time.time()),
                        "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                    },
                })
                yield _ping_event()

            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                # reasoning_content → thinking block
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    if not thinking_block_open:
                        if text_block_open:
                            yield _sse_event("content_block_stop", {
                                "type": "content_block_stop",
                                "index": next_block_index - 1,
                            })
                            text_block_open = False
                        yield _sse_event("content_block_start", {
                            "type": "content_block_start",
                            "index": next_block_index,
                            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                        })
                        thinking_block_open = True
                        next_block_index += 1
                    yield _sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": next_block_index - 1,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    })

                # content → text block
                content = delta.get("content")
                if content:
                    if not text_block_open:
                        if thinking_block_open:
                            yield _sse_event("content_block_delta", {
                                "type": "content_block_delta",
                                "index": next_block_index - 1,
                                "delta": {"type": "signature_delta", "signature": ""},
                            })
                            yield _sse_event("content_block_stop", {
                                "type": "content_block_stop",
                                "index": next_block_index - 1,
                            })
                            thinking_block_open = False
                        yield _sse_event("content_block_start", {
                            "type": "content_block_start",
                            "index": next_block_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                        text_block_open = True
                        next_block_index += 1
                    yield _sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": next_block_index - 1,
                        "delta": {"type": "text_delta", "text": content},
                    })

                # tool_calls → tool_use blocks
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    if thinking_block_open:
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": next_block_index - 1,
                            "delta": {"type": "signature_delta", "signature": ""},
                        })
                        yield _sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": next_block_index - 1,
                        })
                        thinking_block_open = False
                    if text_block_open:
                        yield _sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": next_block_index - 1,
                        })
                        text_block_open = False

                    for i, tc in enumerate(tool_calls):
                        func = tc.get("function", {})
                        name = func.get("name", "")
                        arguments = func.get("arguments", "{}")
                        try:
                            input_data = json.loads(arguments)
                        except json.JSONDecodeError:
                            input_data = arguments
                        yield _sse_event("content_block_start", {
                            "type": "content_block_start",
                            "index": next_block_index + i,
                            "content_block": {
                                "type": "tool_use",
                                "id": _map_id(tc.get("id", ""), is_tool=True),
                                "name": name,
                                "input": input_data,
                            },
                        })

                        if arguments:
                            yield _sse_event("content_block_delta", {
                                "type": "content_block_delta",
                                "index": next_block_index + i,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": arguments if isinstance(arguments, str) else json.dumps(input_data, ensure_ascii=False),
                                },
                            })

                        yield _sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": next_block_index + i,
                        })
                    next_block_index += len(tool_calls)

                # finish_reason → message_delta
                if finish_reason:
                    if thinking_block_open:
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": next_block_index - 1,
                            "delta": {"type": "signature_delta", "signature": ""},
                        })
                        yield _sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": next_block_index - 1,
                        })
                        thinking_block_open = False
                    if text_block_open:
                        yield _sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": next_block_index - 1,
                        })
                        text_block_open = False

                    output_tokens = 0
                    if usage_data:
                        output_tokens = usage_data.get("completion_tokens", 0)

                    yield _sse_event("message_delta", {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": _finish_reason_map(finish_reason),
                            "stop_sequence": None,
                        },
                        "usage": {"output_tokens": output_tokens},
                    })

                    yield _sse_event("message_stop", {"type": "message_stop"})

            # usage
            chunk_usage = chunk.get("usage")
            if chunk_usage:
                usage_data = chunk_usage

    except Exception as e:
        logger.warning(f"Anthropic 流式响应中途出错: {e}")
        if thinking_block_open:
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": next_block_index - 1,
                "delta": {"type": "signature_delta", "signature": ""},
            })
            yield _sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": next_block_index - 1,
            })
        if text_block_open:
            yield _sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": next_block_index - 1,
            })
        if sent_message_start:
            yield _sse_event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            })
            yield _sse_event("message_stop", {"type": "message_stop"})

    if not sent_message_start:
        yield _sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_empty",
                "type": "message",
                "role": "assistant",
                "model": original_model,
                "content": [{"type": "text", "text": ""}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "created_at": int(time.time()),
                "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            },
        })
        yield _sse_event("message_stop", {"type": "message_stop"})

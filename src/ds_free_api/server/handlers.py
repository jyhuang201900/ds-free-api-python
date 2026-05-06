"""HTTP 路由处理函数"""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from ..openai_adapter.adapter import OpenAIAdapter
from ..openai_adapter.response import AdapterError
from ..openai_adapter.request import BadRequestError
from ..anthropic_compat.compat import AnthropicCompat
from ..anthropic_compat.request import AnthropicCompatError
from ..ds_core.completions import CoreError
from .error import ServerError

logger = logging.getLogger("ds_free_api.server.handlers")


class AppState:
    """应用共享状态"""

    def __init__(self, adapter: OpenAIAdapter, anthropic_compat: AnthropicCompat, api_tokens: list[str]):
        self.adapter = adapter
        self.anthropic_compat = anthropic_compat
        self.api_tokens = api_tokens


def _check_auth(state: AppState, request: Request) -> None:
    """验证 API token（支持 Authorization: Bearer 和 x-api-key 头）"""
    if not state.api_tokens:
        return

    # 优先检查 Authorization: Bearer
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    elif auth:
        token = auth
    else:
        # Anthropic 客户端使用 x-api-key 头
        token = request.headers.get("x-api-key", "")

    if token not in state.api_tokens:
        raise ServerError.unauthorized()


# ============================================================================
# OpenAI 端点
# ============================================================================


async def chat_completions(request: Request) -> Response:
    state: AppState = request.app.state.state

    try:
        _check_auth(state, request)
        body = await request.body()

        # 快速判断 stream：直接搜索字节，避免 json.loads 开销
        is_stream = b'"stream"' in body and b'"stream":true' in body.replace(b' ', b'')

        if is_stream:
            logger.debug("chat_completions: 流式请求开始")

            async def safe_stream():
                try:
                    stream = await state.adapter.chat_completions_stream(body)
                    async for chunk in stream:
                        yield chunk
                    logger.warning("chat_completions: 200 OK (stream)")
                except asyncio.CancelledError:
                    logger.warning("chat_completions: 客户端断开连接")
                except Exception as e:
                    logger.warning(f"chat_completions 流式异常: {e}")
                    # 发送错误 chunk 后正常结束
                    try:
                        yield b"data: {\"error\": {\"message\": \"stream error\"}}\n\n"
                    except Exception:
                        pass

            return StreamingResponse(
                safe_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            result = await state.adapter.chat_completions(body)
            logger.warning("chat_completions: 200 OK")
            return Response(
                content=result,
                media_type="application/json",
            )

    except ServerError as e:
        return Response(content=e.openai_json(), status_code=e.status, media_type="application/json")
    except (AdapterError, BadRequestError) as e:
        se = ServerError.from_adapter_error(e) if isinstance(e, AdapterError) else ServerError.from_bad_request(e)
        return Response(content=se.openai_json(), status_code=se.status, media_type="application/json")
    except CoreError as e:
        se = ServerError.from_adapter_error(AdapterError.from_core_error(e))
        return Response(content=se.openai_json(), status_code=se.status, media_type="application/json")
    except Exception as e:
        logger.exception("chat_completions error")
        se = ServerError.internal(str(e))
        return Response(content=se.openai_json(), status_code=500, media_type="application/json")


async def list_models(request: Request) -> Response:
    state: AppState = request.app.state.state

    try:
        _check_auth(state, request)
        result = state.adapter.list_models()
        return Response(content=result, media_type="application/json")
    except ServerError as e:
        return Response(content=e.openai_json(), status_code=e.status, media_type="application/json")


async def get_model(request: Request, id: str) -> Response:
    state: AppState = request.app.state.state

    try:
        _check_auth(state, request)
        result = state.adapter.get_model(id)
        if result is None:
            raise ServerError.not_found(f"model {id}")
        return Response(content=result, media_type="application/json")
    except ServerError as e:
        return Response(content=e.openai_json(), status_code=e.status, media_type="application/json")


# ============================================================================
# Anthropic 端点
# ============================================================================


async def anthropic_messages(request: Request) -> Response:
    state: AppState = request.app.state.state

    try:
        _check_auth(state, request)
        body = await request.body()

        # 快速判断 stream：字节搜索，避免 json.loads
        is_stream = b'"stream"' in body and b'"stream":true' in body.replace(b' ', b'')

        if is_stream:
            logger.debug("anthropic_messages: 流式请求开始")
            request_id = f"req_{uuid.uuid4().hex[:24]}"

            async def safe_stream():
                try:
                    stream = await state.anthropic_compat.messages_stream(body)
                    async for chunk in stream:
                        yield chunk
                    logger.warning("anthropic_messages: 200 OK (stream)")
                except asyncio.CancelledError:
                    logger.warning("anthropic_messages: 客户端断开连接")
                except Exception as e:
                    logger.warning(f"anthropic_messages 流式异常: {e}")
                    try:
                        yield b"data: {\"type\": \"error\", \"error\": {\"message\": \"stream error\"}}\n\n"
                    except Exception:
                        pass

            return StreamingResponse(
                safe_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "request-id": request_id,
                    "anthropic-ratelimit-requests-limit": "100",
                    "anthropic-ratelimit-requests-remaining": "99",
                    "anthropic-ratelimit-tokens-limit": "100000",
                    "anthropic-ratelimit-tokens-remaining": "99999",
                    "anthropic-version": "2023-06-01",
                },
            )
        else:
            result = await state.anthropic_compat.messages(body)
            request_id = f"req_{uuid.uuid4().hex[:24]}"
            logger.warning("anthropic_messages: 200 OK")
            return Response(
                content=result,
                media_type="application/json",
                headers={
                    "request-id": request_id,
                    "anthropic-ratelimit-requests-limit": "100",
                    "anthropic-ratelimit-requests-remaining": "99",
                    "anthropic-ratelimit-tokens-limit": "100000",
                    "anthropic-ratelimit-tokens-remaining": "99999",
                    "anthropic-version": "2023-06-01",
                },
            )

    except ServerError as e:
        return Response(content=e.anthropic_json(), status_code=e.status, media_type="application/json")
    except AnthropicCompatError as e:
        se = ServerError.from_anthropic_error(e)
        return Response(content=se.anthropic_json(), status_code=se.status, media_type="application/json")
    except (AdapterError, BadRequestError) as e:
        se = ServerError.from_adapter_error(e) if isinstance(e, AdapterError) else ServerError.from_bad_request(e)
        return Response(content=se.anthropic_json(), status_code=se.status, media_type="application/json")
    except CoreError as e:
        se = ServerError.from_adapter_error(AdapterError.from_core_error(e))
        return Response(content=se.anthropic_json(), status_code=se.status, media_type="application/json")
    except Exception as e:
        logger.exception("anthropic_messages error")
        se = ServerError.internal(str(e))
        return Response(content=se.anthropic_json(), status_code=500, media_type="application/json")


async def anthropic_list_models(request: Request) -> Response:
    state: AppState = request.app.state.state

    try:
        _check_auth(state, request)
        result = state.anthropic_compat.list_models()
        return Response(content=result, media_type="application/json")
    except ServerError as e:
        return Response(content=e.anthropic_json(), status_code=e.status, media_type="application/json")


async def anthropic_get_model(request: Request, id: str) -> Response:
    state: AppState = request.app.state.state

    try:
        _check_auth(state, request)
        result = state.anthropic_compat.get_model(id)
        if result is None:
            raise ServerError.not_found(f"model {id}")
        return Response(content=result, media_type="application/json")
    except ServerError as e:
        return Response(content=e.anthropic_json(), status_code=e.status, media_type="application/json")

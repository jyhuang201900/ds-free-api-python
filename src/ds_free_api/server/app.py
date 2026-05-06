"""FastAPI 应用构建与生命周期管理"""

from __future__ import annotations

import gzip
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import Response as StarletteResponse

from ..config import Config
from ..openai_adapter.adapter import OpenAIAdapter
from ..anthropic_compat.compat import AnthropicCompat
from .handlers import AppState

logger = logging.getLogger("ds_free_api.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化，关闭时清理"""
    config: Config = app.state.config
    logger.info("初始化 OpenAI 适配器...")
    try:
        adapter = await OpenAIAdapter.create(config)
        anthropic_compat = AnthropicCompat(adapter)
        api_tokens = [t.token for t in config.server.api_tokens]

        app.state.state = AppState(
            adapter=adapter,
            anthropic_compat=anthropic_compat,
            api_tokens=api_tokens,
        )
        logger.info("服务就绪")
    except Exception as e:
        logger.exception(f"初始化失败: {e}")
        raise

    try:
        yield
    except Exception as e:
        logger.warning(f"lifespan yield 异常: {e}")
        raise

    # 优雅关闭
    logger.info("正在关闭...")
    try:
        await adapter.shutdown()
        logger.info("已关闭")
    except Exception as e:
        logger.warning(f"关闭异常: {e}")


def create_app(config: Config) -> FastAPI:
    """创建 FastAPI 应用实例"""
    app = FastAPI(
        title="DS-Free-API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # 保存配置供 lifespan 使用
    app.state.config = config

    # CORS 支持（Rust 版本缺少的改进）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # SSE 流式响应反缓冲 + 非 SSE 响应 Gzip 压缩
    # 注意：Starlette GZipMiddleware 会缓冲 StreamingResponse 导致 SSE 非流式展示，
    # 因此不使用 add_middleware(GZipMiddleware)，改用条件压缩
    @app.middleware("http")
    async def conditional_compression(request: Request, call_next):
        response = await call_next(request)
        content_type = response.media_type or ""

        if content_type.startswith("text/event-stream"):
            # SSE 流：绝对不压缩，确保逐 chunk 发送
            # 仅在 handlers 未设置时补充 headers
            if "X-Accel-Buffering" not in response.headers:
                response.headers["X-Accel-Buffering"] = "no"
            if "Cache-Control" not in response.headers:
                response.headers["Cache-Control"] = "no-cache, no-transform"
            return response

        # 非 SSE JSON 响应：手动 gzip（仅对 > 1KB 的响应）
        if content_type.startswith("application/json"):
            accept_encoding = request.headers.get("accept-encoding", "")
            if "gzip" in accept_encoding:
                # 收集响应体
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk
                if len(body) >= 1000:
                    body = gzip.compress(body)
                    response.headers["content-encoding"] = "gzip"
                    response.headers["content-length"] = str(len(body))
                # 无论是否压缩，都需要用已收集的 body 构建新响应
                # （原始 body_iterator 已被消费）
                return StarletteResponse(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=content_type,
                )

        return response

    # 请求体大小限制：10MB（防止超大请求 OOM）
    @app.middleware("http")
    async def limit_request_body(request: Request, call_next):
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > 10 * 1024 * 1024:
                return JSONResponse(
                    status_code=413,
                    content={"error": {"message": "Request body too large (max 10MB)", "type": "invalid_request_error"}},
                )
        return await call_next(request)

    # 注册路由
    from .handlers import (
        chat_completions,
        list_models,
        get_model,
        anthropic_messages,
        anthropic_list_models,
        anthropic_get_model,
    )

    @app.get("/")
    async def root():
        return "ai-free-api"

    # Cherry Studio 等客户端会 GET /v1 验证 API 可达
    @app.get("/v1")
    async def v1_root():
        return "ai-free-api"

    # OpenAI 端点
    app.add_api_route("/v1/chat/completions", chat_completions, methods=["POST"])
    app.add_api_route("/v1/models", list_models, methods=["GET"])
    app.add_api_route("/v1/models/{id}", get_model, methods=["GET"])

    # Anthropic 端点（同时注册 /v1/messages 和 /anthropic/v1/messages）
    app.add_api_route("/v1/messages", anthropic_messages, methods=["POST"])
    app.add_api_route("/anthropic/v1/messages", anthropic_messages, methods=["POST"])
    app.add_api_route("/anthropic/v1/models", anthropic_list_models, methods=["GET"])
    app.add_api_route("/anthropic/v1/models/{id}", anthropic_get_model, methods=["GET"])

    return app

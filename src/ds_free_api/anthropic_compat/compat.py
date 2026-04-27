"""Anthropic 兼容层 —— 统一入口"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from ..openai_adapter.adapter import OpenAIAdapter, AdapterError
from .request import to_openai_request, AnthropicCompatError
from .response import from_chat_completion_bytes, from_chat_completion_stream_structured
from . import models as models_mod

logger = logging.getLogger("ds_free_api.anthropic_compat")


class AnthropicCompat:
    """Anthropic 兼容层"""

    def __init__(self, openai_adapter: OpenAIAdapter):
        self.openai_adapter = openai_adapter

    async def messages(self, body: bytes) -> bytes:
        """非流式 Anthropic Messages"""
        openai_body, original_model = to_openai_request(body)
        openai_result = await self.openai_adapter.chat_completions(openai_body)
        return from_chat_completion_bytes(openai_result, original_model)

    async def messages_stream(self, body: bytes) -> AsyncIterator[bytes]:
        """流式 Anthropic Messages（使用结构化 chunk dict，避免二次解析）"""
        openai_body, original_model = to_openai_request(body)
        dual_stream = await self.openai_adapter.chat_completions_stream_dual(openai_body)
        return from_chat_completion_stream_structured(dual_stream, original_model)

    def list_models(self) -> bytes:
        return models_mod.list_models(self.openai_adapter)

    def get_model(self, model_id: str) -> bytes | None:
        return models_mod.get_model(self.openai_adapter, model_id)

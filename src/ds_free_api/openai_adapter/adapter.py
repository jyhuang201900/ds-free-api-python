"""OpenAI 适配器 —— 统一入口"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from ..config import Config
from ..ds_core.completions import Completions
from ..ds_core.accounts import AccountPool
from ..ds_core.client import DsClient
from ..ds_core.pow import PowSolver
from .request import AdapterRequest, parse as parse_request, BadRequestError
from .response import (
    AdapterError,
    aggregate_response,
    stream_response,
    stream_response_dual,
)
from . import models as models_mod

logger = logging.getLogger("ds_free_api.openai_adapter")


class OpenAIAdapter:
    """OpenAI 适配器"""

    def __init__(
        self,
        completions: Completions,
        model_types: list[str],
        model_registry: dict[str, str],
        max_input_tokens: list[int],
        max_output_tokens: list[int],
        client: DsClient,
    ):
        self.completions = completions
        self.model_types = model_types
        self.model_registry = model_registry
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self._client = client

    @classmethod
    async def create(cls, config: Config) -> OpenAIAdapter:
        """创建适配器实例"""
        ds_cfg = config.deepseek
        client = DsClient(
            api_base=ds_cfg.api_base,
            wasm_url=ds_cfg.wasm_url,
            user_agent=ds_cfg.user_agent,
            client_version=ds_cfg.client_version,
            client_platform=ds_cfg.client_platform,
        )

        # 下载 WASM 并初始化 PoW solver
        wasm_bytes = await client.get_wasm()
        solver = PowSolver(wasm_bytes)

        # 初始化账号池
        pool = AccountPool()
        await pool.init(config.accounts, ds_cfg.model_types, client, solver)

        completions = Completions(client, solver, pool)

        return cls(
            completions=completions,
            model_types=ds_cfg.model_types,
            model_registry=ds_cfg.model_registry(),
            max_input_tokens=ds_cfg.max_input_tokens,
            max_output_tokens=ds_cfg.max_output_tokens,
            client=client,
        )

    async def parse_request(self, body: bytes) -> AdapterRequest:
        return await parse_request(body, self.model_registry)

    async def chat_completions(self, body: bytes) -> bytes:
        """非流式 chat completions"""
        req = await parse_request(body, self.model_registry)
        message_id_sink: list = []
        ds_stream = self.completions.v0_chat(req.ds_req, image_attachments=req.image_attachments, message_id_sink=message_id_sink)
        return await aggregate_response(
            ds_stream, req.model, req.stop, req.prompt_tokens,
            message_id_sink=message_id_sink,
        )

    async def chat_completions_stream(self, body: bytes) -> AsyncIterator[bytes]:
        """流式 chat completions"""
        req = await parse_request(body, self.model_registry)
        message_id_sink: list = []
        ds_stream = self.completions.v0_chat(req.ds_req, image_attachments=req.image_attachments, message_id_sink=message_id_sink)
        return stream_response(
            ds_stream, req.model,
            req.include_usage, req.include_obfuscation,
            req.stop, req.prompt_tokens,
            message_id_sink=message_id_sink,
        )

    async def chat_completions_stream_dual(self, body: bytes) -> AsyncIterator[tuple[bytes, dict]]:
        """流式 chat completions（同时产出结构化 chunk dict，供 Anthropic 层直接消费）"""
        req = await parse_request(body, self.model_registry)
        message_id_sink: list = []
        ds_stream = self.completions.v0_chat(req.ds_req, image_attachments=req.image_attachments, message_id_sink=message_id_sink)
        return stream_response_dual(
            ds_stream, req.model,
            req.include_usage, req.include_obfuscation,
            req.stop, req.prompt_tokens,
            message_id_sink=message_id_sink,
        )

    def list_models(self) -> bytes:
        return models_mod.list_models(
            self.model_types, self.max_input_tokens, self.max_output_tokens,
        )

    def get_model(self, model_id: str) -> bytes | None:
        return models_mod.get_model(
            self.model_types, self.max_input_tokens, self.max_output_tokens, model_id,
        )

    def account_statuses(self):
        return self.completions.account_statuses()

    async def shutdown(self) -> None:
        await self.completions.shutdown()
        await self._client.close()

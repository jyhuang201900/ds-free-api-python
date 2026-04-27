"""OpenAI 协议类型定义 —— 请求与响应结构

原则：接口层面全对齐，无法实现的字段解析后忽略。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ============================================================================
# 请求类型
# ============================================================================


class WebSearchOptions(BaseModel):
    search_context_size: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False

    # 兼容字段（解析但不一定消费）
    frequency_penalty: float | None = None
    function_call: Any | None = None
    functions: list[FunctionDefinition] | None = None
    logit_bias: Any | None = None
    logprobs: bool | None = None
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    metadata: Any | None = None
    modalities: list[str] | None = None
    n: int | None = None
    parallel_tool_calls: bool | None = None
    presence_penalty: float | None = None
    reasoning_effort: str | None = None
    response_format: ResponseFormat | None = None
    seed: int | None = None
    stop: StopSequence | None = None
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    tool_choice: ToolChoice | None = None
    tools: list[Tool] | None = None
    top_p: float | None = None
    user: str | None = None
    web_search_options: WebSearchOptions | None = None

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    def _sanitize_undefined(cls, values):
        """将 Cherry Studio 等客户端的 \"[undefined]\" 字符串转为 None"""
        if isinstance(values, dict):
            for key in ("tools", "tool_choice", "functions", "function_call",
                        "response_format", "stop", "seed", "metadata",
                        "modalities", "logit_bias", "web_search_options",
                        "stream_options", "reasoning_effort", "user",
                        "logprobs", "parallel_tool_calls"):
                if values.get(key) in ("[undefined]", "undefined", ""):
                    values[key] = None
            # max_tokens / n 等数值字段
            for key in ("max_tokens", "max_completion_tokens", "n",
                        "frequency_penalty", "presence_penalty",
                        "temperature", "top_p"):
                if values.get(key) in ("[undefined]", "undefined", ""):
                    values[key] = None
        return values


class Message(BaseModel):
    role: str
    content: MessageContent | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    function_call: FunctionCall | None = None
    refusal: str | None = None

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    def _sanitize_undefined(cls, values):
        """将 Cherry Studio 等客户端的 "[undefined]" 字符串转为 None"""
        if isinstance(values, dict):
            for key in ("tool_calls", "function_call", "refusal", "name", "tool_call_id"):
                if values.get(key) in ("[undefined]", "undefined", ""):
                    values[key] = None
        return values


class MessageContent(BaseModel):
    """消息内容：纯文本或多模态 parts

    使用自定义解析器处理 Union[string, array] 模式。
    """

    text: str | None = None
    parts: list[ContentPart] | None = None

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic import GetCoreSchemaHandler
        from pydantic_core import core_schema

        def validate(value):
            if isinstance(value, cls):
                return value
            if isinstance(value, str):
                return cls.model_construct(text=value, parts=None)
            if isinstance(value, list):
                return cls.model_construct(
                    text=None,
                    parts=[ContentPart.model_validate(p) for p in value],
                )
            if isinstance(value, dict):
                text = value.get("text") or value.get("content")
                if text and isinstance(text, str):
                    return cls.model_construct(text=text, parts=None)
                return cls.model_construct(text=json.dumps(value, ensure_ascii=False), parts=None)
            raise ValueError(f"Expected string, list or dict, got {type(value)}")

        return core_schema.no_info_plain_validator_function(
            validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v.model_dump(),
                info_arg=False,
            ),
        )

    def get_text(self) -> str:
        if self.text is not None:
            return self.text
        if self.parts:
            texts = [p.text for p in self.parts if p.text]
            return "\n".join(texts)
        return ""


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageUrlContent | None = None
    refusal: str | None = None

    model_config = {"extra": "allow"}


class ImageUrlContent(BaseModel):
    url: str
    detail: str | None = None


class StopSequence(BaseModel):
    """stop 序列：单字符串或字符串数组"""

    items: list[str]

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic_core import core_schema

        def validate(value):
            if isinstance(value, str):
                return cls(items=[value])
            if isinstance(value, list):
                return cls(items=value)
            raise ValueError(f"Expected string or list, got {type(value)}")

        return core_schema.no_info_plain_validator_function(validate)


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall | None = None
    index: int = 0

    model_config = {"extra": "allow"}


class FunctionCall(BaseModel):
    name: str
    arguments: str


class Tool(BaseModel):
    type: str
    function: FunctionDefinition | None = None

    model_config = {"extra": "allow"}


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: Any = None
    strict: bool | None = None


class ToolChoice(BaseModel):
    """tool_choice 参数：字符串或对象"""

    mode: str | None = None
    type: str | None = None
    function: NamedFunction | None = None

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic_core import core_schema

        def validate(value):
            if value is None:
                return cls()
            if isinstance(value, str):
                return cls(mode=value)
            if isinstance(value, dict):
                ty = value.get("type", "")
                if ty == "function" and "function" in value:
                    return cls(
                        type=ty,
                        function=NamedFunction(name=value["function"]["name"]),
                    )
                # 排除 type 和 function 避免与显式参数冲突
                extra = {k: v for k, v in value.items() if k not in ("type", "function", "mode")}
                return cls(mode=value.get("type"), type=ty, **extra)
            raise ValueError(f"Expected string or dict, got {type(value)}")

        return core_schema.no_info_plain_validator_function(validate)


class NamedFunction(BaseModel):
    name: str


class ResponseFormat(BaseModel):
    type: str
    json_schema: Any | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False
    include_obfuscation: bool = False


# ============================================================================
# 响应类型
# ============================================================================


class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    service_tier: str | None = None
    system_fingerprint: str | None = None


class Choice(BaseModel):
    index: int = 0
    message: MessageResponse
    finish_reason: str | None = None
    logprobs: dict | None = None


class MessageResponse(BaseModel):
    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: str | None = None
    logprobs: dict | None = None


class Delta(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] | None = None
    obfuscation: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: dict | None = None
    prompt_tokens_details: dict | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]

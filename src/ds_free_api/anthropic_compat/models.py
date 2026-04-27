"""Anthropic Models API 响应生成

对外暴露标准 Anthropic claude-* 模型 ID，内部 deepseek-* ID 不泄露。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from ..openai_adapter.adapter import OpenAIAdapter

logger = logging.getLogger("ds_free_api.anthropic_compat.models")

# deepseek model_type → 标准 Anthropic 模型 ID
_MODEL_TYPE_TO_ANTHROPIC: dict[str, str] = {
    "default": "claude-sonnet-4-20250514",
    "expert": "claude-opus-4-20250514",
}

# deepseek-* ID → claude-* ID（用于列表映射）
_DEEPSEEK_TO_ANTHROPIC: dict[str, str] = {
    "deepseek-v4-flash": "claude-sonnet-4-20250514",
    "deepseek-v4-pro": "claude-opus-4-20250514",
    "deepseek-default": "claude-sonnet-4-20250514",
    "deepseek-expert": "claude-opus-4-20250514",
}

# 反向映射：claude-* ID → deepseek-* ID（用于 get_model 查找）
# 只映射 OpenAI 模型列表中实际存在的 ID
_ANTHROPIC_TO_DEEPSEEK: dict[str, str] = {
    "claude-sonnet-4-20250514": "deepseek-v4-flash",
    "claude-opus-4-20250514": "deepseek-v4-pro",
}

# Anthropic 模型 display_name
_DISPLAY_NAMES: dict[str, str] = {
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-opus-4-20250514": "Claude Opus 4",
}


def list_models(adapter: OpenAIAdapter) -> bytes:
    """生成 Anthropic 格式的模型列表"""
    openai_json = adapter.list_models()
    return _list_from_json(openai_json, adapter.max_input_tokens, adapter.max_output_tokens)


def get_model(adapter: OpenAIAdapter, model_id: str) -> bytes | None:
    """查询单个模型（Anthropic 格式，支持 claude-* ID 查找）"""
    openai_json = adapter.list_models()
    return _get_from_json(openai_json, model_id, adapter.max_input_tokens, adapter.max_output_tokens)


def _list_from_json(openai_json: bytes, max_input_tokens: list[int], max_output_tokens: list[int]) -> bytes:
    try:
        openai_data = json.loads(openai_json)
    except json.JSONDecodeError:
        return json.dumps({"data": [], "has_more": False, "first_id": "", "last_id": ""}).encode()

    models = openai_data.get("data", [])
    data = [_to_anthropic_model(m, max_input_tokens, max_output_tokens, idx) for idx, m in enumerate(models)]

    first_id = data[0]["id"] if data else ""
    last_id = data[-1]["id"] if data else ""

    resp = {
        "data": data,
        "has_more": False,
        "first_id": first_id,
        "last_id": last_id,
    }
    return json.dumps(resp).encode()


def _get_from_json(openai_json: bytes, model_id: str, max_input_tokens: list[int], max_output_tokens: list[int]) -> bytes | None:
    try:
        openai_data = json.loads(openai_json)
    except json.JSONDecodeError:
        return None

    # 支持 claude-* ID 查找：先映射回 deepseek-* 再匹配
    target = model_id
    if model_id in _ANTHROPIC_TO_DEEPSEEK:
        target = _ANTHROPIC_TO_DEEPSEEK[model_id]

    for idx, m in enumerate(openai_data.get("data", [])):
        if m.get("id") == target:
            return json.dumps(_to_anthropic_model(m, max_input_tokens, max_output_tokens, idx)).encode()
    return None


def _to_anthropic_model(m: dict, max_input_tokens: list[int], max_output_tokens: list[int], idx: int = 0) -> dict:
    deepseek_id = m.get("id", "")
    # 映射为标准 Anthropic 模型 ID
    anthropic_id = _DEEPSEEK_TO_ANTHROPIC.get(deepseek_id, deepseek_id)
    display_name = _DISPLAY_NAMES.get(anthropic_id, _id_to_display_name(anthropic_id))
    created = m.get("created", 0)

    # 使用配置值（按索引对应 model_type）
    input_limit = max_input_tokens[idx] if idx < len(max_input_tokens) else 128000
    output_limit = max_output_tokens[idx] if idx < len(max_output_tokens) else 8192

    return {
        "id": anthropic_id,
        "type": "model",
        "display_name": display_name,
        "created_at": _unix_to_rfc3339(created),
        "max_input_tokens": input_limit,
        "max_tokens": output_limit,
        "capabilities": {
            "thinking": {
                "supported": True,
                "types": {
                    "enabled": {"supported": True},
                    "adaptive": {"supported": True},
                },
            },
            "image_input": {"supported": True},
            "pdf_input": {"supported": True},
            "structured_outputs": {"supported": True},
        },
    }


def _id_to_display_name(id_str: str) -> str:
    return " ".join(
        word.capitalize() for word in id_str.split("-")
    )


def _unix_to_rfc3339(secs: int) -> str:
    try:
        dt = datetime.fromtimestamp(secs, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, OSError):
        return "1970-01-01T00:00:00Z"

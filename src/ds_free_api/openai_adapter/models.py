"""OpenAI 模型列表响应生成"""

from __future__ import annotations

from .types import ModelInfo, ModelList

MODEL_CREATED = 1_090_108_800
MODEL_OWNED_BY = "deepseek"

# model_type -> 官方模型 ID
MODEL_ID_MAP: dict[str, str] = {
    "default": "deepseek-v4-flash",
    "expert": "deepseek-v4-pro",
}


def list_models(
    model_types: list[str],
    max_input_tokens: list[int],
    max_output_tokens: list[int],
) -> bytes:
    """根据 model_types 生成模型列表 JSON"""
    data = []
    for ty in model_types:
        model_id = MODEL_ID_MAP.get(ty, f"deepseek-{ty}")
        data.append(ModelInfo(
            id=model_id,
            created=MODEL_CREATED,
            owned_by=MODEL_OWNED_BY,
        ))

    result = ModelList(data=data)
    return result.model_dump_json(exclude_none=True).encode()


def get_model(
    model_types: list[str],
    max_input_tokens: list[int],
    max_output_tokens: list[int],
    model_id: str,
) -> bytes | None:
    """查询单个模型"""
    target = model_id.lower()
    for ty in model_types:
        mid = MODEL_ID_MAP.get(ty, f"deepseek-{ty}")
        if mid.lower() == target:
            model = ModelInfo(
                id=mid,
                created=MODEL_CREATED,
                owned_by=MODEL_OWNED_BY,
            )
            return model.model_dump_json(exclude_none=True).encode()
    return None

"""服务器错误类型定义"""

from __future__ import annotations

import json
import uuid

from ..openai_adapter.response import AdapterError
from ..openai_adapter.request import BadRequestError
from ..anthropic_compat.request import AnthropicCompatError


class ServerError(Exception):
    """HTTP 层错误"""

    def __init__(self, status: int, body: dict):
        self.status = status
        self.body = body
        super().__init__(json.dumps(body))

    @classmethod
    def unauthorized(cls) -> ServerError:
        return cls(401, {
            "error": {
                "message": "Invalid API token",
                "type": "authentication_error",
            }
        })

    @classmethod
    def not_found(cls, path: str) -> ServerError:
        return cls(404, {
            "error": {
                "message": f"Not found: {path}",
                "type": "not_found_error",
            }
        })

    @classmethod
    def from_adapter_error(cls, e: AdapterError) -> ServerError:
        """适配器错误 → 统一 HTTP 状态码"""
        status_map = {
            "overloaded": 429,  # Too Many Requests
            "provider_error": 502,  # Bad Gateway
            "provider_timeout": 504,  # Gateway Timeout
            "stream_error": 500,  # Internal Stream Error
            "pow_failed": 422,  # Unprocessable Entity
            "account_unhealthy": 503,  # Service Unavailable
        }
        
        status = status_map.get(e.kind, 500)
        return cls(status, {
            "error": {
                "message": e.message,
                "type": "api_error",
                "code": e.kind,
            }
        })

    @classmethod
    def from_anthropic_error(cls, e: AnthropicCompatError) -> ServerError:
        type_map = {
            "bad_request": "invalid_request_error",
            "overloaded": "overloaded_error",
            "internal": "api_error",
        }
        return cls(e.status_code(), {
            "error": {
                "type": type_map.get(e.kind, "api_error"),
                "message": e.message,
            }
        })

    @classmethod
    def from_bad_request(cls, e: BadRequestError) -> ServerError:
        return cls(400, {
            "error": {
                "message": str(e),
                "type": "invalid_request_error",
            }
        })

    @classmethod
    def internal(cls, msg: str) -> ServerError:
        return cls(500, {
            "error": {
                "message": msg,
                "type": "internal_error",
            }
        })

    def openai_json(self) -> bytes:
        return json.dumps(self.body).encode()

    def anthropic_json(self) -> bytes:
        request_id = f"req_{uuid.uuid4().hex[:24]}"
        error = self.body.get("error", {})
        # 统一生成 Anthropic 格式
        return json.dumps({
            "type": "error",
            "error": {
                "type": error.get("type", "api_error"),
                "message": error.get("message", "unknown error"),
            },
            "request_id": request_id,
        }).encode()

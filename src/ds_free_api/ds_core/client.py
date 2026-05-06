"""DeepSeek HTTP 客户端 —— 原始 API 调用层

无状态管理：无缓存、无重试、无会话状态。
每个方法对应一个 REST 端点。流方法返回原始字节流，由上层解析 SSE。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, AsyncIterator

import httpx
from pydantic import BaseModel, Field

from .waf_bypass import get_waf_cookies

logger = logging.getLogger("ds_free_api.ds_core.client")


class RetryConfig(BaseModel):
    """重试配置"""
    max_attempts: int = 6
    initial_backoff: float = 1.0
    max_backoff: float = 30.0
    jitter: bool = True


_DEFAULT_RETRY_CFG = RetryConfig()


class SmartRetry:
    """智能重试机制：指数退避 + 抖动"""
    
    @staticmethod
    def get_delay(attempt: int, cfg: RetryConfig | None = None) -> float:
        """计算重试延迟（指数退避 + 抖动）"""
        if cfg is None:
            cfg = _DEFAULT_RETRY_CFG
        delay = cfg.initial_backoff * (2 ** (attempt - 1))
        delay = min(delay, cfg.max_backoff)
        
        # 添加抖动，避免雷群效应
        if cfg.jitter:
            jitter = random.uniform(0.8, 1.2)
            delay *= jitter
        
        return delay


# API 端点常量
EP_USERS_LOGIN = "/users/login"
EP_CHAT_SESSION_CREATE = "/chat_session/create"
EP_CHAT_SESSION_DELETE = "/chat_session/delete"
EP_CHAT_SESSION_UPDATE_TITLE = "/chat_session/update_title"
EP_CHAT_CREATE_POW_CHALLENGE = "/chat/create_pow_challenge"
EP_CHAT_COMPLETION = "/chat/completion"
EP_CHAT_EDIT_MESSAGE = "/chat/edit_message"
EP_FILE_UPLOAD = "/file/upload_file"
EP_FILE_FETCH = "/file/fetch_files"


# ============================================================================
# 请求/响应模型
# ============================================================================


class LoginPayload(BaseModel):
    email: str | None = None
    mobile: str | None = None
    password: str
    area_code: str | None = None
    device_id: str = ""
    os: str = "web"

    def to_api_dict(self) -> dict[str, Any]:
        d = self.model_dump(exclude_none=True, exclude={"device_id", "os"})
        d["device_id"] = self.device_id
        d["os"] = self.os
        return d


class UserInfo(BaseModel):
    id: str
    token: str
    email: str | None = None
    mobile_number: str | None = None


class LoginData(BaseModel):
    code: int
    msg: str
    user: UserInfo


class ChallengeData(BaseModel):
    algorithm: str
    challenge: str
    salt: str
    signature: str
    difficulty: int
    expire_after: int
    expire_at: int
    target_path: str


class CompletionPayload(BaseModel):
    chat_session_id: str
    parent_message_id: int | None = None
    model_type: str
    prompt: str
    ref_file_ids: list[str] = Field(default_factory=list)
    thinking_enabled: bool = False
    search_enabled: bool = False
    preempt: bool = False


class EditMessagePayload(BaseModel):
    chat_session_id: str
    message_id: int = 1
    prompt: str
    search_enabled: bool = False
    thinking_enabled: bool = False
    model_type: str = ""


class UpdateTitlePayload(BaseModel):
    chat_session_id: str
    title: str


# ============================================================================
# 客户端错误
# ============================================================================


class ClientError(Exception):
    """HTTP 客户端错误"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class HttpError(ClientError):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body[:200]}")


class BusinessError(ClientError):
    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(f"Business error: code={code}, msg={msg}")


# ============================================================================
# 客户端实现
# ============================================================================


class DsClient:
    """DeepSeek HTTP 客户端 —— 原始 API 调用层（增强重试机制）"""
    
    def __init__(
        self,
        api_base: str,
        wasm_url: str,
        user_agent: str,
        client_version: str,
        client_platform: str,
        timeout: float = 120.0,
        stream_idle_timeout: float = 60.0,
        retry_config: RetryConfig | None = None,
    ):
        self.api_base = api_base
        self.wasm_url = wasm_url
        self.user_agent = user_agent
        self.client_version = client_version
        self.client_platform = client_platform
        self._stream_idle_timeout = stream_idle_timeout
        self._retry_config = retry_config or RetryConfig()
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30.0, read=timeout, write=30.0, pool=30.0),
            follow_redirects=True,
            # 高并发优化：提升连接池上限，支持超多账号同时请求
            limits=httpx.Limits(
                max_connections=200,
                max_keepalive_connections=100,
                keepalive_expiry=30.0,  # 30 秒后回收空闲连接
            ),
        )
        # 缓存 per-token auth headers，避免每次请求创建新 dict
        # AWS WAF bypass 状态（多账号并发登录只触发一次 Playwright）
        self._waf_lock = asyncio.Lock()
        self._waf_ready = False

        self._auth_cache: dict[str, dict[str, str]] = {}
        self._base_headers: dict[str, str] = {
            "User-Agent": user_agent,
            "X-Client-Version": client_version,
            "X-Client-Platform": client_platform,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Origin": "https://chat.deepseek.com",
            "Referer": "https://chat.deepseek.com/sign_in",
            "Content-Type": "application/json",
            "Sec-Ch-Ua": '"Chromium";v="145", "Not:A-Brand";v="99"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Priority": "u=1, i",
        }
        # 浏览器导航专用 headers（GET sign_in 不能用 Content-Type，会 405）
        self._signin_headers: dict[str, str] = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Priority": "u=0, i",
        }

    def _auth_headers(self, token: str) -> dict[str, str]:
        """获取认证 headers（缓存 per-token，避免重复创建 dict）"""
        h = self._auth_cache.get(token)
        if h is None:
            h = {**self._base_headers, "Authorization": f"Bearer {token}"}
            self._auth_cache[token] = h
        return h

    def _auth_headers_with_pow(self, token: str, pow_response: str) -> dict[str, str]:
        """获取认证+PoW headers（直接构建，避免浅拷贝）"""
        return {
            **self._base_headers,
            "Authorization": f"Bearer {token}",
            "X-Ds-Pow-Response": pow_response,
        }

    def _parse_envelope(self, resp: httpx.Response) -> Any:
        """解析 DeepSeek 标准信封格式"""
        if resp.status_code >= 400:
            raise HttpError(resp.status_code, resp.text)

        if not resp.text or not resp.text.strip():
            raise ClientError(f"API 返回空响应 (status={resp.status_code})")

        data = resp.json()
        code = data.get("code", -1)
        msg = data.get("msg", "")

        if code != 0:
            raise BusinessError(code, msg)

        inner = data.get("data")
        if inner is None:
            return None

        biz_code = inner.get("biz_code", 0)
        biz_msg = inner.get("biz_msg", "")
        if biz_code != 0:
            raise BusinessError(biz_code, biz_msg)

        return inner.get("biz_data")

    async def _ensure_waf_cookies(self) -> None:
        """确保 httpx client 拥有有效的 AWS WAF cookies。

        首次调用或 token 失效后（_waf_ready=False）会启动 Playwright 获取新 token。
        多账号并发时通过 _waf_lock 保证只启动一次浏览器。
        """
        if self._waf_ready:
            return
        async with self._waf_lock:
            if self._waf_ready:
                return

            # 直接启动 Playwright 获取新 WAF token（不再探测，更快）
            logger.info("启动 Playwright 获取 WAF cookies...")
            raw_cookies = await get_waf_cookies(user_agent=self.user_agent)

            # 清除旧 WAF cookies，避免新旧 token 混杂
            # httpx CookieJar 没有按 name 删除的 API，直接清空再注入
            self.http.cookies.clear()

            # 使用原始 domain 注入 cookies（如 aws-waf-token domain=.deepseek.com）
            for c in raw_cookies:
                domain = c.get("domain", "chat.deepseek.com")
                path = c.get("path", "/")
                self.http.cookies.set(c["name"], c["value"], domain=domain, path=path)

            logger.info(f"已注入 {len(raw_cookies)} 个 WAF cookies 到 httpx client")
            self._waf_ready = True

    async def login(self, payload: LoginPayload) -> LoginData:
        # 先确保通过 AWS WAF JS Challenge（14 个账号并发也只启动一次 Playwright）
        await self._ensure_waf_cookies()

        # 预热：先访问 sign_in 页面以建立 cookie/session（使用浏览器导航 headers）
        try:
            await self.http.get(
                "https://chat.deepseek.com/sign_in",
                headers=self._signin_headers,
                follow_redirects=True,
            )
        except Exception:
            pass  # 预热失败不影响主流程

        last_resp: httpx.Response | None = None
        last_exc: Exception | None = None
        for attempt in range(1, self._retry_config.max_attempts + 1):
            try:
                resp = await self.http.post(
                    f"{self.api_base}{EP_USERS_LOGIN}",
                    headers=self._base_headers,
                    json=payload.to_api_dict(),
                )
                last_resp = resp
                if resp.status_code >= 400:
                    raise HttpError(resp.status_code, resp.text)

                # 部分情况下服务端会返回 202 Accepted 且无 body（异步受理），稍后重试即可。
                body = resp.text
                if resp.status_code == 202 or not body or not body.strip():
                    raise ClientError(f"login 返回空响应 (status={resp.status_code})")

                data = resp.json()
                break
            except HttpError as e:
                last_exc = e
                # 405 Human Verification = WAF token 失效/超限，刷新后重试
                # 注意：202 不会走 HttpError 分支（202 < 400），由 ClientError 分支处理
                if e.status == 405 and "Human Verification" in e.body:
                    if attempt >= self._retry_config.max_attempts:
                        raise
                    logger.warning(
                        f"login 失败 (status={e.status})，刷新 WAF cookies 后重试"
                    )
                    self._waf_ready = False
                    await self._ensure_waf_cookies()
                    # 重新预热 sign_in 页面
                    try:
                        await self.http.get(
                            "https://chat.deepseek.com/sign_in",
                            headers=self._signin_headers,
                            follow_redirects=True,
                        )
                    except Exception:
                        pass
                    continue
                # 其他 HTTP 错误正常重试
                if attempt >= self._retry_config.max_attempts:
                    raise
                delay = SmartRetry.get_delay(attempt, self._retry_config)
                logger.warning(
                    f"login 失败，第 {attempt} 次重试，等待 {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)
            except ClientError as e:
                last_exc = e
                # login 返回 202 空 body = WAF token 超限，刷新后重试
                if "login 返回空响应" in str(e):
                    if attempt >= self._retry_config.max_attempts:
                        raise
                    logger.warning("login 返回空响应，刷新 WAF cookies 后重试")
                    self._waf_ready = False
                    await self._ensure_waf_cookies()
                    # 重新预热 sign_in 页面
                    try:
                        await self.http.get(
                            "https://chat.deepseek.com/sign_in",
                            headers=self._signin_headers,
                            follow_redirects=True,
                        )
                    except Exception:
                        pass
                    continue
                if attempt >= self._retry_config.max_attempts:
                    raise
                delay = SmartRetry.get_delay(attempt, self._retry_config)
                logger.warning(
                    f"login 失败，第 {attempt} 次重试，等待 {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)
            except Exception as e:
                last_exc = e
                if attempt >= self._retry_config.max_attempts:
                    raise
                delay = SmartRetry.get_delay(attempt, self._retry_config)
                logger.warning(
                    f"login 失败，第 {attempt} 次重试，等待 {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)
        else:
            # 理论不可达：for 循环内要么 break 要么 raise
            raise last_exc or ClientError(
                f"login 失败且无异常信息 (status={last_resp.status_code if last_resp else 'unknown'})"
            )

        code = data.get("code", -1)
        msg = data.get("msg", "")
        if code != 0:
            raise BusinessError(code, msg)

        inner = data.get("data", {})
        biz_code = inner.get("biz_code", 0)
        if biz_code != 0:
            raise BusinessError(biz_code, inner.get("biz_msg", ""))

        user_data = inner.get("biz_data", {})
        user_info = user_data.get("user", {})
        return LoginData(
            code=code,
            msg=msg,
            user=UserInfo(
                id=user_info.get("id", ""),
                token=user_info.get("token", ""),
                email=user_info.get("email"),
                mobile_number=user_info.get("mobile_number"),
            ),
        )

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        **kwargs: Any,
    ) -> httpx.Response:
        """带重试的 HTTP 请求"""
        cfg = self._retry_config
        last_exc: Exception | None = None

        for attempt in range(1, cfg.max_attempts + 1):
            try:
                resp = await self.http.request(method, url, headers=headers, **kwargs)
                if resp.status_code < 500:
                    return resp
                last_exc = HttpError(resp.status_code, resp.text[:200])
            except (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.PoolTimeout) as e:
                last_exc = ClientError(f"网络错误: {e}")

            if attempt < cfg.max_attempts:
                delay = SmartRetry.get_delay(attempt)
                logger.warning(f"请求失败，第 {attempt} 次重试，等待 {delay:.1f}s: {last_exc}")
                await asyncio.sleep(delay)

        raise last_exc or ClientError("重试次数耗尽")

    async def create_session(self, token: str) -> str:
        resp = await self._request_with_retry(
            "POST",
            f"{self.api_base}{EP_CHAT_SESSION_CREATE}",
            self._auth_headers(token),
            json={},
        )
        
        try:
            # DeepSeek API 响应格式：data.biz_data.chat_session
            response_json = resp.json()
            data = response_json.get("data", {})
            biz_data = data.get("biz_data", {})
            chat_session = biz_data.get("chat_session", {})
            
            # 检查响应格式，添加调试信息
            if not chat_session:
                logger.error(f"session 创建响应格式异常: biz_data={biz_data}, data={data}, 完整响应: {response_json}")
                raise ClientError("session 创建响应格式异常：chat_session 为空")
            
            if "id" not in chat_session:
                logger.error(f"session 创建响应缺少 id 字段: chat_session={chat_session}")
                raise ClientError("session 创建响应格式异常：缺少 id 字段")
            
            session_id = chat_session["id"]
            logger.debug(f"session 创建成功: {session_id}")
            return session_id
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"session 创建响应解析失败: {e}, 响应内容: {resp.text[:500]}")
            raise ClientError(f"session 创建响应解析失败: {e}")

    async def delete_session(self, token: str, session_id: str) -> None:
        resp = await self._request_with_retry(
            "POST",
            f"{self.api_base}{EP_CHAT_SESSION_DELETE}",
            self._auth_headers(token),
            json={"chat_session_id": session_id},
        )
        data = resp.json()
        code = data.get("code", -1)
        msg = data.get("msg", "")
        if code != 0:
            raise BusinessError(code, msg)

    async def create_pow_challenge(self, token: str, target_path: str = "/api/v0/chat/completion") -> ChallengeData:
        resp = await self._request_with_retry(
            "POST",
            f"{self.api_base}{EP_CHAT_CREATE_POW_CHALLENGE}",
            self._auth_headers(token),
            json={"target_path": target_path},
        )
        envelope = resp.json()
        data = envelope.get("data", {})
        biz_data = data.get("biz_data", {})
        challenge = biz_data.get("challenge", {})
        if not challenge:
            raise BusinessError(-1, "missing challenge in pow response")
        return ChallengeData(**challenge)

    async def _with_idle_timeout(self, stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """为流添加逐 chunk 空闲超时，防止上游停止发送时挂起"""
        timeout = self._stream_idle_timeout
        async_iterator = stream.__aiter__()
        
        while True:
            try:
                chunk = await asyncio.wait_for(async_iterator.__anext__(), timeout=timeout)
                yield chunk
            except asyncio.TimeoutError:
                raise ClientError(f"流空闲超时（{timeout}s 无数据）")
            except StopAsyncIteration:
                break

    async def completion_stream(
        self,
        token: str,
        pow_response: str,
        payload: CompletionPayload,
    ) -> AsyncIterator[bytes]:
        """流式补全（带空闲超时）

        注意：completion 端点 HTTP 200 时可能仍含业务错误（biz_code 非 0），
        需检查首个 chunk。
        """
        async with self.http.stream(
            "POST",
            f"{self.api_base}{EP_CHAT_COMPLETION}",
            headers=self._auth_headers_with_pow(token, pow_response),
            json=payload.model_dump(exclude_none=True),
        ) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise HttpError(resp.status_code, body.decode(errors="replace"))

            first_chunk = True
            async for chunk in self._with_idle_timeout(resp.aiter_bytes()):
                if first_chunk:
                    first_chunk = False
                    # 检查是否为业务错误（非 SSE 格式的 JSON envelope）
                    text = chunk.decode("utf-8", errors="replace").strip()
                    if text.startswith("{") and "biz_code" in text:
                        try:
                            envelope = json.loads(text)
                            biz_code = envelope.get("data", {}).get("biz_code", 0)
                            biz_msg = envelope.get("data", {}).get("biz_msg", "")
                            if biz_code != 0:
                                raise ClientError(
                                    f"completion 业务错误: biz_code={biz_code}, msg={biz_msg}"
                                )
                        except ClientError:
                            raise
                        except (json.JSONDecodeError, KeyError, TypeError):
                            pass  # 非 envelope 格式，可能是 SSE 数据中碰巧包含 biz_code
                yield chunk

    async def edit_message_stream(
        self,
        token: str,
        pow_response: str,
        payload: EditMessagePayload,
    ) -> AsyncIterator[bytes]:
        """编辑消息流式接口（带空闲超时）"""
        async with self.http.stream(
            "POST",
            f"{self.api_base}{EP_CHAT_EDIT_MESSAGE}",
            headers=self._auth_headers_with_pow(token, pow_response),
            json=payload.model_dump(exclude_none=True),
        ) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise HttpError(resp.status_code, body.decode(errors="replace"))
            async for chunk in self._with_idle_timeout(resp.aiter_bytes()):
                yield chunk

    async def update_title(self, token: str, payload: UpdateTitlePayload) -> None:
        resp = await self._request_with_retry(
            "POST",
            f"{self.api_base}{EP_CHAT_SESSION_UPDATE_TITLE}",
            self._auth_headers(token),
            json=payload.model_dump(),
        )
        self._parse_envelope(resp)

    async def get_wasm(self) -> bytes:
        last_exc: Exception | None = None
        for attempt in range(1, self._retry_config.max_attempts + 1):
            try:
                resp = await self.http.get(self.wasm_url)
                if resp.status_code < 500:
                    if resp.status_code >= 400:
                        raise HttpError(resp.status_code, resp.text[:200])
                    return resp.content
                last_exc = HttpError(resp.status_code, resp.text[:200])
            except (httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.PoolTimeout) as e:
                last_exc = ClientError(f"网络错误: {e}")
            if attempt < self._retry_config.max_attempts:
                delay = SmartRetry.get_delay(attempt)
                logger.warning(f"WASM 下载失败，第 {attempt} 次重试，等待 {delay:.1f}s: {last_exc}")
                await asyncio.sleep(delay)
        raise last_exc or ClientError("WASM 下载重试次数耗尽")

    async def upload_file(
        self,
        token: str,
        pow_response: str,
        filename: str,
        data: bytes,
    ) -> str:
        """上传文件，返回 file_id

        DeepSeek 网页版 multipart 字段名为 "text"，content_type 统一 "text/plain"，
        无论实际文件类型如何（图片/PDF/文档均以 text/plain 上传）。
        """
        files = {"text": (filename, data, "text/plain")}
        resp = await self._request_with_retry(
            "POST",
            f"{self.api_base}{EP_FILE_UPLOAD}",
            self._auth_headers_with_pow(token, pow_response),
            files=files,
        )
        result = self._parse_envelope(resp)
        if isinstance(result, dict):
            return result.get("id", "")
        return ""

    async def fetch_files(self, token: str, file_ids: list[str]) -> list[dict]:
        """查询文件状态"""
        ids = ",".join(file_ids)
        resp = await self._request_with_retry(
            "GET",
            f"{self.api_base}{EP_FILE_FETCH}",
            self._auth_headers(token),
            params={"file_ids": ids},
        )
        result = self._parse_envelope(resp)
        if isinstance(result, dict):
            files = result.get("files", [])
            if isinstance(files, list):
                return files
        if isinstance(result, list):
            return result
        return []

    async def close(self) -> None:
        await self.http.aclose()

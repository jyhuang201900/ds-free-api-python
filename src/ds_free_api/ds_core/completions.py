"""对话请求编排 —— 调用 completion 返回 SSE 流"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator

from .accounts import Account, AccountPool, PoolError
from .client import ClientError, CompletionPayload, DsClient, EditMessagePayload, HttpError
from .pow import PowError, PowResult, PowSolver

logger = logging.getLogger("ds_free_api.ds_core.completions")


@dataclass
class ChatRequest:
    """对话请求"""

    prompt: str
    thinking_enabled: bool = False
    search_enabled: bool = False
    model_type: str = "default"
    ref_file_ids: list[str] = field(default_factory=list)


class CoreError(Exception):
    """核心层错误"""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.message = message
        super().__init__(message)

    @classmethod
    def overloaded(cls) -> CoreError:
        return cls("overloaded", "no available account")

    @classmethod
    def provider_error(cls, msg: str) -> CoreError:
        return cls("provider", msg)

    @classmethod
    def stream_error(cls, msg: str) -> CoreError:
        return cls("stream", msg)

    @classmethod
    def pow_failed(cls, msg: str) -> CoreError:
        return cls("pow", msg)


class Completions:
    """对话编排"""

    def __init__(self, client: DsClient, solver: PowSolver, pool: AccountPool):
        self.client = client
        self.solver = solver
        self.pool = pool

    async def v0_chat(self, req: ChatRequest, image_attachments: list[dict] | None = None, message_id_sink: list | None = None) -> AsyncIterator[bytes]:
        """发起对话请求，返回 SSE 流

        完整历史模式：
        - prompt 包含完整对话历史（ChatML 格式），无需依赖 parent_message_id
        - 有文件附件时附加 ref_file_ids
        - edit_message 仅作为 completion 失败时的 fallback

        message_id_sink: 可选的 list，StreamState 提取到 message_id 后写入 sink[0]，
                         由上层 response.py 回传，避免 SSE 双重解析。
        """
        has_attachments = bool(image_attachments) or bool(req.ref_file_ids)
        try:
            async with self.pool.get_account(req.model_type) as account:
                # 不健康账号尝试恢复
                if not account.is_healthy:
                    logger.info(f"账号 {account.email or account.mobile} 不健康，尝试重新初始化")
                    reinit_ok = await self.pool.reinit_account(
                        account, self.client, self.solver,
                    )
                    if not reinit_ok:
                        raise CoreError.provider_error(
                            f"账号 {account.email or account.mobile} 重新初始化失败"
                        )

                token = account.token

                # session 轮换：消息数超阈值时重建
                if account.should_rotate_session(req.model_type):
                    await self._rotate_session(account, req.model_type)

                session_id = account.sessions.get(req.model_type)
                if session_id is None:
                    raise CoreError.provider_error(f"账号缺少 {req.model_type} 的 session")

                # 完整历史模式：不再需要 parent_message_id
                parent_message_id = None

                # 处理文件附件（与 PoW 并行：文件上传需要独立的 PoW）
                ref_file_ids = list(req.ref_file_ids)
                upload_task = None
                if image_attachments:
                    # 启动图片上传（异步，不阻塞 PoW 计算）
                    upload_task = asyncio.create_task(self._upload_images(token, image_attachments))

                try:
                    # 计算 PoW（completion 端点）
                    pow_header = await self._compute_pow(
                        token, target_path="/api/v0/chat/completion", account=account,
                    )

                    # 等待图片上传完成（与 PoW 并行执行）
                    if upload_task is not None:
                        try:
                            file_ids = await upload_task
                            ref_file_ids.extend(file_ids)
                        except Exception as e:
                            logger.warning(f"图片上传失败: {e}")

                    # 如果文件上传全部失败，附加提示
                    if has_attachments and not ref_file_ids:
                        logger.info("所有文件上传失败或处理失败，prompt 中附加提示")

                    # 构建 completion payload
                    payload = CompletionPayload(
                        chat_session_id=session_id,
                        parent_message_id=parent_message_id,
                        model_type=req.model_type,
                        prompt=req.prompt,
                        ref_file_ids=ref_file_ids,
                        thinking_enabled=req.thinking_enabled,
                        search_enabled=False if ref_file_ids else req.search_enabled,
                        preempt=False,
                    )

                    try:
                        async for chunk in self.client.completion_stream(token, pow_header, payload):
                            yield chunk
                        # completion 成功：增加消息计数
                        account.incr_message_count(req.model_type)
                    except ClientError as e:
                        # completion 失败时 fallback 到 edit_message
                        logger.warning(f"completion 失败，fallback 到 edit_message: {e}")
                        try:
                            pow_header2 = await self._compute_pow(
                                token, target_path="/api/v0/chat/edit_message", account=account,
                            )
                            fallback_payload = EditMessagePayload(
                                chat_session_id=session_id,
                                message_id=1,
                                prompt=req.prompt,
                                search_enabled=req.search_enabled,
                                thinking_enabled=req.thinking_enabled,
                                model_type=req.model_type,
                            )
                            async for chunk in self.client.edit_message_stream(token, pow_header2, fallback_payload):
                                yield chunk
                            # fallback 成功：增加消息计数
                            account.incr_message_count(req.model_type)
                        except ClientError as e2:
                            account.mark_unhealthy()
                            raise CoreError.provider_error(str(e2))
                    except Exception as e:
                        # 流式响应异常（客户端断开、网络错误等），记录后正常结束
                        logger.warning(f"v0_chat 流式异常: {e}")
                        return
                finally:
                    # 确保取消未完成的图片上传任务，防止孤儿任务
                    if upload_task is not None and not upload_task.done():
                        upload_task.cancel()
                        try:
                            await upload_task
                        except asyncio.CancelledError:
                            pass

        except PoolError:
            raise CoreError.overloaded()
        except ClientError as e:
            raise CoreError.provider_error(str(e))
        except CoreError:
            raise
        except Exception as e:
            logger.warning(f"v0_chat 异常: {e}")
            raise CoreError.stream_error(str(e))

    async def _rotate_session(self, account: Account, model_type: str) -> None:
        """轮换 session：删除旧的，创建新的，重置消息计数"""
        old_session_id = account.sessions.get(model_type)
        try:
            if old_session_id:
                await self.client.delete_session(account.token, old_session_id)
                logger.info(f"账号 {account.email or account.mobile} 轮换 session {model_type}: 删除旧 {old_session_id[:8]}...")
        except Exception as e:
            logger.debug(f"删除旧 session 失败（不影响）: {e}")

        try:
            new_session_id = await self.client.create_session(account.token)
            account.sessions[model_type] = new_session_id
            account.reset_message_count(model_type)

            # health_check：轻量验证 token 有效性
            try:
                await self.pool._health_check(
                    account.token, new_session_id,
                    self.client, self.solver, model_type, account,
                )
            except Exception as e:
                logger.debug(f"轮换 session health_check 失败（不影响）: {e}")

            logger.info(f"账号 {account.email or account.mobile} 轮换 session {model_type}: 新 {new_session_id[:8]}...")
            self.pool._cache_dirty = True
        except Exception as e:
            logger.warning(f"创建新 session 失败: {e}")

    async def _upload_images(self, token: str, images: list[dict]) -> list[str]:
        """上传图片文件，返回成功处理的 file_id 列表

        仅返回状态为 SUCCESS 的文件 ID，CONTENT_EMPTY/ERROR 的会被过滤掉。
        并行上传多张图片，加速处理。
        """
        # 计算 PoW 用于文件上传
        try:
            pow_header = await self._compute_pow(token, target_path="/api/v0/file/upload_file")
        except CoreError:
            logger.warning("图片上传 PoW 计算失败，跳过图片上传")
            return []

        async def _upload_one(img: dict) -> str | None:
            """上传单张图片，成功返回 file_id，失败返回 None"""
            try:
                data = img.get("data")
                if data is None:
                    url = img.get("url", "")
                    if url:
                        resp = await self.client.http.get(url, timeout=30)
                        if resp.status_code >= 400:
                            logger.warning(f"图片下载失败: {url} -> HTTP {resp.status_code}")
                            return None
                        data = resp.content
                    else:
                        return None

                file_id = await self.client.upload_file(
                    token=token,
                    pow_response=pow_header,
                    filename=img.get("filename", "image.png"),
                    data=data,
                )
                if file_id:
                    logger.info(f"图片上传成功: {img.get('filename', '?')} -> {file_id}")
                return file_id
            except Exception as e:
                logger.warning(f"图片上传失败: {img.get('filename', '?')}: {e}")
                return None

        # 并行上传所有图片
        results = await asyncio.gather(*[_upload_one(img) for img in images])
        uploaded_ids = [fid for fid in results if fid]

        # 等待文件处理完成，仅保留 SUCCESS 的文件
        if uploaded_ids:
            return await self._wait_for_files(token, uploaded_ids)

        return []

    async def _wait_for_files(self, token: str, file_ids: list[str], max_wait: float = 30.0) -> list[str]:
        """等待文件处理完成，返回 SUCCESS 的 file_id 列表

        文件状态流转：PENDING → PARSING → SUCCESS / CONTENT_EMPTY / ERROR
        仅 SUCCESS 的文件可被 completion 引用，其他状态会导致
        "invalid ref file id" 错误。
        渐进式轮询：首次0.5s，逐步增加到2.5s，平衡速度和API压力。
        """
        remaining = list(file_ids)
        success_ids: list[str] = []
        deadline = asyncio.get_running_loop().time() + max_wait
        poll_count = 0
        poll_intervals = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5]  # 渐进式间隔

        while remaining and asyncio.get_running_loop().time() < deadline:
            interval = poll_intervals[min(poll_count, len(poll_intervals) - 1)]
            await asyncio.sleep(interval)
            poll_count += 1
            try:
                files = await self.client.fetch_files(token, remaining)
                done_ids = set()
                for f in files:
                    fid = f.get("id", "")
                    status = f.get("status", "")
                    if status not in ("PENDING", "PARSING"):
                        if status == "SUCCESS":
                            success_ids.append(fid)
                        else:
                            logger.warning(f"文件处理失败: {fid[:16]}..., status={status}")
                        done_ids.add(fid)
                remaining = [fid for fid in remaining if fid not in done_ids]
            except Exception as e:
                logger.debug(f"查询文件状态失败: {e}")

        if remaining:
            logger.warning(f"文件处理超时，仍有 {len(remaining)} 个未完成")
        else:
            logger.info(f"文件处理完成，共轮询 {poll_count} 次，成功 {len(success_ids)}/{len(file_ids)}")

        return success_ids

    async def _compute_pow(self, token: str, target_path: str = "/api/v0/chat/completion", account: Account | None = None) -> str:
        """计算 PoW（在线程池中执行避免阻塞事件循环）"""
        try:
            challenge = await self.client.create_pow_challenge(token, target_path=target_path)
            result: PowResult = await asyncio.to_thread(self.solver.solve, challenge)
            return result.to_header()
        except HttpError as e:
            # token 过期（401/403），标记账号不健康
            if e.status in (401, 403):
                if account:
                    account.mark_unhealthy()
                raise CoreError.provider_error(f"token expired (HTTP {e.status})")
            raise CoreError.pow_failed(f"proof of work failed: {e}")
        except Exception as e:
            raise CoreError.pow_failed(f"proof of work failed: {e}")

    def account_statuses(self):
        return self.pool.account_statuses()

    async def shutdown(self) -> None:
        await self.pool.shutdown(self.client)

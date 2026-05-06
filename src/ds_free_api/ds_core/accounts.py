"""账号池管理 —— 多账号负载均衡

1 account = 1 session = 1 concurrency。多并发需横向扩展账号数。
改进：支持 token 过期自动重新登录和 session 失效重建。
支持账号状态缓存，避免每次启动都重新初始化。

优化：
- 启动速度：并行验证缓存账号 + 懒健康检查（跳过 health_check）
- 高并发：per-model-type 锁减少争用 + 请求排队等待账号释放
- 超多账号：后台健康监控自动恢复 + 高效选择算法
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from ..config import AccountConfig
from .client import (
    BusinessError,
    ClientError,
    DsClient,
    HttpError,
    LoginPayload,
    UpdateTitlePayload,
)
from .pow import PowError, PowSolver

# 启动优化：缓存验证并发度
VERIFY_CONCURRENCY = 8
# 后台健康监控间隔（秒）- 缩短到 60s 更快发现不健康账号
HEALTH_MONITOR_INTERVAL = 60.0
# 请求排队等待超时（秒）
QUEUE_WAIT_TIMEOUT = 30.0

logger = logging.getLogger("ds_free_api.ds_core.accounts")

# 默认缓存文件路径
DEFAULT_CACHE_FILE = "accounts_cache.json"


class AccountStatus:
    """账号状态信息"""

    def __init__(self, email: str, mobile: str, busy: bool, healthy: bool):
        self.email = email
        self.mobile = mobile
        self.busy = busy
        self.healthy = healthy


class Account:
    """单个账号"""

    def __init__(
        self,
        token: str,
        email: str,
        mobile: str,
        sessions: dict[str, str],
        creds: AccountConfig,
    ):
        self.token = token
        self.email = email
        self.mobile = mobile
        self.sessions = sessions  # model_type -> session_id
        self.creds = creds
        self._busy = False
        self._last_released: float = 0.0
        self._healthy = True
        # 熔断机制：失败计数和冷却时间
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._circuit_cooldown = 300  # 5分钟冷却时间
        # session 消息计数（用于决定何时重建 session）
        self._message_counts: dict[str, int] = {mt: 0 for mt in sessions}
        self._max_messages_per_session = 200  # 多轮对话由后端管理，阈值可提高
        # 多轮对话：per-model-type 的最后 response message_id
        self._parent_message_ids: dict[str, int | None] = {mt: None for mt in sessions}

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    def mark_busy(self) -> None:
        self._busy = True

    def mark_free(self) -> None:
        self._busy = False
        self._last_released = time.time()

    def mark_unhealthy(self) -> None:
        self._healthy = False
        self._failure_count += 1
        self._last_failure_time = time.time()

    def mark_healthy(self) -> None:
        self._healthy = True
        self._failure_count = 0
        self._last_failure_time = 0.0

    @property
    def is_circuit_open(self) -> bool:
        """检查账号是否处于熔断冷却期"""
        if self._failure_count < 3:
            return False
        cooldown_remaining = self._circuit_cooldown - (time.time() - self._last_failure_time)
        return cooldown_remaining > 0

    @property
    def idle_seconds(self) -> float:
        if self._busy:
            return 0.0
        return time.time() - self._last_released

    def incr_message_count(self, model_type: str) -> int:
        """增加 session 消息计数，返回当前计数"""
        count = self._message_counts.get(model_type, 0) + 1
        self._message_counts[model_type] = count
        return count

    def get_parent_message_id(self, model_type: str) -> int | None:
        """获取该 model_type 的最后 response message_id（用于多轮对话）"""
        return self._parent_message_ids.get(model_type)

    def set_parent_message_id(self, model_type: str, msg_id: int | None) -> None:
        """设置该 model_type 的最后 response message_id"""
        self._parent_message_ids[model_type] = msg_id

    def reset_message_count(self, model_type: str) -> None:
        """重置 session 消息计数和 parent_message_id"""
        self._message_counts[model_type] = 0
        self._parent_message_ids[model_type] = None

    def reset_all_message_state(self, model_types: list[str]) -> None:
        """重置所有 model_type 的消息计数和 parent_message_id"""
        for mt in model_types:
            self._message_counts[mt] = 0
            self._parent_message_ids[mt] = None

    def should_rotate_session(self, model_type: str) -> bool:
        """检查 session 是否需要重建"""
        return self._message_counts.get(model_type, 0) >= self._max_messages_per_session

    def selection_score(self) -> float:
        """计算账号选择分数（越高越优先）"""
        # 基础分：空闲时间（最多贡献 100 分）
        idle_score = min(self.idle_seconds / 60.0, 100.0)  # 每分钟 1 分，上限 100
        
        # 健康分：健康状态贡献 50 分
        health_score = 50.0 if self.is_healthy else 0.0
        
        # 惩罚分：失败次数惩罚（每次失败扣 20 分）
        failure_penalty = self._failure_count * 20.0
        
        # 熔断惩罚：熔断期惩罚 100 分
        circuit_penalty = 100.0 if self.is_circuit_open else 0.0
        
        total = idle_score + health_score - failure_penalty - circuit_penalty
        return max(total, 0.0)  # 确保非负


class PoolError(Exception):
    """账号池错误"""

    pass


class AccountPool:
    """账号池（优化：per-model-type 锁 + 请求排队 + 后台健康监控）"""

    def __init__(self, cache_file: str = DEFAULT_CACHE_FILE):
        self._accounts: list[Account] = []
        self._by_type: dict[str, list[Account]] = {}  # per-model-type 索引，O(1)选择
        self._type_locks: dict[str, asyncio.Lock] = {}  # per-model-type 锁
        self._type_events: dict[str, asyncio.Event] = {}  # 账号释放通知
        self._cache_lock = asyncio.Lock()  # 缓存读写锁（独立）
        self._cache_file = Path(cache_file)
        self._cache_dirty = False  # 缓存脏标记
        self._last_cache_save_time: float = 0.0
        self._cache_save_interval = 60.0  # 最小保存间隔（秒）
        self._monitor_task: asyncio.Task | None = None  # 后台健康监控
        self._client: DsClient | None = None  # 供后台监控使用
        self._solver: PowSolver | None = None  # 供后台监控使用

    async def init(
        self,
        creds_list: list[AccountConfig],
        model_types: list[str],
        client: DsClient,
        solver: PowSolver,
    ) -> None:
        """初始化账号池：优先从缓存加载，缓存不存在或失效的账号才重新初始化"""
        cached = self._load_cache()
        accounts: list[Account] = []
        need_init: list[AccountConfig] = []

        # 从缓存恢复
        if cached:
            for creds in creds_list:
                key = creds.email or creds.mobile
                entry = cached.get(key)
                if entry and entry.get("sessions"):
                    acct = Account(
                        token=entry["token"],
                        email=creds.email,
                        mobile=creds.mobile,
                        sessions=entry["sessions"],
                        creds=creds,
                    )
                    # 恢复 parent_message_ids（多轮对话链）
                    saved_pmids = entry.get("parent_message_ids", {})
                    for mt, mid in saved_pmids.items():
                        if mt in acct._parent_message_ids and mid is not None:
                            acct._parent_message_ids[mt] = mid
                    accounts.append(acct)
                else:
                    need_init.append(creds)

            if accounts:
                logger.info(f"从缓存恢复 {len(accounts)} 个账号，并行验证中...")
                # 并行验证缓存账号（并发度 VERIFY_CONCURRENCY）
                verify_sem = asyncio.Semaphore(VERIFY_CONCURRENCY)

                async def _verify_parallel(acct: Account) -> tuple[Account, bool]:
                    async with verify_sem:
                        ok = await self._verify_cached_account(acct, client, solver)
                        return (acct, ok)

                results = await asyncio.gather(
                    *[_verify_parallel(acct) for acct in accounts],
                    return_exceptions=True,
                )
                valid_accounts = []
                for r in results:
                    if isinstance(r, Exception):
                        logger.warning(f"缓存验证异常: {r}")
                        continue
                    acct, ok = r
                    if ok:
                        valid_accounts.append(acct)
                        logger.info(f"账号 {acct.email or acct.mobile} 缓存验证通过")
                    else:
                        logger.info(f"账号 {acct.email or acct.mobile} 缓存验证失败，重新初始化")
                        need_init.append(acct.creds)
                accounts = valid_accounts
        else:
            need_init = list(creds_list)

        # 缓存未命中的账号才需要初始化（并发 + login 失败自动刷新 WAF token）
        if need_init:
            # client.py 已有自动刷新机制：login 遇到 202/405 会立即获取新 WAF token
            # 因此可以较高并发，WAF token 用完会自动续期
            semaphore = asyncio.Semaphore(3)

            async def _limited_init(creds: AccountConfig):
                async with semaphore:
                    # 短延迟避免同时请求
                    await asyncio.sleep(random.uniform(2, 4))
                    return await self._init_account(creds, model_types, client, solver)

            tasks = [_limited_init(creds) for creds in need_init]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for creds, result in zip(need_init, results):
                display_id = creds.email or creds.mobile
                if isinstance(result, Exception):
                    logger.warning(f"账号 {display_id} 初始化失败: {result}")
                else:
                    logger.info(f"账号 {display_id} 初始化成功")
                    accounts.append(result)

        # 允许部分账号失败，只要有至少一个成功就启动
        if not accounts:
            logger.error("所有账号初始化失败，服务无法启动")
            raise PoolError("所有账号初始化失败")
        
        # 如果成功账号少于总账号的 30%，发出警告但仍启动
        success_rate = len(accounts) / len(creds_list)
        if success_rate < 0.3:
            logger.warning(
                f"账号成功率较低: {len(accounts)}/{len(creds_list)} ({success_rate:.1%})，"
                "建议检查账号状态或网络连接"
            )

        self._accounts = accounts

        # 构建 per-model-type 索引 + 初始化锁和事件
        self._by_type.clear()
        for acct in accounts:
            for mt in acct.sessions:
                self._by_type.setdefault(mt, []).append(acct)
        for mt in self._by_type:
            self._type_locks[mt] = asyncio.Lock()
            self._type_events[mt] = asyncio.Event()
            self._type_events[mt].set()  # 初始：有可用账号

        failed = len(creds_list) - len(accounts)
        cached_count = len(creds_list) - len(need_init)
        logger.info(
            f"账号池初始化完成: {len(accounts)}/{len(creds_list)} 成功"
            + (f" (缓存恢复 {cached_count}, 新初始化 {len(accounts) - cached_count})" if cached_count else "")
            + (f", {failed} 失败" if failed else "")
        )

        # 保存缓存
        await self._save_cache(force=True)

        # 启动后台健康监控
        self._client = client
        self._solver = solver
        self._monitor_task = asyncio.create_task(self._health_monitor_loop())

    async def _init_account(
        self,
        creds: AccountConfig,
        model_types: list[str],
        client: DsClient,
        solver: PowSolver,
    ) -> Account:
        """初始化单个账号（带重试，使用指数退避 + 抖动）"""
        last_error = None
        base_delay = 2.0  # 基础延迟 2 秒
        max_delay = 30.0  # 最大延迟 30 秒
        
        for attempt in range(3):
            try:
                return await self._try_init_account(creds, model_types, client, solver)
            except Exception as e:
                last_error = e
                if attempt < 2:
                    # 指数退避：base_delay * 2^attempt + 随机抖动
                    exponential_delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = exponential_delay * 0.2  # 20% 抖动
                    wait = exponential_delay + random.uniform(-jitter, jitter)
                    logger.debug(f"账号 {creds.email or creds.mobile} 第 {attempt+1} 次失败，{wait:.1f}s 后重试: {e}")
                    await asyncio.sleep(wait)
        raise last_error  # type: ignore

    async def _try_init_account(
        self,
        creds: AccountConfig,
        model_types: list[str],
        client: DsClient,
        solver: PowSolver,
    ) -> Account:
        """尝试初始化单个账号（登录 + 并行创建 session + health_check）"""
        # 生成 device_id：浏览器使用 B+base64 随机字符串
        device_id = "B" + base64.b64encode(os.urandom(96)).decode()
        login_payload = LoginPayload(
            email=creds.email or None,
            mobile=creds.mobile or None,
            password=creds.password,
            area_code=creds.area_code or None,
            device_id=device_id,
        )

        login_data = await client.login(login_payload)
        logger.debug(
            f"登录响应: code={login_data.code}, user_id={login_data.user.id}"
        )
        token = login_data.user.token

        # 并行创建所有 model_type 的 session
        async def _create_and_setup(mt: str) -> tuple[str, str | None]:
            """创建 session + health_check + title，返回 (model_type, session_id)"""
            try:
                session_id = await client.create_session(token)
                # 轻量 health_check：仅验证 token 有效性
                try:
                    await self._health_check(token, session_id, client, solver, mt)
                except Exception as e:
                    logger.debug(f"账号 {creds.email or creds.mobile} session {mt} health_check 失败（不影响可用性）: {e}")
                # title 更新是装饰性的，失败不影响可用性
                try:
                    title_payload = UpdateTitlePayload(
                        chat_session_id=session_id,
                        title=f"auto-managed-{mt}-DO-NOT-DELETE",
                    )
                    await client.update_title(token, title_payload)
                except Exception as e:
                    logger.debug(f"账号 {creds.email or creds.mobile} session {mt} title 更新失败（不影响可用性）: {e}")
                return (mt, session_id)
            except Exception as e:
                logger.warning(f"账号 {creds.email or creds.mobile} session {mt} 创建失败: {e}")
                return (mt, None)

        results = await asyncio.gather(
            *[_create_and_setup(mt) for mt in model_types],
            return_exceptions=True,
        )

        sessions: dict[str, str] = {}
        for r in results:
            if isinstance(r, Exception):
                continue
            mt, sid = r
            if sid:
                sessions[mt] = sid

        if not sessions:
            raise RuntimeError("所有 session 创建失败")

        return Account(
            token=token,
            email=creds.email,
            mobile=creds.mobile,
            sessions=sessions,
            creds=creds,
        )

    async def _health_check(
        self,
        token: str,
        session_id: str,
        client: DsClient,
        solver: PowSolver,
        model_type: str,
        account: Account | None = None,
    ) -> int | None:
        """轻量健康检查：仅验证 token 有效性（create_pow_challenge），不发消息

        不再消耗 completion 配额。parent_message_id 在首次真正请求时自然建立。
        返回 None（不再有 message_id）。
        """
        try:
            await client.create_pow_challenge(token, target_path="/api/v0/chat/completion")
            logger.debug(f"health_check 完成 model_type={model_type}（轻量验证）")
        except (ClientError, HttpError, BusinessError) as e:
            raise ClientError(f"health_check 失败: {e}")
        return None

    async def _verify_cached_account(
        self,
        account: Account,
        client: DsClient,
        solver: PowSolver,
    ) -> bool:
        """验证缓存恢复的账号：对第一个 session 做 health_check"""
        # 优先验证 default session
        for model_type in ("default", *account.sessions.keys()):
            session_id = account.sessions.get(model_type)
            if session_id is None:
                continue
            try:
                await self._health_check(account.token, session_id, client, solver, model_type, account)
                return True
            except Exception as e:
                logger.debug(f"账号 {account.email or account.mobile} 验证 session {model_type} 失败: {e}")
        return False

    def _get_type_lock(self, model_type: str) -> asyncio.Lock:
        """获取 model_type 对应的锁（懒创建）"""
        if model_type not in self._type_locks:
            self._type_locks[model_type] = asyncio.Lock()
            self._type_events[model_type] = asyncio.Event()
            self._type_events[model_type].set()
        return self._type_locks[model_type]

    def _get_type_event(self, model_type: str) -> asyncio.Event:
        """获取 model_type 对应的事件（懒创建）"""
        if model_type not in self._type_events:
            self._type_locks[model_type] = asyncio.Lock()
            self._type_events[model_type] = asyncio.Event()
            self._type_events[model_type].set()
        return self._type_events[model_type]

    @asynccontextmanager
    async def get_account(self, model_type: str) -> AsyncIterator[Account]:
        """获取空闲账号（per-model-type 锁 + 请求排队等待）

        优化：
        - 使用 per-model-type 锁，不同模型类型的请求互不阻塞
        - 所有账号忙时排队等待，超时后报错（而非立即报错）
        """
        lock = self._get_type_lock(model_type)
        event = self._get_type_event(model_type)

        account = None
        async with lock:
            account = self._select_account(model_type)
            if account is not None:
                account.mark_busy()

        # 无可用账号：循环等待其他请求释放，直到超时
        if account is None:
            deadline = asyncio.get_running_loop().time() + QUEUE_WAIT_TIMEOUT
            while asyncio.get_running_loop().time() < deadline:
                event.clear()
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

                # 重新尝试获取
                async with lock:
                    account = self._select_account(model_type)
                    if account is not None:
                        account.mark_busy()
                        break

            if account is None:
                raise PoolError(f"no available account for {model_type} (waited {QUEUE_WAIT_TIMEOUT}s)")

        try:
            yield account
        finally:
            account.mark_free()
            # 通知排队中的请求
            event.set()

    def _select_account(self, model_type: str) -> Account | None:
        """选择账号（高效：使用 per-model-type 索引，O(k) k=该类型账号数）"""
        candidates = self._by_type.get(model_type)
        if not candidates:
            return None

        # 单遍扫描：选评分最高的可用账号
        best: Account | None = None
        best_score = -1.0
        for a in candidates:
            if a.is_busy or a.is_circuit_open:
                continue
            score = a.selection_score()
            if score > best_score:
                best_score = score
                best = a
        return best

    async def reinit_account(self, account: Account, client: DsClient, solver: PowSolver) -> bool:
        """重新初始化失败的账号（改进：健康恢复）"""
        try:
            new_account = await self._try_init_account(
                account.creds, list(account.sessions.keys()), client, solver
            )
            account.token = new_account.token
            account.sessions = new_account.sessions
            account.reset_all_message_state(list(account.sessions.keys()))
            account.mark_healthy()
            logger.info(f"账号 {account.email or account.mobile} 重新初始化成功")
            self._cache_dirty = True
            await self._save_cache()
            return True
        except Exception as e:
            logger.warning(f"账号 {account.email or account.mobile} 重新初始化失败: {e}")
            return False

    def account_statuses(self) -> list[AccountStatus]:
        return [
            AccountStatus(
                email=a.email,
                mobile=a.mobile,
                busy=a.is_busy,
                healthy=a.is_healthy,
            )
            for a in self._accounts
        ]

    async def shutdown(self, client: DsClient) -> None:
        """优雅关闭：停止监控、保存缓存，不删除 session 以便下次复用"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        await self._save_cache(force=True)
        logger.info("账号缓存已保存")

    # ========================================================================
    # 后台健康监控
    # ========================================================================

    async def _health_monitor_loop(self) -> None:
        """后台健康监控：定期检查不健康账号并尝试恢复"""
        while True:
            await asyncio.sleep(HEALTH_MONITOR_INTERVAL)
            try:
                await self._check_and_recover()
            except Exception as e:
                logger.debug(f"健康监控异常: {e}")

    async def _check_and_recover(self) -> None:
        """检查不健康账号，尝试恢复"""
        if not self._client or not self._solver:
            return

        unhealthy = [a for a in self._accounts if not a.is_healthy and not a.is_busy]
        if not unhealthy:
            return

        logger.info(f"健康监控: 发现 {len(unhealthy)} 个不健康账号，尝试恢复")
        sem = asyncio.Semaphore(2)  # 恢复并发度低，避免限流

        async def _recover(acct: Account) -> None:
            async with sem:
                await asyncio.sleep(random.uniform(0.5, 2.0))
                try:
                    ok = await self.reinit_account(acct, self._client, self._solver)  # type: ignore
                    if ok:
                        logger.info(f"健康监控: 账号 {acct.email or acct.mobile} 恢复成功")
                except Exception as e:
                    logger.debug(f"健康监控: 账号 {acct.email or acct.mobile} 恢复失败: {e}")

        await asyncio.gather(*[_recover(a) for a in unhealthy], return_exceptions=True)

    # ========================================================================
    # 缓存管理
    # ========================================================================

    def _load_cache(self) -> dict | None:
        """从文件加载账号缓存"""
        if not self._cache_file.exists():
            return None
        try:
            data = json.loads(self._cache_file.read_text(encoding="utf-8"))
            ts = data.get("saved_at", 0)
            age_hours = (time.time() - ts) / 3600
            if age_hours > 24:
                logger.info(f"缓存已过期 ({age_hours:.1f}h > 24h)，跳过")
                return None
            logger.info(f"从 {self._cache_file} 加载缓存 (保存于 {age_hours:.1f}h 前)")
            return data.get("accounts", {})
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            return None

    async def _save_cache(self, force: bool = False) -> None:
        """保存账号状态到缓存文件（异步写入，不阻塞事件循环）"""
        # 检查是否需要保存
        now = time.time()
        if not force and not self._cache_dirty:
            return
        if not force and (now - self._last_cache_save_time) < self._cache_save_interval:
            # 距离上次保存时间不足间隔，标记为脏但不保存
            return

        # 先读取数据（不加锁），然后加锁写入
        cache_data = {
            "saved_at": time.time(),
            "accounts": {},
        }
        # 快速读取账号数据（在锁外）
        for account in self._accounts:
            key = account.email or account.mobile
            cache_data["accounts"][key] = {
                "token": account.token,
                "sessions": account.sessions,
                "parent_message_ids": account._parent_message_ids,
            }
        # 序列化在锁外完成（CPU密集）
        json_str = json.dumps(cache_data, ensure_ascii=False, indent=2)

        # 异步写入文件（不阻塞事件循环）
        async with self._cache_lock:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self._cache_file.write_text(json_str, encoding="utf-8"),
                )
                self._cache_dirty = False
                self._last_cache_save_time = now
            except Exception as e:
                logger.warning(f"保存缓存失败: {e}")

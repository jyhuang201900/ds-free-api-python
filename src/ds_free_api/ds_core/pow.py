"""PoW 计算器 —— 基于 DeepSeek WASM 的 DeepSeekHashV1 算法实现

通过 wasmtime Python 绑定加载并执行 WASM 模块。
动态探测 wasm-bindgen 导出符号，避免硬编码导致 WASM 更新后无法启动。
"""

from __future__ import annotations

import base64
import json
import logging
import struct
from typing import Any

import wasmtime

from .client import ChallengeData

logger = logging.getLogger("ds_free_api.ds_core.pow")


class PowError(Exception):
    """PoW 计算错误"""

    pass


class PowResult:
    """PoW 计算结果"""

    def __init__(
        self,
        algorithm: str,
        challenge: str,
        salt: str,
        answer: int,
        signature: str,
        target_path: str,
    ):
        self.algorithm = algorithm
        self.challenge = challenge
        self.salt = salt
        self.answer = answer
        self.signature = signature
        self.target_path = target_path

    def to_header(self) -> str:
        """将 PoW 结果转换为 base64 编码的 header"""
        payload = {
            "algorithm": self.algorithm,
            "challenge": self.challenge,
            "salt": self.salt,
            "answer": self.answer,
            "signature": self.signature,
            "target_path": self.target_path,
        }
        return base64.b64encode(json.dumps(payload).encode()).decode()


class PowSolver:
    """PoW 求解器（基于 wasmtime，实例池化）

    优化：预创建 Store+Instance 池，避免每次 solve 重新 instantiate。
    wasmtime Store 不可重入，但 solve 是同步阻塞调用且在线程池中执行，
    所以可以用 threading.Lock 保护实例池。
    """

    _POOL_SIZE = 4  # 实例池大小，匹配默认线程池大小

    def __init__(self, wasm_bytes: bytes):
        self._engine = wasmtime.Engine()
        self._module = wasmtime.Module(self._engine, wasm_bytes)
        self._linker = wasmtime.Linker(self._engine)

        # 动态探测导出符号
        exports = list(self._module.exports)

        self._add_to_stack_name = self._find_export_by_names(
            exports, ["__wbindgen_add_to_stack_pointer"],
            [wasmtime.ValType.i32()], [wasmtime.ValType.i32()],
        )
        if self._add_to_stack_name is None:
            raise PowError("__wbindgen_add_to_stack_pointer not found")

        # allocator: 优先 __wbindgen_malloc，其次签名匹配的 __wbindgen_export_*
        self._alloc_name = self._find_export_by_names(
            exports, ["__wbindgen_malloc"],
            [wasmtime.ValType.i32(), wasmtime.ValType.i32()], [wasmtime.ValType.i32()],
        )
        if self._alloc_name is None:
            self._alloc_name = self._find_export_by_prefix(
                exports, "__wbindgen_export_",
                [wasmtime.ValType.i32(), wasmtime.ValType.i32()], [wasmtime.ValType.i32()],
            )
        if self._alloc_name is None:
            raise PowError("allocator export not found")

        # wasm_solve: 优先显式名称，再按唯一签名探测
        solve_params = [
            wasmtime.ValType.i32(), wasmtime.ValType.i32(),
            wasmtime.ValType.i32(), wasmtime.ValType.i32(),
            wasmtime.ValType.i32(), wasmtime.ValType.f64(),
        ]
        self._solve_name = self._find_export_by_names(
            exports, ["wasm_solve"], solve_params, [],
        )
        if self._solve_name is None:
            # 按签名探测唯一匹配
            candidates = [
                e.name for e in exports
                if self._matches_sig(e, solve_params, [])
            ]
            if len(candidates) == 1:
                self._solve_name = candidates[0]
        if self._solve_name is None:
            raise PowError("wasm_solve export not found")

        # 实例池：预创建 Store+Instance，solve 时直接取用
        import threading
        from collections import deque
        self._pool_lock = threading.Lock()
        self._pool: deque[tuple[wasmtime.Store, wasmtime.Instance]] = deque(maxlen=self._POOL_SIZE)
        for _ in range(self._POOL_SIZE):
            store = wasmtime.Store(self._engine)
            instance = self._linker.instantiate(store, self._module)
            self._pool.append((store, instance))

    def solve(self, challenge: ChallengeData) -> PowResult:
        """求解 PoW challenge（使用实例池，避免重复 instantiate）"""
        if challenge.algorithm != "DeepSeekHashV1":
            raise PowError(f"unsupported algorithm: {challenge.algorithm}")

        # 从池中获取实例（LIFO：最近使用的实例内存更热）
        with self._pool_lock:
            if self._pool:
                store, instance = self._pool.pop()
            else:
                # 池空，临时创建
                store = wasmtime.Store(self._engine)
                instance = self._linker.instantiate(store, self._module)

        try:
            return self._solve_with_instance(store, instance, challenge)
        finally:
            # 归还实例到池中
            with self._pool_lock:
                if len(self._pool) < self._POOL_SIZE:
                    self._pool.append((store, instance))

    def _solve_with_instance(
        self,
        store: wasmtime.Store,
        instance: wasmtime.Instance,
        challenge: ChallengeData,
    ) -> PowResult:
        """使用给定 Store+Instance 执行 PoW 计算"""
        # wasmtime 44+ API: exports(store) 返回 InstanceExports
        exports = instance.exports(store)

        memory = exports.get("memory")
        if memory is None or not isinstance(memory, wasmtime.Memory):
            raise PowError("memory not found")

        add_to_stack = exports.get(self._add_to_stack_name)
        alloc = exports.get(self._alloc_name)
        wasm_solve = exports.get(self._solve_name)

        if add_to_stack is None or alloc is None or wasm_solve is None:
            raise PowError("required WASM functions not found")

        prefix = f"{challenge.salt}_{challenge.expire_at}_"

        # 分配栈空间（wasmtime 44: 单返回值直接返回 int，多返回值返回 list）
        retptr = add_to_stack(store, wasmtime.Val.i32(-16))
        if isinstance(retptr, list):
            retptr = retptr[0]

        # 写入字符串到 WASM 内存
        ptr_challenge, len_challenge = self._write_string(store, memory, alloc, challenge.challenge)
        ptr_prefix, len_prefix = self._write_string(store, memory, alloc, prefix)

        # 调用 wasm_solve
        wasm_solve(
            store,
            wasmtime.Val.i32(retptr),
            wasmtime.Val.i32(ptr_challenge),
            wasmtime.Val.i32(len_challenge),
            wasmtime.Val.i32(ptr_prefix),
            wasmtime.Val.i32(len_prefix),
            wasmtime.Val.f64(float(challenge.difficulty)),
        )

        # 读取结果（wasmtime 44: memory.read(store, start, stop) 是切片语义）
        status_buf = memory.read(store, retptr, retptr + 4)
        status = int.from_bytes(status_buf, "little")

        value_buf = memory.read(store, retptr + 8, retptr + 16)
        # WASM 返回 f64（双精度浮点），不是 i64
        value_f64 = struct.unpack("<d", bytes(value_buf))[0]
        value = int(value_f64)

        # 恢复栈
        add_to_stack(store, wasmtime.Val.i32(16))

        if status == 0:
            raise PowError("no solution found")

        return PowResult(
            algorithm=challenge.algorithm,
            challenge=challenge.challenge,
            salt=challenge.salt,
            answer=value,
            signature=challenge.signature,
            target_path=challenge.target_path,
        )

    def _write_string(self, store: wasmtime.Store, memory: wasmtime.Memory, alloc: Any, text: str) -> tuple[int, int]:
        """将字符串写入 WASM 内存"""
        data = text.encode("utf-8")
        length = len(data)
        ptr = alloc(store, wasmtime.Val.i32(length), wasmtime.Val.i32(1))
        if isinstance(ptr, list):
            ptr = ptr[0]
        memory.write(store, data, ptr)
        return ptr, length

    @staticmethod
    def _matches_sig(export: Any, params: list, results: list) -> bool:
        """检查导出函数签名是否匹配"""
        ty = export.type
        if not isinstance(ty, wasmtime.FuncType):
            return False
        ep = list(ty.params)
        er = list(ty.results)
        if len(ep) != len(params) or len(er) != len(results):
            return False
        return all(a == b for a, b in zip(ep, params)) and all(a == b for a, b in zip(er, results))

    @staticmethod
    def _find_export_by_names(exports: list, names: list[str], params: list, results: list) -> str | None:
        for name in names:
            for e in exports:
                if e.name == name and PowSolver._matches_sig(e, params, results):
                    return name
        return None

    @staticmethod
    def _find_export_by_prefix(exports: list, prefix: str, params: list, results: list) -> str | None:
        for e in exports:
            if e.name.startswith(prefix) and PowSolver._matches_sig(e, params, results):
                return e.name
        return None

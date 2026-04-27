"""配置加载模块 —— 统一配置入口（支持热重载、环境变量、动态更新）"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

import logging

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger("ds_free_api.config")


# ============================================================================
# 账号文件加载
# ============================================================================


def load_accounts_txt(filename: str, base_dir: Path | None = None) -> list[dict]:
    """从 TXT 文件加载账号列表（支持环境变量覆盖）"""
    accounts = []
    
    # 支持环境变量覆盖
    env_accounts = os.getenv("DS_ACCOUNTS")
    if env_accounts:
        # 环境变量格式：email1,pass1;email2,pass2
        for account_str in env_accounts.split(";"):
            parts = account_str.strip().split(",", 1)
            if len(parts) == 2:
                accounts.append({"email": parts[0], "password": parts[1]})
        logger.info(f"从环境变量加载了 {len(accounts)} 个账号")
        return accounts
    
    # 文件路径优先级：环境变量 > 命令行参数 > 默认路径
    if base_dir and filename:
        file_path = base_dir / filename
    elif filename:
        file_path = Path(filename)
    elif base_dir:
        file_path = base_dir / "accounts.txt"
    else:
        file_path = Path("accounts.txt")
    
    if not file_path.exists():
        logger.warning(f"账号文件不存在: {file_path}")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 支持注释（以 # 开头）
                if line.startswith("#"):
                    continue
                
                # 逗号分隔
                parts = line.split(",")
                if len(parts) == 2:
                    # email,password
                    accounts.append({"email": parts[0], "password": parts[1]})
                elif len(parts) == 3:
                    # mobile,area_code,password
                    accounts.append({"mobile": parts[0], "area_code": parts[1], "password": parts[2], "email": ""})
                else:
                    logger.warning(f"第 {line_num} 行格式错误: {line}")
                    continue
                
    except Exception as e:
        logger.error(f"读取账号文件失败: {e}")
        return []

    logger.info(f"从文件加载了 {len(accounts)} 个账号")
    return accounts


# ============================================================================
# 配置模型
# ============================================================================


class ApiToken(BaseModel):
    """API 访问令牌"""

    token: str
    description: str = ""


class ServerConfig(BaseModel):
    """HTTP 服务器配置"""

    host: str = "127.0.0.1"
    port: int = 5317
    api_tokens: list[ApiToken] = Field(default_factory=list)
    accounts_file: str = "accounts.txt"


class AccountConfig(BaseModel):
    """单个账号配置"""

    email: str = ""
    mobile: str = ""
    area_code: str = ""
    password: str

    @model_validator(mode="after")
    def validate_contact(self) -> AccountConfig:
        if not self.email and not self.mobile:
            raise ValueError("email 和 mobile 不能同时为空")
        return self


class DeepSeekConfig(BaseModel):
    """DeepSeek 客户端配置"""

    api_base: str = "https://chat.deepseek.com/api/v0"
    wasm_url: str = "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/134.0.0.0 Safari/537.36"
    )
    client_version: str = "1.8.0"
    client_platform: str = "web"
    model_types: list[str] = Field(default_factory=lambda: ["default", "expert"])
    max_input_tokens: list[int] = Field(default_factory=lambda: [1_048_576, 1_048_576])
    max_output_tokens: list[int] = Field(default_factory=lambda: [384_000, 384_000])

    @model_validator(mode="after")
    def validate_lengths(self) -> DeepSeekConfig:
        n = len(self.model_types)
        if n == 0:
            raise ValueError("model_types 不能为空")
        if len(self.max_input_tokens) != n:
            raise ValueError(f"max_input_tokens 长度({len(self.max_input_tokens)})必须与 model_types 长度({n})一致")
        if len(self.max_output_tokens) != n:
            raise ValueError(f"max_output_tokens 长度({len(self.max_output_tokens)})必须与 model_types 长度({n})一致")
        return self

    # 模型 ID 别名：让客户端能用官方 ID 访问
    MODEL_ALIASES: dict[str, list[str]] = {
        "default": ["deepseek-v4-flash"],
        "expert": ["deepseek-v4-pro"],
    }

    def model_registry(self) -> dict[str, str]:
        """生成 OpenAI 模型注册表映射: model_id -> model_type（含别名）"""
        registry = {}
        for ty in self.model_types:
            registry[f"deepseek-{ty}".lower()] = ty
            # 添加别名
            for alias in self.MODEL_ALIASES.get(ty, []):
                registry[alias.lower()] = ty
        return registry


class Config(BaseModel):
    """应用配置根结构"""

    accounts: list[AccountConfig] = Field(default_factory=list)
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @model_validator(mode="after")
    def validate_accounts(self) -> Config:
        if not self.accounts:
            raise ValueError("至少需要一个账号配置")
        return self

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """从指定路径加载配置"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        content = p.read_text(encoding="utf-8")
        data = tomllib.loads(content)

        # 从 accounts_file 加载账号（优先）
        server_cfg = data.get("server", {})
        accounts_file = server_cfg.get("accounts_file", "accounts.txt")
        accounts_from_file = load_accounts_txt(accounts_file, base_dir=p.parent)

        if accounts_from_file:
            data["accounts"] = accounts_from_file

        return cls.model_validate(data)

    @classmethod
    def load_with_args(cls, args: list[str] | None = None) -> Config:
        """解析命令行参数并加载配置"""
        if args is None:
            args = sys.argv[1:]

        config_path = "config.toml"
        i = 0
        while i < len(args):
            if args[i] == "-c":
                if i + 1 < len(args):
                    config_path = args[i + 1]
                    i += 2
                else:
                    raise SystemExit("-c 参数需要指定路径")
            else:
                i += 1

        return cls.load(config_path)

"""
诊断 login 返回 202 的根因：

测试 1: 用新 WAF token 登录之前 202 失败的账号 → 若成功 = WAF token login 次数限制
测试 2: 用新 WAF token 登录之前缓存成功的账号 → 若也 202 = IP/账号级别限制
测试 3: 等 5 分钟后再用旧 token 登录 → 若成功 = 临时频率限制
"""

from __future__ import annotations

import asyncio
import logging
import sys

import httpx

sys.path.insert(0, "src")

from ds_free_api.ds_core.waf_bypass import get_waf_cookies
from ds_free_api.config import DeepSeekConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_login_202")


cfg = DeepSeekConfig()
API_BASE = cfg.api_base

# 浏览器导航 headers
SIGNIN_HEADERS = {
    "User-Agent": cfg.user_agent,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

# API headers
LOGIN_HEADERS = {
    "User-Agent": cfg.user_agent,
    "X-Client-Version": cfg.client_version,
    "X-Client-Platform": cfg.client_platform,
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
}


def make_payload(email: str, password: str) -> dict:
    return {
        "email": email,
        "password": password,
        "device_id": "B" + "test-device-id-fixed",
        "os": "web",
    }


async def try_login(http: httpx.AsyncClient, email: str, password: str) -> tuple[int, str]:
    """返回 (status_code, body_preview)"""
    resp = await http.post(
        f"{API_BASE}/users/login",
        headers=LOGIN_HEADERS,
        json=make_payload(email, password),
    )
    body = resp.text[:300]
    return resp.status_code, body


async def test_with_fresh_token(email: str, password: str):
    """用全新 WAF token 测试 login"""
    async with httpx.AsyncClient(follow_redirects=True) as http:
        # 获取新 WAF token
        logger.info("获取全新 WAF token...")
        raw_cookies = await get_waf_cookies(user_agent=cfg.user_agent)
        for c in raw_cookies:
            domain = c.get("domain", "chat.deepseek.com")
            http.cookies.set(c["name"], c["value"], domain=domain, path=c.get("path", "/"))

        # 验证 sign_in
        r = await http.get("https://chat.deepseek.com/sign_in", headers=SIGNIN_HEADERS)
        logger.info(f"sign_in with fresh token: {r.status_code}")

        # 尝试 login
        status, body = await try_login(http, email, password)
        logger.info(f"login with fresh token: {status}, body={body[:200]}")
        return status == 200


async def test_with_no_token(email: str, password: str):
    """不用 WAF token 直接 login（预期 202）"""
    async with httpx.AsyncClient(follow_redirects=True) as http:
        status, body = await try_login(http, email, password)
        logger.info(f"login without WAF token: {status}, body={body[:200]}")
        return status


def load_accounts_from_file() -> list[tuple[str, str]]:
    """从 accounts.txt 加载所有账号"""
    accounts: list[tuple[str, str]] = []
    try:
        with open("accounts.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",", 2)
                if len(parts) >= 2:
                    email_or_mobile = parts[0]
                    password = parts[-1]
                    accounts.append((email_or_mobile, password))
    except FileNotFoundError:
        pass
    return accounts


async def main():
    # 自动从 accounts.txt 加载第 8 个之后的账号
    all_accounts = load_accounts_from_file()
    if len(all_accounts) < 8:
        print(f"⚠️ accounts.txt 只有 {len(all_accounts)} 个账号，需要至少 8 个")
        print("前 7 个账号从缓存恢复，需要测试第 8 个之后的账号")
        sys.exit(1)

    # 取第 8 个账号（索引 7）
    email, password = all_accounts[7]
    logger.info(f"测试账号: {email}")

    print("\n=== 测试 A: 无 WAF token 直接 login ===")
    await test_with_no_token(email, password)

    print("\n=== 测试 B: 用全新 WAF token login ===")
    ok = await test_with_fresh_token(email, password)

    print("\n" + "=" * 40)
    if ok:
        print("✅ 全新 WAF token 下 login 成功！")
        print("   结论: WAF token 有 login 次数上限（约 7 次），超限后返回 202")
        print("   修复方向: 每 7 个账号刷新一次 WAF token")
    else:
        print("❌ 即使全新 WAF token 也 202")
        print("   结论: 可能是 IP 频率限制或账号级别风控")
        print("   修复方向: 增加更长冷却间隔，或检查账号状态")


if __name__ == "__main__":
    asyncio.run(main())

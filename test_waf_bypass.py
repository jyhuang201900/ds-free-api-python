"""
WAF Bypass 独立验证脚本（不依赖主代码 DsClient / waf_bypass）

修复点：
1. Playwright 使用 headless=False + 反检测参数（headless=True 会被 WAF bot 检测过滤）
2. GET sign_in 不使用 Content-Type: application/json（会导致 405）
3. 打印完整 cookie 元信息用于诊断域名/路径问题

用法:
    python test_waf_bypass.py
    python test_waf_bypass.py --login email@example.com password
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_waf_bypass")

# DeepSeek 配置（硬编码，避免依赖主代码）
API_BASE = "https://chat.deepseek.com/api/v0"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36"
)
CLIENT_VERSION = "2.0.0"
CLIENT_PLATFORM = "web"


# ============================================================================
# Playwright 内联 bypass（headless=False + anti-detection）
# ============================================================================

async def _playwright_bypass() -> list[dict]:
    """返回 Playwright 原始 cookie dict 列表（含 domain/path 等元信息）"""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,  # 必须用有界面窗口，headless 会被 WAF 检测
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            user_agent=USER_AGENT,
        )
        page = await context.new_page()

        # 注入反检测脚本
        await page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            """
        )

        try:
            logger.info("Playwright 访问 sign_in ...")
            resp = await page.goto(
                "https://chat.deepseek.com/sign_in",
                wait_until="networkidle",
                timeout=30000,
            )
            logger.info(f"第一次 sign_in status={resp.status if resp else 'None'}")

            # 若被 WAF 202 拦截，等待 JS challenge 执行
            if resp is not None and resp.status == 202:
                logger.info("WAF 202 拦截，等待 JS challenge 执行 8s...")
                await asyncio.sleep(8)
                resp = await page.goto(
                    "https://chat.deepseek.com/sign_in",
                    wait_until="networkidle",
                    timeout=30000,
                )
                logger.info(f"第二次 sign_in status={resp.status if resp else 'None'}")

            if resp is not None and resp.status == 202:
                logger.info("仍 202，再等待 5s 后刷新...")
                await asyncio.sleep(5)
                resp = await page.reload(wait_until="networkidle", timeout=30000)
                logger.info(f"刷新后 sign_in status={resp.status if resp else 'None'}")

            # 拿原始 cookies（含 domain / path / httpOnly 等）
            raw_cookies = await context.cookies()
            logger.info(f"Playwright 拿到 {len(raw_cookies)} 个 cookies")
            for c in raw_cookies:
                domain = c.get("domain", "")
                name = c.get("name", "")
                val = c.get("value", "")[:60]
                logger.info(f"  cookie: domain={domain} name={name} value={val}...")

            return raw_cookies
        finally:
            await browser.close()


def _cookies_for_httpx(raw_cookies: list[dict], target_domain: str = "chat.deepseek.com") -> dict[str, str]:
    """把 Playwright 原始 cookies 过滤为 httpx 可用的 domain->name->value dict"""
    result: dict[str, str] = {}
    for c in raw_cookies:
        domain = c.get("domain", "")
        # 匹配目标域名及其父域
        if target_domain in domain or domain in target_domain:
            result[c["name"]] = c["value"]
    return result


# ============================================================================
# httpx 探测（GET 不带 Content-Type）
# ============================================================================

SIGNIN_HEADERS = {
    "User-Agent": USER_AGENT,
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


def _login_headers(token: str | None = None) -> dict[str, str]:
    """POST /users/login 专用 headers（带 Content-Type 和 Origin/Referer）"""
    h = {
        "User-Agent": USER_AGENT,
        "X-Client-Version": CLIENT_VERSION,
        "X-Client-Platform": CLIENT_PLATFORM,
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
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


# ============================================================================
# 测试流程
# ============================================================================

async def test_cookies_only():
    """测试：Playwright 获取 cookies → 注入 httpx → GET sign_in 返回 200"""
    async with httpx.AsyncClient(follow_redirects=True) as http:
        # 阶段 A: 裸请求 → 预期 202
        print("\n=== 阶段 A: 未注入 WAF cookies ===")
        resp_a = await http.get("https://chat.deepseek.com/sign_in", headers=SIGNIN_HEADERS)
        print(f"  GET sign_in status: {resp_a.status_code}")
        if resp_a.status_code == 202:
            print("  ✅ 符合预期：裸请求返回 202（被 WAF 拦截）")
        else:
            print(f"  ⚠️ 意外状态码: {resp_a.status_code}")

        # 阶段 B: Playwright 获取 cookies
        print("\n=== 阶段 B: Playwright 获取 WAF cookies (headless=False) ===")
        raw_cookies = await _playwright_bypass()
        cookie_dict = _cookies_for_httpx(raw_cookies)
        print(f"  过滤后可用 cookies ({len(cookie_dict)} 个):")
        for k, v in cookie_dict.items():
            print(f"    {k}: {v[:80]}{'...' if len(v) > 80 else ''}")

        if "aws-waf-token" not in cookie_dict:
            print("  ❌ 未获取到 aws-waf-token，bypass 失败")
            return False
        print("  ✅ aws-waf-token 已获取")

        # 阶段 C: 注入 cookies 后再 GET sign_in（预期 200）
        print("\n=== 阶段 C: 注入 cookies 后再次 GET sign_in ===")
        for name, value in cookie_dict.items():
            http.cookies.set(name, value, domain="chat.deepseek.com", path="/")

        resp_c = await http.get("https://chat.deepseek.com/sign_in", headers=SIGNIN_HEADERS)
        print(f"  GET sign_in status: {resp_c.status_code}")
        if resp_c.status_code == 200:
            print("  ✅ WAF bypass 成功！sign_in 返回 200")
            return True
        elif resp_c.status_code == 202:
            print("  ❌ 仍返回 202，cookies 未生效")
            return False
        else:
            print(f"  ⚠️ 意外状态码: {resp_c.status_code}，body={resp_c.text[:200]}")
            return False


async def test_full_login(email: str, password: str):
    """测试完整登录流程"""
    async with httpx.AsyncClient(follow_redirects=True) as http:
        # 先 bypass WAF
        print("\n=== 先执行 WAF bypass ===")
        raw_cookies = await _playwright_bypass()
        cookie_dict = _cookies_for_httpx(raw_cookies)
        for name, value in cookie_dict.items():
            http.cookies.set(name, value, domain="chat.deepseek.com", path="/")
        print(f"  已注入 {len(cookie_dict)} 个 cookies")

        # 预热 sign_in（可选，建立 session cookie）
        print("\n=== 预热 sign_in ===")
        await http.get("https://chat.deepseek.com/sign_in", headers=SIGNIN_HEADERS)

        # POST login
        print("\n=== POST /users/login ===")
        device_id = "B" + "test-device-id-123456"  # 简化，真实应随机
        payload = {
            "email": email,
            "password": password,
            "device_id": device_id,
            "os": "web",
        }
        resp = await http.post(
            f"{API_BASE}/users/login",
            headers=_login_headers(),
            json=payload,
        )
        print(f"  status: {resp.status_code}")
        print(f"  body: {resp.text[:500]}")
        if resp.status_code == 200:
            try:
                data = resp.json()
                code = data.get("code", -1)
                if code == 0:
                    inner = data.get("data", {})
                    biz_data = inner.get("biz_data", {})
                    user = biz_data.get("user", {})
                    token = user.get("token", "")
                    print(f"  ✅ 登录成功！token={token[:10]}...")
                    return True
                else:
                    print(f"  ❌ 业务错误: code={code}, msg={data.get('msg', '')}")
                    return False
            except Exception as e:
                print(f"  ❌ 解析响应失败: {e}")
                return False
        else:
            print(f"  ❌ HTTP 错误: {resp.status_code}")
            return False


async def main():
    parser = argparse.ArgumentParser(description="WAF Bypass 独立验证")
    parser.add_argument("--login", nargs=2, metavar=("EMAIL", "PASSWORD"), help="测试完整登录")
    args = parser.parse_args()

    if args.login:
        ok = await test_full_login(args.login[0], args.login[1])
    else:
        ok = await test_cookies_only()

    print("\n" + "=" * 40)
    if ok:
        print("🎉 测试通过！现在可以安全修改主代码了。")
        sys.exit(0)
    else:
        print("💥 测试失败，请根据日志排查后再修改主代码。")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

"""AWS WAF JS Challenge 绕过 —— 使用 Playwright 获取有效 cookies

DeepSeek 前置 AWS WAF Bot Control，首次访问返回 202 + JS Challenge。
httpx 无法执行 JavaScript，因此需要 Playwright 真实浏览器完成挑战后，
提取 cookies（尤其是 aws-waf-token）注入 httpx client。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("ds_free_api.ds_core.waf_bypass")


async def get_waf_cookies(
    user_agent: str | None = None,
    proxy: str | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> list[dict]:
    """使用 Playwright 访问 DeepSeek sign_in 页面，通过 AWS WAF JS Challenge。

    返回 Playwright 原始 cookie 列表（含 domain / path / httpOnly 等元信息），
    供上层使用原始 domain 注入 httpx CookieJar。
    核心 cookie: aws-waf-token (domain=.deepseek.com), ds_session_id。

    失败时自动重试最多 max_retries 次。
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:
        raise ImportError("未安装 playwright，请运行: pip install playwright && playwright install chromium") from e

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        browser = None
        try:
            async with async_playwright() as p:
                launch_opts: dict[str, Any] = {
                    "headless": False,  # headless=True 会被 AWS WAF bot 检测拦截
                    "args": [
                        "--disable-blink-features=AutomationControlled",
                        "--disable-web-security",
                        "--disable-features=IsolateOrigins,site-per-process",
                    ],
                }
                if proxy:
                    launch_opts["proxy"] = {"server": proxy}

                browser = await p.chromium.launch(**launch_opts)

                context_opts: dict[str, Any] = {
                    "viewport": {"width": 1920, "height": 1080},
                    "locale": "zh-CN",
                    "timezone_id": "Asia/Shanghai",
                }
                if user_agent:
                    context_opts["user_agent"] = user_agent

                context = await browser.new_context(**context_opts)
                page = await context.new_page()

                # 注入反检测脚本，隐藏 navigator.webdriver
                await page.add_init_script(
                    """
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    window.chrome = { runtime: {} };
                    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                    """
                )

                logger.info(f"启动 Playwright 获取 AWS WAF cookies (headless=False, attempt={attempt})...")
                # 使用 domcontentloaded 避免 networkidle 在长连接页面超时
                resp = await page.goto(
                    "https://chat.deepseek.com/sign_in",
                    wait_until="domcontentloaded",
                    timeout=int(timeout * 1000),
                )

                # 等待页面稳定 + WAF JS 执行
                await asyncio.sleep(5)

                # 检查是否仍被 WAF 拦截（页面内容包含 challenge）
                content = await page.content()
                if "challenge" in content.lower() or (resp and resp.status == 202):
                    logger.info("AWS WAF challenge 检测中，等待 JS 执行 8s...")
                    await asyncio.sleep(8)

                cookies = await context.cookies()
                names = {c["name"] for c in cookies}

                if "aws-waf-token" not in names:
                    logger.warning(f"未获取到 aws-waf-token (cookies={list(names)})")
                else:
                    logger.info("AWS WAF cookies 获取成功 (aws-waf-token 已拿到)")

                return cookies

        except Exception as e:
            last_error = e
            logger.error(f"Playwright WAF bypass 失败 (attempt={attempt}): {e}")
            if attempt < max_retries:
                logger.info(f"等待 3s 后重试...")
                await asyncio.sleep(3)
        finally:
            if browser:
                try:
                    await browser.close()
                except Exception:
                    pass

    raise last_error or RuntimeError("Playwright WAF bypass 失败，原因未知")


async def test_bypass():
    """快速测试 bypass 功能"""
    logging.basicConfig(level=logging.INFO)
    cookies = await get_waf_cookies()
    print(f"拿到 {len(cookies)} 个 cookies:")
    for c in cookies:
        print(f"  {c['name']}: domain={c.get('domain')} path={c.get('path')} {c['value'][:80]}{'...' if len(c['value']) > 80 else ''}")


if __name__ == "__main__":
    asyncio.run(test_bypass())

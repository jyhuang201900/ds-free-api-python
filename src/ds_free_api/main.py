"""DS-Free-API 入口"""

from __future__ import annotations

import logging
import multiprocessing
import sys

from .config import Config


def main() -> None:
    """主入口函数"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        config = Config.load_with_args()
    except Exception as e:
        logging.fatal(f"配置加载失败: {e}")
        sys.exit(1)

    logging.info(f"配置加载成功: host={config.server.host}, port={config.server.port}")
    logging.info(f"账号数: {len(config.accounts)}, 模型类型: {config.deepseek.model_types}")

    import uvicorn
    from .server.app import create_app

    app = create_app(config)

    # 高并发配置
    # 单进程模式（共享账号池状态），通过 backlog 和 timeout 支持高并发
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="info",
        backlog=2048,  # 增大连接队列，支持突发流量
        timeout_keep_alive=30,  # 保持连接 30 秒
        h11_max_incomplete_event_size=None,  # 不限制请求大小
    )


if __name__ == "__main__":
    main()

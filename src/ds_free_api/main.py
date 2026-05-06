"""DS-Free-API 入口"""

from __future__ import annotations

import logging
import multiprocessing
import sys

from .config import Config


def main() -> None:
    """主入口函数"""
    # 先加载配置获取日志级别
    try:
        config = Config.load_with_args()
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logging.fatal(f"配置加载失败: {e}")
        sys.exit(1)

    # 配置日志级别
    log_level = getattr(logging, config.server.log_level.upper(), logging.WARNING)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # 抑制第三方库日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.warning(f"服务启动: host={config.server.host}, port={config.server.port}, log_level={config.server.log_level}")

    import uvicorn
    from .server.app import create_app

    app = create_app(config)

    try:
        # 高并发配置
        uvicorn.run(
            app,
            host=config.server.host,
            port=config.server.port,
            log_level=config.server.log_level.lower(),
            backlog=2048,
            timeout_keep_alive=30,
            h11_max_incomplete_event_size=None,
            access_log=False,  # 关闭访问日志
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.exception(f"服务异常退出: {e}")


if __name__ == "__main__":
    main()

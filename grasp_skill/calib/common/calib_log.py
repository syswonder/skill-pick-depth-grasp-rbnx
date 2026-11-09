"""标定台统一日志：一条函数，一个文件。

所有步骤（steps/、common/、vision/）都从这儿拿 logger，不各自 print。
日志写到 artifacts/logs/calibration_YYYYMMDD.log，行尾带 时间、级别、模块、消息。
需要排查"为什么这张没被采纳""为什么卡住"时，按时间戳对着看就行。
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

_LOGGER: logging.Logger | None = None
_LOGGED_PATH: Path | None = None


def setup_logging(root: Path, level: int = logging.DEBUG) -> logging.Logger:
    """初始化一次：日志写到 root/logs/ 下，控制台也打一份。幂等。"""
    global _LOGGER, _LOGGED_PATH
    if _LOGGER is not None:
        return _LOGGER
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _LOGGED_PATH = log_dir / ("calibration_%s.log" % datetime.now().strftime("%Y%m%d"))
    logger = logging.getLogger("calibration")
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-24s %(message)s",
                                  datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(_LOGGED_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    _LOGGER = logger
    return logger


def get_logger(module: str) -> logging.Logger:
    """取一个命名的 logger（第一次用时自动初始化）。"""
    if _LOGGER is None:
        setup_logging(Path(__file__).resolve().parent.parent / "artifacts")
    return logging.getLogger("calibration." + module)


def log_path() -> Path | None:
    return _LOGGED_PATH

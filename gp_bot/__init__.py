"""None."""
import logging
from logging import Logger


def create_logger() -> Logger:
    """ロガーの生成.

    Returns
    -------
    Logger
        ロガー

    """
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] in %(filename)s: %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger

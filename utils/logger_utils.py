# utils/logger_utils.py
import logging
import colorlog
import os
import sys
from datetime import datetime

def get_logger(name=__name__, level=logging.INFO, log_file=None):
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Prevent duplicate handlers

    logger.setLevel(level)

    # Extract filename (instead of module path)
    filename = os.path.basename(sys.modules[name].__file__) if name in sys.modules and hasattr(sys.modules[name], '__file__') else name

    # Colored formatter for console
    formatter = colorlog.ColoredFormatter(
        fmt=f"%(log_color)s[%(asctime)s] [{filename}] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional: File handler (no color)
    if log_file:
        file_formatter = logging.Formatter(
            fmt=f"[%(asctime)s] [{filename}] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler(log_file)
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger
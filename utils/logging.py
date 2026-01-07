"""
Logging configuration for the application.

This module sets up both Loguru (for modern logging) and bridges it with
Python's standard logging module (used by many packages).
"""

import sys
import logging
from loguru import logger
from config.settings import settings


# Remove default loguru handler and add customized ones
logger.remove()

# Add console handler with color support
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL,
    colorize=True,
)

# Add file handler for persistent logs
logger.add(
    "app.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",  # Always log DEBUG to file
    enqueue=True,  # Thread-safe
)


class InterceptHandler(logging.Handler):
    """
    Handler that intercepts standard logging calls and redirects them to Loguru.
    
    This allows packages using standard logging.getLogger() to have their
    logs appear in Loguru's output.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    """
    Configure Python's standard logging to work with Loguru.
    
    This intercepts all standard logging calls and redirects them to Loguru,
    ensuring consistent logging across the entire application.
    """
    # Get the root logger
    logging.root.setLevel(logging.NOTSET)
    
    # Remove all existing handlers from root logger
    logging.root.handlers = [InterceptHandler()]
    
    # Set level for all existing loggers
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    
    # Configure specific third-party loggers to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.INFO)
    logging.getLogger("langchain").setLevel(logging.INFO)
    logging.getLogger("langchain_core").setLevel(logging.INFO)
    logging.getLogger("langchain_community").setLevel(logging.INFO)
    
    logger.info(f"Logging configured with level: {settings.LOG_LEVEL}")


# Setup logging when module is imported
setup_logging()
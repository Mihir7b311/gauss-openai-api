"""
Logging configuration
"""

import logging
import sys
from typing import Optional

from app.config.settings import get_settings


def setup_logging(level: Optional[str] = None) -> None:
    """
    Setup application logging configuration
    
    Args:
        level: Log level override (uses settings if not provided)
    """
    settings = get_settings()
    
    # Determine log level
    log_level = level or settings.log_level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=settings.log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    loggers = {
        'uvicorn': logging.INFO,
        'uvicorn.access': logging.INFO,
        'fastapi': logging.INFO,
        'httpx': logging.WARNING,
        'requests': logging.WARNING,
        'urllib3': logging.WARNING,
        'app': numeric_level,
    }
    
    for logger_name, logger_level in loggers.items():
        logging.getLogger(logger_name).setLevel(logger_level)
    
    # Create app-specific logger
    app_logger = logging.getLogger('app')
    app_logger.info(f"Logging configured with level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"app.{name}")
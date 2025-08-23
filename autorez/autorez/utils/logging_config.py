"""AutoResolve Central Logging Configuration"""

import logging
import logging.config
import os
from pathlib import Path

# Create logs directory
LOG_DIR = Path.home() / "Library/Logs/AutoResolve"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(LOG_DIR / 'app.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': str(LOG_DIR / 'error.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
    },
    'loggers': {
        'autoresolve': {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'handlers': ['console', 'file', 'error_file'],
            'propagate': False
        },
        'autoresolve.pipeline': {
            'level': 'DEBUG',
            'handlers': ['file'],
            'propagate': False
        },
        'autoresolve.backend': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
            'propagate': False
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}

def setup_logging():
    """Initialize logging for AutoResolve"""
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger('autoresolve')
    logger.info("AutoResolve logging initialized")
    return logger

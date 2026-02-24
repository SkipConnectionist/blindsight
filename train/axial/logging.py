import logging
import sys
import os
import socket

from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color codes for terminal output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        # Get the original formatted message
        formatted = super().format(record)

        # Add color if the level exists in our color mapping
        level_name = record.levelname
        if level_name in self.COLORS:
            color = self.COLORS[level_name]
            formatted = f"{color}{formatted}{self.RESET}"

        return formatted


def setup_logging(
    level: int = logging.INFO,
    log_directory: str | None = None,
    file_prefix: str = "log",
    console_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    file_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    date_format: str = '%Y-%m-%d %H:%M:%S'
) -> str | None:
    """
    Set up colored logging to console and optionally to file.
    This modifies the root logger and affects all subsequent logging calls.

    Args:
        level: Logging level (default: logging.INFO)
        log_directory: Directory for log file. If None, no file logging.
        file_prefix: Prefix for log filename (default: "log")
        console_format: Format string for console output
        file_format: Format string for file output
        date_format: Date format string

    Returns:
        Path to log file if file logging is enabled, None otherwise
    """
    # Get the root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    # Set up console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        fmt=console_format,
        datefmt=date_format
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    log_file_path = None

    # Set up file handler if directory is provided
    if log_directory:
        # Create directory if it doesn't exist
        os.makedirs(log_directory, exist_ok=True)

        # Generate filename with prefix, timestamp, and hostname
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hostname = socket.gethostname()
        filename = f"{file_prefix}_{timestamp}_{hostname}.log"
        log_file_path = os.path.join(log_directory, filename)

        # Set up file handler without colors
        file_handler = logging.FileHandler(log_file_path)
        file_formatter = logging.Formatter(
            fmt=file_format,
            datefmt=date_format
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return log_file_path

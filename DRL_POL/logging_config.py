import os
from typing import Dict, Any
import logging

## This file sets the logging levels for the project
# IN GENERAL: logging messages should use %s formatting, not f-strings

# Possible logging levels (for reference)
possible_logging_levels = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

# Set the DEFAULT logging level, passed to individual modules

DEFAULT_LOGGING_LEVEL = "INFO"

# Dictionary to define logging levels and bool for each module

logging_config_dict: Dict[str, Dict[str, Any]] = {
    "PARALLEL_TRAINING_3D_CHANNEL_MARL": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "Env3D_MARL_channel": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "parameters": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "alya": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "coco_calc_avg_vel": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "coco_calc_reward": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "cr": {
        "console_level": "WARNING",
        "file_level": "WARNING",
        "override": True,
    },
    "env_utils": {
        "console_level": "WARNING",
        "file_level": "WARNING",
        "override": True,
    },
    "extract_forces": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "jets": {
        "console_level": "DEBUG",
        "file_level": "DEBUG",
        "override": False,
    },
    "witness": {
        "console_level": "WARNING",
        "file_level": "WARNING",
        "override": True,
    },
}


# TODO: @pietero use `funcName` to automatically add the function name to the log - Pieter
def configure_logger(module_name: str, default_level: str = "INFO") -> logging.Logger:
    """
    Configure a logger with specified settings or defaults.

    Args:
        module_name (str): The name of the module for the logger.
        default_level (str): The default logging level (default is "INFO").

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(module_name)

    if module_name in logging_config_dict:
        config = logging_config_dict[module_name]
        if config.get("override", False):
            console_level = config["console_level"].upper()
            file_level = config["file_level"].upper()
        else:
            console_level = default_level.upper()
            file_level = default_level.upper()
    else:
        console_level = default_level.upper()
        file_level = default_level.upper()

    # Clear existing handlers if override is specified
    if config.get("override", False):
        logger.handlers = []

    if not logger.hasHandlers():
        logger.setLevel(
            min(
                possible_logging_levels[console_level],
                possible_logging_levels[file_level],
            )
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        formatter_ch = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter_ch)
        logger.addHandler(ch)

        # File handler
        if not os.path.exists("pythonlogs"):
            os.makedirs("pythonlogs")
        fh = logging.FileHandler(f"pythonlogs/{module_name}.log")
        fh.setLevel(file_level)
        formatter_fh = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter_fh)
        logger.addHandler(fh)

    # Disable propagation to prevent duplicate logs
    logger.propagate = False

    return logger

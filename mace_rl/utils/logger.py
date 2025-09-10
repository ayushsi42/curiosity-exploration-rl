import logging
import os
import torch
import numpy as np
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO, console_output=True):
    """Set up a logger with both file and console handlers.
    
    Args:
        name (str): Name of the logger
        log_file (str, optional): Path to log file. If None, creates a timestamped file in logs directory
        level (int): Logging level
        console_output (bool): Whether to also output to console
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Only create file handler if explicitly provided log_file (for experiment suite)
    if log_file is not None:
        # Create directory for log file if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Console handler (only for experiment suite logger)
    if console_output and log_file is not None:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

# Create default loggers for different components
def get_tensor_info(tensor):
    """Get a formatted string with tensor information."""
    if torch.is_tensor(tensor):
        return f"Tensor(shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device})"
    elif isinstance(tensor, np.ndarray):
        return f"ndarray(shape={tensor.shape}, dtype={tensor.dtype})"
    else:
        return f"Type: {type(tensor)}"

def get_state_info(state):
    """Get a formatted string with state information."""
    if isinstance(state, tuple):
        return f"Tuple(len={len(state)}, elements={[get_tensor_info(s) for s in state]})"
    else:
        return get_tensor_info(state)

def get_logger(name, log_file=None):
    """Get or create a logger for a specific component - DISABLED for performance."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.CRITICAL + 1)  # Disable all logging
    logger.handlers = []  # Remove all handlers
    logger.propagate = False  # Don't propagate to parent loggers
    return logger

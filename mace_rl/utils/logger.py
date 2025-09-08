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
    # Create logs directory if it doesn't exist
    if log_file is None:
        logs_dir = os.path.join(os.getcwd(), 'logs', 'debug_logs')
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(logs_dir, f'{name}_{timestamp}.log')

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    if console_output:
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
    """Get or create a logger for a specific component."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # Only setup if logger doesn't exist
        logger = setup_logger(name, log_file)
    return logger

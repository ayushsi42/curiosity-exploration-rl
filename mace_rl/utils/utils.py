import torch
import numpy as np
import random
import time
import csv

def log_data(log_file, data):
    """Logs data to a CSV file."""
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(data.keys())
        writer.writerow(data.values())

def profile(func):
    """A decorator to profile a function's execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run.")
        return result
    return wrapper

def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


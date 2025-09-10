import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mace_rl.utils.logger import get_logger

logger = get_logger('EpisodicMemory')

class EpisodicMemory(nn.Module):
    """
    Differentiable Neural Dictionary for Episodic Memory.
    """
    def __init__(self, capacity, key_dim, lru=True):
        super(EpisodicMemory, self).__init__()
        self.capacity = capacity
        self.key_dim = key_dim
        self.lru = lru
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing EpisodicMemory with capacity {capacity} and key_dim {key_dim} on device {self.device}")
        
        # Initialize storage - keep keys on CPU for memory efficiency, move to GPU when needed
        self.keys = torch.zeros(capacity, key_dim)
        self.values = [None] * capacity  # Pre-allocate value slots
        self.usage = torch.zeros(capacity)
        self.pointer = 0
        self.size = 0

    def add(self, key, value):
        """Adds a key-value pair to the memory."""
        try:
            # Ensure key is the correct dimension
            if key.dim() == 1:
                if key.shape[0] != self.key_dim:
                    logger.error(f"Key dimension mismatch. Expected {self.key_dim}, got {key.shape[0]}")
                    raise ValueError(f"Key dimension mismatch. Expected {self.key_dim}, got {key.shape[0]}")
            else:
                key = key.view(-1)  # Flatten if multi-dimensional
                if key.shape[0] != self.key_dim:
                    logger.error(f"Flattened key dimension mismatch. Expected {self.key_dim}, got {key.shape[0]}")
                    raise ValueError(f"Flattened key dimension mismatch. Expected {self.key_dim}, got {key.shape[0]}")

            logger.debug(f"Adding to memory - Key shape: {key.shape}, Value shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")

            if self.size < self.capacity:
                self.keys[self.pointer] = key
                self.values[self.pointer] = value
                self.usage[self.pointer] = 1
                self.pointer = (self.pointer + 1) % self.capacity
                self.size += 1
                logger.debug(f"Added to memory at position {self.pointer-1}, current size: {self.size}")
            else:
                logger.debug("Memory at capacity, implementing replacement strategy")
                if self.lru:
                    # Find least recently used position
                    replace_idx = torch.argmin(self.usage).item()
                else:
                    # Random replacement
                    replace_idx = self.pointer
                    self.pointer = (self.pointer + 1) % self.capacity
                
                self.keys[replace_idx] = key
                self.values[replace_idx] = value
                self.usage[replace_idx] = 1
        except Exception as e:
            logger.error(f"Error adding to memory: {e}")
            raise
        else:
            if self.lru:
                # Replace least recently used
                lru_index = torch.argmin(self.usage)
                self.keys[lru_index] = key
                self.values[lru_index] = value
                self.usage[lru_index] = torch.max(self.usage) + 1
            else:
                # FIFO
                self.keys[self.pointer] = key
                self.values[self.pointer] = value
                self.usage[self.pointer] = 1
                self.pointer = (self.pointer + 1) % self.capacity


    def query(self, query_key, k=1):
        """Returns the values associated with the k nearest keys to the query key."""
        try:
            # Handle empty memory case
            if self.size == 0:
                logger.warning("Memory is empty, returning None")
                return None, None

            # Ensure query key is the correct dimension
            if query_key.dim() > 1:
                query_key = query_key.view(-1)  # Flatten if multi-dimensional
            
            if query_key.shape[0] != self.key_dim:
                logger.error(f"Query key dimension mismatch. Expected {self.key_dim}, got {query_key.shape[0]}")
                raise ValueError(f"Query key dimension mismatch. Expected {self.key_dim}, got {query_key.shape[0]}")

            logger.debug(f"Querying memory with key shape: {query_key.shape}, k={k}")

            # Compute distances to all keys
            distances = torch.norm(self.keys[:self.size] - query_key.unsqueeze(0), dim=1)
            
            # Get indices of k nearest neighbors
            _, indices = torch.topk(distances, min(k, self.size), largest=False)
            
            # Update usage for retrieved items
            self.usage[indices] += 1

            logger.debug(f"Retrieved {len(indices)} values from memory")
            values = [self.values[i] for i in indices]
            
            # Additional debug logging for retrieved values
            logger.debug(f"Retrieved value shapes: {[v.shape if hasattr(v, 'shape') else 'scalar' for v in values]}")
            
            return values, indices

        except Exception as e:
            logger.error(f"Error during memory query: {str(e)}")
            return None, None

    def forward(self, query_key, k=1):
        return self.retrieve(query_key, k)

if __name__ == '__main__':
    # Example usage
    memory = EpisodicMemory(capacity=1000, key_dim=64, value_dim=10)
    
    # Add some memories
    for i in range(10):
        key = torch.randn(64)
        value = torch.randn(10)
        memory.add(key, value)

    # Retrieve a memory
    query = torch.randn(64)
    retrieved_value, score = memory.retrieve(query)
    print("Retrieved value:", retrieved_value)
    print("Score:", score)
    print("Memory size:", memory.size)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EpisodicMemory(nn.Module):
    """
    Differentiable Neural Dictionary for Episodic Memory.
    """
    def __init__(self, capacity, key_dim, lru=True):
        super(EpisodicMemory, self).__init__()
        self.capacity = capacity
        self.key_dim = key_dim
        self.lru = lru

        self.keys = torch.zeros(capacity, key_dim)
        self.values = []
        self.usage = torch.zeros(capacity)
        self.pointer = 0
        self.size = 0

    def add(self, key, value):
        """Adds a key-value pair to the memory."""
        if self.size < self.capacity:
            self.keys[self.pointer] = key
            self.values.append(value)
            self.usage[self.pointer] = 1
            self.pointer = (self.pointer + 1) % self.capacity
            self.size += 1
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


    def retrieve(self, query_key, k=1):
        """
        Retrieves the top-k most similar values for a given query key
        using an attention-based mechanism.
        """
        if self.size == 0:
            return None, None

        # Attention mechanism
        similarities = F.cosine_similarity(query_key.unsqueeze(0), self.keys[:self.size], dim=1)
        scores = F.softmax(similarities, dim=0)

        # Update usage for LRU
        self.usage[:self.size] += 1
        self.usage[torch.argmax(scores)] = 1 # Most recently used

        # Retrieve top-k
        top_scores, top_indices = torch.topk(scores, k)
        
        retrieved_values = [self.values[i] for i in top_indices]
        return retrieved_values, top_scores

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


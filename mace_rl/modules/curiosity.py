import torch
import torch.nn as nn
from mace_rl.modules.episodic_memory import EpisodicMemory

class CuriosityModule(nn.Module):
    """
    Calculates the curiosity bonus.
    C(s,a) = η(s,a) × M(s,a) × T(s,a)
    """
    def __init__(self, state_dim, action_dim, memory: EpisodicMemory, continuous=False):
        super(CuriosityModule, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = memory
        self.continuous = continuous

        if self.continuous:
            # k-NN based novelty for continuous spaces
            self.k = 10
            self.state_buffer = torch.zeros(10000, self.state_dim)
            self.buffer_pointer = 0
        else:
            # Count-based novelty for discrete spaces
            self.visitation_counts = {}

        # Temporal importance weighting (placeholder)
        self.temporal_decay = 0.99

    def get_novelty(self, state):
        """Calculates the novelty score η(s,a)."""
        if self.continuous:
            if self.buffer_pointer == 0:
                return 1.0
            
            # Use k-NN to estimate density
            distances = torch.norm(self.state_buffer[:self.buffer_pointer] - state, dim=1)
            k_nearest_distances, _ = torch.topk(distances, self.k, largest=False)
            novelty = torch.mean(k_nearest_distances)

            # Add to buffer
            if self.buffer_pointer < self.state_buffer.size(0):
                self.state_buffer[self.buffer_pointer] = state
                self.buffer_pointer += 1
            return novelty
        else:
            state_tuple = tuple(state.cpu().numpy())
            count = self.visitation_counts.get(state_tuple, 0)
            self.visitation_counts[state_tuple] = count + 1
            # Novelty is inversely proportional to visitation count
            return 1.0 / (count + 1)

    def get_memory_relevance(self, state):
        """Calculates the memory relevance score M(s,a)."""
        _, scores = self.memory.retrieve(state)
        if scores is None:
            return 0.0
        return torch.mean(scores)

    def forward(self, state, action):
        """Calculates the total curiosity bonus."""
        novelty_score = self.get_novelty(state)
        memory_relevance_score = self.get_memory_relevance(state)
        
        # Temporal weighting is simplified for now
        temporal_score = 1.0 

        curiosity_bonus = novelty_score * memory_relevance_score * temporal_score
        return curiosity_bonus

if __name__ == '__main__':
    memory = EpisodicMemory(capacity=100, key_dim=10, value_dim=1)
    curiosity = CuriosityModule(state_dim=10, action_dim=2, memory=memory)

    # Add a memory
    key = torch.randn(10)
    value = torch.randn(1)
    memory.add(key, value)

    state = torch.randn(10)
    action = torch.tensor(1)
    bonus = curiosity(state, action)
    print("Curiosity bonus:", bonus)


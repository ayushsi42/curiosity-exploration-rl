import torch
import torch.nn as nn
from mace_rl.modules.episodic_memory import EpisodicMemory
from mace_rl.utils.logger import get_logger

logger = get_logger('CuriosityModule')

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
        try:
            # Ensure state is properly flattened
            if state.dim() > 1:
                flat_state = state.view(-1)
                logger.debug(f"Novelty: Flattened state from shape {state.shape} to {flat_state.shape}")
            else:
                flat_state = state
                
            if flat_state.shape[0] != self.state_dim:
                logger.error(f"State dimension mismatch in novelty. Expected {self.state_dim}, got {flat_state.shape[0]}")
                return 1.0  # Return maximum novelty on error

            if self.continuous:
                if self.buffer_pointer == 0:
                    logger.debug("Empty buffer, returning maximum novelty")
                    return 1.0
                
                # Use k-NN to estimate density
                distances = torch.norm(self.state_buffer[:self.buffer_pointer] - flat_state, dim=1)
                k_nearest_distances, _ = torch.topk(distances, min(self.k, self.buffer_pointer), largest=False)
                novelty = torch.mean(k_nearest_distances)
                logger.debug(f"Continuous novelty score: {novelty.item():.4f}")
                return novelty
            else:
                # Discrete state space - use count-based novelty
                state_hash = hash(str(flat_state.cpu().detach().numpy().tobytes()))
                count = self.visitation_counts.get(state_hash, 0)
                self.visitation_counts[state_hash] = count + 1
                novelty = 1.0 / (count + 1)
                logger.debug(f"Discrete novelty score: {novelty:.4f}")
                return novelty
        except Exception as e:
            logger.error(f"Error calculating novelty: {e}")
            return 1.0  # Return maximum novelty on error
        except Exception as e:
            logger.error(f"Error calculating novelty: {e}")
            return 1.0  # Return maximum novelty on error

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
        try:
            # If memory is empty, return default relevance of 1.0
            if self.memory.size == 0:
                logger.debug("Memory is empty, returning default relevance of 1.0")
                return 1.0

            # Ensure state is properly flattened
            if state.dim() > 1:
                flat_state = state.view(-1)
                logger.debug(f"Flattened state from shape {state.shape} to {flat_state.shape}")
            else:
                flat_state = state

            if flat_state.shape[0] != self.state_dim:
                logger.error(f"State dimension mismatch. Expected {self.state_dim}, got {flat_state.shape[0]}")
                return 1.0  # Return default relevance on dimension mismatch

            values, indices = self.memory.query(flat_state)
            if values is None or indices is None:
                logger.debug("No memory matches found, returning default relevance of 1.0")
                return 1.0

            # Calculate relevance based on distances
            distances = torch.norm(self.memory.keys[indices] - flat_state.unsqueeze(0), dim=1)
            relevance = 1.0 / (distances + 1e-6)  # Add small epsilon to prevent division by zero
            mean_relevance = torch.mean(relevance).item()
            logger.debug(f"Memory relevance score: {mean_relevance:.4f}")
            return mean_relevance
        except Exception as e:
            logger.error(f"Error calculating memory relevance: {e}")
            return 1.0  # Return default relevance on error

    def forward(self, state, action):
        """Calculates the total curiosity bonus."""
        # Ensure state is properly flattened for all operations
        if state.dim() > 1:
            flat_state = state.view(-1)
            logger.debug(f"Forward pass: Flattened state from shape {state.shape} to {flat_state.shape}")
        else:
            flat_state = state
            
        if flat_state.shape[0] != self.state_dim:
            logger.error(f"State dimension mismatch in forward pass. Expected {self.state_dim}, got {flat_state.shape[0]}")
            return 0.0

        novelty_score = self.get_novelty(flat_state)
        memory_relevance_score = self.get_memory_relevance(flat_state)
        
        # Temporal weighting is simplified for now
        temporal_score = 1.0 

        curiosity_bonus = novelty_score * memory_relevance_score * temporal_score
        logger.debug(f"Curiosity bonus components - Novelty: {novelty_score:.4f}, "
                    f"Memory relevance: {memory_relevance_score:.4f}, "
                    f"Total: {curiosity_bonus:.4f}")
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


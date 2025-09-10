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
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.continuous:
            # Advanced k-NN with adaptive clustering and ensemble methods
            self.k_values = [10, 15, 20]  # Multiple k values for ensemble
            self.state_buffer = torch.zeros(20000, self.state_dim).to(self.device)  # Larger buffer
            self.buffer_pointer = 0
            self.buffer_full = False
            
            # Advanced distance metrics with learned weights
            self.distance_weights = torch.nn.Parameter(
                torch.tensor([0.5, 0.3, 0.2]).to(self.device)  # L2, L1, cosine weights (learnable)
            )
            
            # Density estimation network
            self.density_net = torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
                torch.nn.Sigmoid()
            ).to(self.device)
            
            # Adaptive novelty threshold
            self.novelty_threshold = torch.nn.Parameter(torch.tensor(0.5).to(self.device))
        else:
            # Enhanced count-based novelty with temporal weighting
            self.visitation_counts = {}
            self.count_decay = 0.995
            self.temporal_weights = {}

        # Advanced temporal importance with learned decay
        self.temporal_decay = torch.nn.Parameter(torch.tensor(0.99))
        self.time_step = 0
        
        # Curiosity history for adaptation
        self.curiosity_history = []
        self.history_size = 1000
        
        # Forward Dynamics Model for prediction-based curiosity
        self.forward_model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim + self.action_dim, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.state_dim)
        ).to(self.device)
        
        # Inverse Dynamics Model for feature representation
        self.inverse_model = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim * 2, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.action_dim)
        ).to(self.device)
        
        # Feature encoder for ICM (Intrinsic Curiosity Module)
        self.feature_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
        ).to(self.device)
        
        # Prediction error history for adaptive thresholding
        self.prediction_errors = []
        self.error_history_size = 500

    def get_novelty(self, state, action=None):
        """Advanced novelty calculation using multiple methods."""
        try:
            # Ensure state is properly flattened
            if state.dim() > 1:
                flat_state = state.view(-1)
                logger.debug(f"Novelty: Flattened state from shape {state.shape} to {flat_state.shape}")
            else:
                flat_state = state
                
            if flat_state.shape[0] != self.state_dim:
                logger.error(f"State dimension mismatch in novelty. Expected {self.state_dim}, got {flat_state.shape[0]}")
                return 1.0

            if self.continuous:
                if self.buffer_pointer == 0:
                    return 1.0
                
                # Ensure state is on device
                flat_state = flat_state.to(self.device)
                
                # Ensemble k-NN novelty estimation
                ensemble_novelty = 0.0
                for k in self.k_values:
                    if self.buffer_pointer >= k:
                        distances = torch.norm(self.state_buffer[:self.buffer_pointer] - flat_state, dim=1)
                        k_nearest_distances, _ = torch.topk(distances, k, largest=False)
                        ensemble_novelty += torch.mean(k_nearest_distances)
                ensemble_novelty /= len(self.k_values)
                
                # Neural density estimation
                state_batch = flat_state.unsqueeze(0)
                neural_novelty = 1.0 - self.density_net(state_batch).squeeze()
                
                # Combine ensemble and neural novelty
                combined_novelty = 0.7 * ensemble_novelty + 0.3 * neural_novelty
                
                # Add state to buffer
                if self.buffer_pointer < self.state_buffer.size(0):
                    self.state_buffer[self.buffer_pointer] = flat_state.cpu()  # Store on CPU for memory efficiency
                    self.buffer_pointer += 1
                else:
                    # Replace oldest entry
                    idx = self.buffer_pointer % self.state_buffer.size(0)
                    self.state_buffer[idx] = flat_state.cpu()
                    self.buffer_pointer += 1
                
                return float(combined_novelty.cpu().item())
            else:
                # Enhanced count-based novelty with temporal weighting
                state_hash = hash(str(flat_state.cpu().detach().numpy().tobytes()))
                current_time = self.time_step
                
                if state_hash in self.visitation_counts:
                    # Apply temporal decay to old visits
                    last_visit_time = self.temporal_weights.get(state_hash, 0)
                    time_diff = current_time - last_visit_time
                    decayed_count = self.visitation_counts[state_hash] * (self.count_decay ** time_diff)
                    self.visitation_counts[state_hash] = decayed_count + 1
                else:
                    self.visitation_counts[state_hash] = 1
                
                self.temporal_weights[state_hash] = current_time
                novelty = 1.0 / (self.visitation_counts[state_hash] + 1)
                logger.debug(f"Enhanced discrete novelty score: {novelty:.4f}")
                return novelty
        except Exception as e:
            logger.error(f"Error calculating novelty: {e}")
            return 1.0  # Return maximum novelty on error

    def get_prediction_curiosity(self, state, action, next_state):
        """ICM-based prediction curiosity using forward and inverse dynamics."""
        try:
            # Move inputs to device
            state = state.to(self.device)
            next_state = next_state.to(self.device)
            
            # Ensure proper dimensions
            if state.dim() > 1:
                state = state.view(-1)
            if next_state.dim() > 1:
                next_state = next_state.view(-1)
            if isinstance(action, (int, float)):
                action = torch.tensor([action], dtype=torch.float32).to(self.device)
            elif action.dim() == 0:
                action = action.unsqueeze(0).to(self.device)
            else:
                action = action.to(self.device)
            
            # Encode states for better feature representation
            encoded_state = self.feature_encoder(state.unsqueeze(0))
            encoded_next_state = self.feature_encoder(next_state.unsqueeze(0))
            
            # Forward model prediction
            state_action = torch.cat([state.unsqueeze(0), action.unsqueeze(0)], dim=1)
            predicted_next_state = self.forward_model(state_action)
            
            # Prediction error (intrinsic curiosity)
            prediction_error = torch.nn.functional.mse_loss(
                predicted_next_state, next_state.unsqueeze(0)
            )
            
            # Inverse model prediction (for auxiliary learning)
            state_pair = torch.cat([encoded_state, encoded_next_state], dim=1)
            predicted_action = self.inverse_model(state_pair)
            
            # Store prediction error for adaptive thresholding
            self.prediction_errors.append(prediction_error.item())
            if len(self.prediction_errors) > self.error_history_size:
                self.prediction_errors.pop(0)
            
            # Normalize prediction error by historical statistics
            if len(self.prediction_errors) > 10:
                mean_error = sum(self.prediction_errors) / len(self.prediction_errors)
                std_error = (sum([(e - mean_error)**2 for e in self.prediction_errors]) / len(self.prediction_errors))**0.5
                normalized_error = (prediction_error.item() - mean_error) / (std_error + 1e-8)
                curiosity_bonus = torch.sigmoid(torch.tensor(normalized_error))
            else:
                curiosity_bonus = torch.sigmoid(prediction_error)
            
            return curiosity_bonus.item()
            
        except Exception as e:
            logger.error(f"Error calculating prediction curiosity: {e}")
            return 0.0

    def get_memory_relevance(self, state):
        """Calculates the memory relevance score M(s,a)."""
        try:
            # Check if memory exists and is not None
            if self.memory is None:
                logger.debug("Memory is None, returning default relevance of 1.0")
                return 1.0
                
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
        
        # Normalize curiosity bonus to reasonable scale (0.0 to 1.0)
        curiosity_bonus = min(curiosity_bonus, 1.0)  # Cap at 1.0
        
        # logger.debug(f"Curiosity bonus components - Novelty: {novelty_score:.4f}, "
        #             f"Memory relevance: {memory_relevance_score:.4f}, "
        #             f"Total: {curiosity_bonus:.4f}")  # Commented to avoid tqdm interference
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


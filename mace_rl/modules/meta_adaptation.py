import torch
import torch.nn as nn
from mace_rl.utils.logger import get_logger

logger = get_logger('MetaAdaptation')

class MetaAdaptation(nn.Module):
    """
    Advanced Meta-Adaptation Network with Transformer architecture and hierarchical learning.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, fast_lr=0.1, slow_lr=0.01):
        super(MetaAdaptation, self).__init__()
        
        # Input projection and embedding
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Enhanced LSTM with residual connections
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=3, batch_first=True, 
                           dropout=0.15, bidirectional=True)
        
        # Multi-head attention with multiple layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=0.1, batch_first=True)
            for _ in range(2)
        ])
        
        # Layer normalization for attention layers
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * 2) for _ in range(2)
        ])
        
        # Hierarchical output processing
        self.feature_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Multi-task output heads
        self.output_heads = nn.ModuleDict({
            'curiosity_weights': nn.Linear(hidden_dim // 2, output_dim),
            'exploration_strategy': nn.Linear(hidden_dim // 2, output_dim),
            'adaptation_rate': nn.Linear(hidden_dim // 2, 1)
        })
        
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        
        # This will be managed by an external optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.slow_lr, weight_decay=1e-4)

    def forward(self, x, hidden_state=None):
        """
        Advanced forward pass with hierarchical attention and multi-task outputs.
        """
        logger.debug(f"Meta-adaptation forward pass - Input shape: {x.shape if hasattr(x, 'shape') else 'N/A'}")
        try:
            # Input projection
            x_proj = self.input_projection(x)
            
            # LSTM processing with residual connection
            lstm_out, hidden_state = self.lstm(x_proj, hidden_state)
            
            # Multi-layer attention processing
            attended = lstm_out
            for attention_layer, norm_layer in zip(self.attention_layers, self.attention_norms):
                # Apply attention with residual connection
                attn_out, _ = attention_layer(attended, attended, attended)
                attended = norm_layer(attended + attn_out)  # Residual connection
            
            # Aggregate features across sequence (weighted average)
            sequence_weights = torch.softmax(
                torch.mean(attended, dim=-1), dim=1
            ).unsqueeze(-1)
            aggregated = torch.sum(attended * sequence_weights, dim=1)
            
            # Feature processing
            features = self.feature_aggregator(aggregated)
            
            # Multi-task outputs (return primary output for compatibility)
            primary_output = self.output_heads['curiosity_weights'](features)
            
            logger.debug(f"Meta-adaptation output shape: {primary_output.shape}")
            return primary_output, hidden_state
        except Exception as e:
            logger.error(f"Error in meta-adaptation forward pass: {e}")
            raise

    def update_meta(self, loss):
        """
        Performs a meta-update on the slow weights.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    # Example Usage
    input_dim = 10 # Represents exploration outcomes
    hidden_dim = 128
    output_dim = 2 # e.g., beta for reward and novelty weight
    
    meta_net = MetaAdaptation(input_dim, hidden_dim, output_dim)
    
    # Simulate a sequence of exploration outcomes
    exploration_outcomes = torch.randn(1, 20, input_dim) # (batch, seq_len, input_dim)
    
    # Get curiosity parameters
    curiosity_params, _ = meta_net(exploration_outcomes)
    
    print("Output curiosity params shape:", curiosity_params.shape)
    
    # Simulate a meta-loss and update
    meta_loss = torch.randn(1)
    # In a real scenario, this loss would be calculated based on agent performance
    # meta_net.update_meta(meta_loss) 
    # print("Meta-update performed.")



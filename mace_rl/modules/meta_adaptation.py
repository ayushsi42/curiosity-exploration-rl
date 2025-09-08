import torch
import torch.nn as nn
from mace_rl.utils.logger import get_logger

logger = get_logger('MetaAdaptation')

class MetaAdaptation(nn.Module):
    """
    Meta-Adaptation Network using an LSTM to adapt curiosity parameters.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, fast_lr=0.1, slow_lr=0.01):
        super(MetaAdaptation, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        
        # This will be managed by an external optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.slow_lr)

    def forward(self, x, hidden_state=None):
        """
        Takes exploration outcomes and outputs curiosity weighting parameters.
        """
        logger.debug(f"Meta-adaptation forward pass - Input shape: {x.shape if hasattr(x, 'shape') else 'N/A'}")
        try:
            lstm_out, hidden_state = self.lstm(x, hidden_state)
            output = self.fc(lstm_out[:, -1, :])  # Use last timestep
            logger.debug(f"Meta-adaptation output shape: {output.shape}")
            return output, hidden_state
        except Exception as e:
            logger.error(f"Error in meta-adaptation forward pass: {e}")
            raise
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        output = self.fc(lstm_out)
        return output, hidden_state

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



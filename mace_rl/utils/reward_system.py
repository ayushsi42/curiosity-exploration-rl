from mace_rl.modules.curiosity import CuriosityModule
from mace_rl.utils.logger import get_logger
from mace_rl.utils.beta_schedulers import BetaScheduler, create_beta_scheduler
from typing import Optional, Union, Dict, Any

logger = get_logger('HybridRewardSystem')

class HybridRewardSystem:
    """
    Combines intrinsic and extrinsic rewards with advanced beta scheduling.
    R_total = R_extrinsic + β(t) × R_curiosity
    
    Where β(t) changes over time according to a scheduling strategy.
    """
    def __init__(self, curiosity_module: Optional[CuriosityModule] = None, 
                 beta_scheduler: Optional[Union[BetaScheduler, str, dict]] = None,
                 # Legacy parameters for backwards compatibility
                 beta_initial: float = 1.0, beta_decay: float = 0.999, beta_min: float = 0.1):
        self.curiosity_module = curiosity_module
        
        # Initialize beta scheduler
        if beta_scheduler is None:
            # Use legacy exponential decay if no scheduler specified
            self.beta_scheduler = create_beta_scheduler(
                'exponential_decay',
                beta_initial=beta_initial,
                beta_min=beta_min,
                decay_rate=beta_decay
            )
        elif isinstance(beta_scheduler, str):
            # Create scheduler from string
            self.beta_scheduler = create_beta_scheduler(beta_scheduler)
        elif isinstance(beta_scheduler, dict):
            # Create scheduler from config dict
            scheduler_type = beta_scheduler.pop('type', 'exponential_decay')
            self.beta_scheduler = create_beta_scheduler(scheduler_type, **beta_scheduler)
        elif isinstance(beta_scheduler, BetaScheduler):
            # Use provided scheduler
            self.beta_scheduler = beta_scheduler
        else:
            raise ValueError(f"Invalid beta_scheduler type: {type(beta_scheduler)}")
        
        # Legacy attributes for backwards compatibility
        self.beta = self.beta_scheduler.get_beta()
        self.beta_decay = beta_decay
        self.beta_min = beta_min
        
        # Episode tracking
        self.current_episode = 0
        self.current_step = 0
        self.episode_performance = []

    def get_total_reward(self, state, action, extrinsic_reward):
        """Calculates the total reward using current beta from scheduler."""
        # logger.debug(f"Computing total reward - State shape: {state.shape if hasattr(state, 'shape') else 'N/A'}, "
        #             f"Action: {action}, Extrinsic reward: {extrinsic_reward}")  # Commented to avoid tqdm interference
        try:
            # Ensure state is flattened
            if state.dim() > 1:
                state = state.view(-1)
                # logger.debug(f"Flattened state to shape: {state.shape}")  # Commented to avoid tqdm interference
            
            # Calculate curiosity bonus
            if self.curiosity_module is not None:
                curiosity_bonus = self.curiosity_module(state, action)
            else:
                curiosity_bonus = 0.0  # No curiosity bonus if module is None
            
            # Scale curiosity bonus relative to typical extrinsic rewards
            # For CartPole, typical rewards are 1.0, so scale accordingly
            scaled_curiosity = curiosity_bonus * abs(extrinsic_reward) if extrinsic_reward != 0 else curiosity_bonus * 0.1
            
            # Get current beta from scheduler
            current_beta = self.beta_scheduler.get_beta()
            self.beta = current_beta  # Update legacy attribute
            
            total_reward = extrinsic_reward + current_beta * scaled_curiosity
            
            # Track performance for adaptive schedulers
            self.episode_performance.append(extrinsic_reward)
            
            # logger.debug(f"Reward computation - Curiosity bonus: {curiosity_bonus}, "
            #             f"Scaled: {scaled_curiosity}, Beta: {current_beta}, Total reward: {total_reward}")  # Commented to avoid tqdm interference
            return total_reward
        except Exception as e:
            logger.error(f"Error in reward computation: {e}")
            return extrinsic_reward

    def update_beta(self, episode: int = None, step: int = None):
        """Updates beta using the scheduler and optionally provides episode/step info."""
        if episode is not None:
            self.current_episode = episode
        if step is not None:
            self.current_step = step
            
        # Calculate performance metric for adaptive schedulers
        performance_metric = None
        if self.episode_performance:
            performance_metric = sum(self.episode_performance)
            # Reset for next episode if episode changed
            if episode is not None and episode != self.current_episode:
                self.episode_performance = []
        
        # Update scheduler
        current_beta = self.beta_scheduler.update(
            episode=self.current_episode,
            step=self.current_step,
            performance_metric=performance_metric
        )
        
        self.beta = current_beta  # Update legacy attribute
        return current_beta
    
    def get_scheduler_info(self) -> Dict[str, Any]:
        """Get information about the current scheduler state."""
        return {
            'current_beta': self.beta_scheduler.get_beta(),
            'scheduler_info': self.beta_scheduler.get_info(),
            'episode': self.current_episode,
            'step': self.current_step
        }

if __name__ == '__main__':
    # This is a placeholder for the actual modules
    class MockMemory:
        def retrieve(self, state):
            return None, None
    class MockCuriosity(CuriosityModule):
        def __init__(self):
            self.memory = MockMemory()
        def __call__(self, state, action):
            return 0.5
    
    import torch

    curiosity = MockCuriosity()
    reward_system = HybridRewardSystem(curiosity)
    
    state = torch.randn(10)
    action = 1
    extrinsic_reward = 0.1
    
    total_reward = reward_system.get_total_reward(state, action, extrinsic_reward)
    # print("Total reward:", total_reward)  # Commented out to avoid tqdm interference
    
    reward_system.update_beta()
    # print("Updated beta:", reward_system.beta)  # Commented out to avoid tqdm interference


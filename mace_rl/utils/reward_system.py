from mace_rl.modules.curiosity import CuriosityModule

class HybridRewardSystem:
    """
    Combines intrinsic and extrinsic rewards.
    R_total = R_extrinsic + β × R_curiosity
    """
    def __init__(self, curiosity_module: CuriosityModule, beta_initial=1.0, beta_decay=0.999, beta_min=0.1):
        self.curiosity_module = curiosity_module
        self.beta = beta_initial
        self.beta_decay = beta_decay
        self.beta_min = beta_min

    def get_total_reward(self, state, action, extrinsic_reward):
        """Calculates the total reward."""
        curiosity_bonus = self.curiosity_module(state, action)
        total_reward = extrinsic_reward + self.beta * curiosity_bonus
        return total_reward

    def update_beta(self):
        """Updates beta based on a decay schedule."""
        self.beta = max(self.beta * self.beta_decay, self.beta_min)

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
    print("Total reward:", total_reward)
    
    reward_system.update_beta()
    print("Updated beta:", reward_system.beta)


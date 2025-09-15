import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical, MultivariateNormal
from mace_rl.utils.logger import get_logger

logger = get_logger('PPO')

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init)

        # Advanced CNN with residual connections and layer normalization
        if isinstance(state_dim, tuple):
            self.cnn = nn.Sequential(
                # First conv block
                nn.Conv2d(state_dim[2], 64, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, 64),  # Use GroupNorm instead of BatchNorm2d
                nn.ReLU(),
                nn.Dropout2d(0.1),
                
                # Second conv block
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(16, 128),
                nn.ReLU(),
                nn.Dropout2d(0.1),
                
                # Third conv block
                nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=0),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                
                # Global Average Pooling + Spatial Attention
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                
                # Enhanced feature processing
                nn.Linear(64 * 4 * 4, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            feature_dim = 256  # Fixed output dimension from enhanced CNN
        else:
            self.cnn = None
            feature_dim = state_dim

        # actor - Enhanced with residual connections and layer normalization (better for single samples)
        if has_continuous_action_space :
            self.actor_backbone = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.LayerNorm(512),  # Use LayerNorm instead of BatchNorm
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
            )
            self.actor_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
            # Residual connection layer
            self.actor_residual = nn.Linear(feature_dim, 128) if feature_dim != 128 else nn.Identity()
        else:
            self.actor_backbone = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
            )
            self.actor_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )
            # Residual connection layer
            self.actor_residual = nn.Linear(feature_dim, 128) if feature_dim != 128 else nn.Identity()

        # critic - Enhanced with attention mechanism and residual connections
        self.critic_backbone = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        
        # Self-attention for critic
        self.critic_attention = nn.MultiheadAttention(128, num_heads=8, dropout=0.1, batch_first=True)
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Residual connection layer for critic
        self.critic_residual = nn.Linear(feature_dim, 128) if feature_dim != 128 else nn.Identity()

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)

    def forward(self, state):
        if self.cnn:
            state = state.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W
            features = self.cnn(state)
        else:
            features = state
        return features

    def actor_forward(self, features):
        """Enhanced actor forward with residual connections"""
        # Main pathway
        x = self.actor_backbone(features)
        
        # Residual connection
        residual = self.actor_residual(features)
        x = x + residual
        
        # Output head
        return self.actor_head(x)

    def critic_forward(self, features):
        """Enhanced critic forward with attention and residual connections"""
        # Main pathway
        x = self.critic_backbone(features)
        
        # Residual connection
        residual = self.critic_residual(features)
        x = x + residual
        
        # Self-attention (expand dimensions for attention)
        x_expanded = x.unsqueeze(1)  # Add sequence dimension
        attended, _ = self.critic_attention(x_expanded, x_expanded, x_expanded)
        x = attended.squeeze(1)  # Remove sequence dimension
        
        # Output head
        return self.critic_head(x)

    def act(self, state):
        features = self.forward(state)
        if self.has_continuous_action_space:
            action_mean = self.actor_forward(features)
            cov_mat = torch.diag(self.action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor_forward(features)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        features = self.forward(state)
        if self.has_continuous_action_space:
            action_mean = self.actor_forward(features)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor_forward(features)
            dist = Categorical(action_probs)
            
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic_forward(features)
        
        return action_logprobs, state_values, dist_entropy


from mace_rl.utils.utils import profile

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
        self.has_continuous_action_space = has_continuous_action_space
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # GPU Support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PPO using device: {self.device}")

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)
        
        # Collect all actor and critic parameters for optimizer
        actor_params = (
            list(self.policy.actor_backbone.parameters()) + 
            list(self.policy.actor_head.parameters()) + 
            list(self.policy.actor_residual.parameters())
        )
        critic_params = (
            list(self.policy.critic_backbone.parameters()) + 
            list(self.policy.critic_attention.parameters()) + 
            list(self.policy.critic_head.parameters()) + 
            list(self.policy.critic_residual.parameters())
        )
        
        self.optimizer = torch.optim.Adam([
            {'params': actor_params, 'lr': lr_actor},
            {'params': critic_params, 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

    def select_action(self, state):
        with torch.no_grad():
            # Ensure state is a torch tensor and on the correct device
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)
            state = state.unsqueeze(0) if state.ndim == 1 else state
            state = state.to(self.device)
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state.squeeze(0).cpu())  # Store on CPU to save GPU memory
        self.buffer.actions.append(action.cpu())
        self.buffer.logprobs.append(action_logprob.cpu())

        if self.has_continuous_action_space:
            return action.cpu().numpy().flatten()
        return action.cpu().item()

    def update(self):
        # logger.debug("Starting policy update")  # Commented to avoid tqdm interference
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # logger.debug(f"Processed rewards - Mean: {rewards.mean():.4f}, Std: {rewards.std():.4f}")  # Commented to avoid tqdm interference

        # convert list to tensor and move to device
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * self.MseLoss(state_values, rewards)
            entropy_loss = -0.01 * dist_entropy.mean()
            loss = actor_loss + critic_loss + entropy_loss

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # logger.debug(f"Policy update stats - Actor Loss: {actor_loss.item():.4f}, "
            #             f"Critic Loss: {critic_loss.item():.4f}, "
            #             f"Entropy Loss: {entropy_loss.item():.4f}")  # Commented to avoid tqdm interference

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # logger.debug("Updated policy network weights")  # Commented to avoid tqdm interference

        # clear buffer
        self.buffer.clear()
        # logger.debug("Cleared replay buffer")  # Commented to avoid tqdm interference

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': loss.item()
        }

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

if __name__ == '__main__':
    # Example usage for discrete action space
    state_dim_disc = 4
    action_dim_disc = 2
    ppo_agent_disc = PPO(state_dim_disc, action_dim_disc, 0.0003, 0.001, 0.99, 4, 0.2, False)
    state_disc = torch.randn(state_dim_disc)
    action_disc = ppo_agent_disc.select_action(state_disc)
    # print("Selected discrete action:", action_disc)  # Commented to avoid tqdm interference

    # Example usage for continuous action space
    state_dim_cont = 3
    action_dim_cont = 2
    ppo_agent_cont = PPO(state_dim_cont, action_dim_cont, 0.0003, 0.001, 0.99, 4, 0.2, True)
    state_cont = torch.randn(state_dim_cont)
    action_cont = ppo_agent_cont.select_action(state_cont)
    # print("Selected continuous action:", action_cont)  # Commented to avoid tqdm interference


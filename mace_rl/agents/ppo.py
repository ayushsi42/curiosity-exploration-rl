import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal

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

        # CNN for image observations
        if isinstance(state_dim, tuple):
            self.cnn = nn.Sequential(
                nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten()
            )
            # Calculate the output size of the CNN
            with torch.no_grad():
                dummy_input = torch.zeros(1, *state_dim)
                cnn_out_dim = self.cnn(dummy_input).shape[1]
            feature_dim = cnn_out_dim
        else:
            self.cnn = None
            feature_dim = state_dim

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)

    def forward(self, state):
        if self.cnn:
            state = state.permute(0, 3, 1, 2) # N, H, W, C -> N, C, H, W
            features = self.cnn(state)
        else:
            features = state
        return features

    def act(self, state):
        features = self.forward(state)
        if self.has_continuous_action_space:
            action_mean = self.actor(features)
            cov_mat = torch.diag(self.action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(features)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        features = self.forward(state)
        if self.has_continuous_action_space:
            action_mean = self.actor(features)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(features)
            dist = Categorical(action_probs)
            
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)
        
        return action_logprobs, state_values, dist_entropy


from mace_rl.utils.utils import profile

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
        self.has_continuous_action_space = has_continuous_action_space
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state.squeeze(0))
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        if self.has_continuous_action_space:
            return action.numpy().flatten()
        return action.item()

    @profile
    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()

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
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

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
    print("Selected discrete action:", action_disc)

    # Example usage for continuous action space
    state_dim_cont = 3
    action_dim_cont = 2
    ppo_agent_cont = PPO(state_dim_cont, action_dim_cont, 0.0003, 0.001, 0.99, 4, 0.2, True)
    state_cont = torch.randn(state_dim_cont)
    action_cont = ppo_agent_cont.select_action(state_cont)
    print("Selected continuous action:", action_cont)


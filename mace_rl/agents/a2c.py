import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
from mace_rl.agents.ppo import ActorCritic, RolloutBuffer

class A2C:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, has_continuous_action_space, action_std_init=0.6):
        self.has_continuous_action_space = has_continuous_action_space
        self.gamma = gamma

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.policy.set_action_std(new_action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, action_logprob = self.policy.act(state)

        self.buffer.states.append(state.squeeze(0))
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        if self.has_continuous_action_space:
            return action.numpy().flatten()
        return action.item()

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
        
        # Evaluate old actions and values
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

        # match state_values tensor dimensions with rewards tensor
        state_values = torch.squeeze(state_values)
        
        # Finding Advantage
        advantages = rewards - state_values.detach()
        
        # Finding the loss
        actor_loss = -(logprobs * advantages).mean()
        critic_loss = self.MseLoss(state_values, rewards)
        
        loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()
        
        # take gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


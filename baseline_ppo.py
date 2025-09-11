import argparse
import torch
import gym

from mace_rl.utils.utils import set_seed, log_data
from mace_rl.agents.ppo import PPO
from mace_rl.envs.minigrid_env import make_minigrid_env
from mace_rl.envs.pybullet_env import make_pybullet_env

def train(args):
    """Main training loop for PPO baseline."""
    set_seed(args.seed)

    # Initialize environment
    if "NoFrameskip" in args.env_name:
    # Atari environment removed
    raise NotImplementedError('Atari environments are removed from this codebase.')
        env = wrap_deepmind(env, frame_stack=True, clip_rewards=True)
        has_continuous_action_space = False
    elif "MiniGrid" in args.env_name:
        env = make_minigrid_env(args.env_name)
        has_continuous_action_space = False
    else:
        env = make_pybullet_env(args.env_name)
        has_continuous_action_space = True

    state_dim = env.observation_space.shape
    
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # Initialize PPO agent
    ppo_agent = PPO(state_dim, action_dim, args.lr_actor, args.lr_critic, 
                    args.gamma, args.k_epochs, args.eps_clip, has_continuous_action_space)

    # Training loop
    for episode in range(args.max_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        for t in range(args.max_timesteps):
            action = ppo_agent.select_action(state)
            next_state, extrinsic_reward, done, _ = env.step(action)
            
            ppo_agent.buffer.rewards.append(extrinsic_reward)
            ppo_agent.buffer.is_terminals.append(done)
            
            state = next_state
            episode_reward += extrinsic_reward

            if done:
                break
        
        ppo_agent.update()

        print(f"Episode {episode}: Total Reward: {episode_reward}")
        
        if args.log_file:
            log_data(args.log_file, {'episode': episode, 'reward': episode_reward})

        if episode % args.save_interval == 0:
            ppo_agent.save(f"ppo_baseline_{args.env_name}_{episode}.pth")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO Baseline')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--env-name', type=str, default='PongNoFrameskip-v4', help='Gym environment name.')
    parser.add_argument('--max-episodes', type=int, default=1000, help='Max training episodes.')
    parser.add_argument('--max-timesteps', type=int, default=10000, help='Max timesteps per episode.')
    parser.add_argument('--lr-actor', type=float, default=0.0003, help='Actor learning rate.')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='Critic learning rate.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--k-epochs', type=int, default=4, help='PPO epochs.')
    parser.add_argument('--eps-clip', type=float, default=0.2, help='PPO clip parameter.')
    parser.add_argument('--save-interval', type=int, default=100, help='Save model every N episodes.')
    parser.add_argument('--log-file', type=str, default=None, help='Log file to save results.')

    args = parser.parse_args()
    train(args)

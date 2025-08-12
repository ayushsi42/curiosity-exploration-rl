import argparse
import torch
import gym

from mace_rl.utils.utils import set_seed, log_data
from mace_rl.agents.ppo import PPO
from mace_rl.modules.episodic_memory import EpisodicMemory
from mace_rl.modules.curiosity import CuriosityModule
from mace_rl.modules.meta_adaptation import MetaAdaptation
from mace_rl.utils.reward_system import HybridRewardSystem
from mace_rl.envs.minigrid_env import make_minigrid_env
from mace_rl.envs.pybullet_env import make_pybullet_env
from mace_rl.envs.atari_env import make_atari_env, wrap_deepmind

def train(args):
    """Main training loop for ablation studies."""
    set_seed(args.seed)

    # Initialize environment
    if "NoFrameskip" in args.env_name:
        env = make_atari_env(args.env_name)
        env = wrap_deepmind(env, frame_stack=True, clip_rewards=True)
        has_continuous_action_space = False
        is_atari = True
    elif "MiniGrid" in args.env_name:
        env = make_minigrid_env(args.env_name)
        has_continuous_action_space = False
        is_atari = False
    else:
        env = make_pybullet_env(args.env_name)
        has_continuous_action_space = True
        is_atari = False

    state_dim = env.observation_space.shape
    
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # Initialize components based on ablation flags
    episodic_memory = None
    if not args.no_episodic_memory and not is_atari:
        episodic_memory = EpisodicMemory(args.memory_capacity, state_dim[0])

    curiosity_module = None
    if not args.no_curiosity and not is_atari:
        curiosity_module = CuriosityModule(state_dim[0], action_dim, episodic_memory, continuous=has_continuous_action_space)

    meta_adaptation = None
    if not args.no_meta_adaptation:
        meta_adaptation = MetaAdaptation(state_dim[0] if not is_atari else 256, 128, 1)

    reward_system = HybridRewardSystem(curiosity_module, beta_initial=args.beta_start)
    
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
            
            total_reward = extrinsic_reward
            if curiosity_module:
                total_reward = reward_system.get_total_reward(torch.FloatTensor(state), torch.FloatTensor(action) if has_continuous_action_space else torch.LongTensor([action]), extrinsic_reward)

            ppo_agent.buffer.rewards.append(total_reward)
            ppo_agent.buffer.is_terminals.append(done)
            
            if episodic_memory:
                value = torch.cat([torch.FloatTensor(next_state), torch.FloatTensor([extrinsic_reward])])
                episodic_memory.add(torch.FloatTensor(state), value)

            state = next_state
            episode_reward += extrinsic_reward

            if done:
                break
        
        ppo_agent.update()

        if meta_adaptation:
            if episode % 10 == 0:
                reward_system.update_beta()

        print(f"Episode {episode}: Total Reward: {episode_reward}, Beta: {reward_system.beta}")
        
        if args.log_file:
            log_data(args.log_file, {'episode': episode, 'reward': episode_reward, 'beta': reward_system.beta})

        if episode % args.save_interval == 0:
            ppo_agent.save(f"ppo_ablation_{args.env_name}_{episode}.pth")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MACE-RL Ablation Studies')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--env-name', type=str, default='PongNoFrameskip-v4', help='Gym environment name.')
    parser.add_argument('--max-episodes', type=int, default=1000, help='Max training episodes.')
    parser.add_argument('--max-timesteps', type=int, default=10000, help='Max timesteps per episode.')
    parser.add_argument('--memory-capacity', type=int, default=1000, help='Episodic memory capacity.')
    parser.add_argument('--lr-actor', type=float, default=0.0003, help='Actor learning rate.')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='Critic learning rate.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--k-epochs', type=int, default=4, help='PPO epochs.')
    parser.add_argument('--eps-clip', type=float, default=0.2, help='PPO clip parameter.')
    parser.add_argument('--beta-start', type=float, default=1.0, help='Initial beta for curiosity.')
    parser.add_argument('--save-interval', type=int, default=100, help='Save model every N episodes.')
    parser.add_argument('--log-file', type=str, default=None, help='Log file to save results.')
    parser.add_argument('--no-episodic-memory', action='store_true', help='Disable episodic memory.')
    parser.add_argument('--no-curiosity', action='store_true', help='Disable curiosity module.')
    parser.add_argument('--no-meta-adaptation', action='store_true', help='Disable meta-adaptation network.')

    args = parser.parse_args()
    train(args)

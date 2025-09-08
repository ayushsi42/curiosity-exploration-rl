import argparse
import torch
import gym
import os
import logging
import numpy as np
from datetime import datetime

from mace_rl.utils.utils import set_seed, log_data
from mace_rl.utils.logger import setup_logger, get_logger
from mace_rl.agents.ppo import PPO
from mace_rl.modules.episodic_memory import EpisodicMemory
from mace_rl.modules.curiosity import CuriosityModule
from mace_rl.modules.meta_adaptation import MetaAdaptation
from mace_rl.utils.reward_system import HybridRewardSystem
from mace_rl.envs.minigrid_env import make_minigrid_env
from mace_rl.envs.pybullet_env import make_pybullet_env
from mace_rl.envs.atari_env import make_atari_env, wrap_deepmind

def train(args):
    """Main training loop."""
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', 'runs', f'{args.env_name}_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logger(
        'train',
        os.path.join(log_dir, 'train.log'),
        level=logging.DEBUG,
        console_output=True
    )
    logger.info(f"Starting training with arguments: {vars(args)}")
    
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Initialize environment
    logger.info(f"Initializing environment: {args.env_name}")
    if "NoFrameskip" in args.env_name:
        env = make_atari_env(args.env_name)
        env = wrap_deepmind(env, frame_stack=True, clip_rewards=True)
        has_continuous_action_space = False
        is_atari = True
        logger.info("Environment type: Atari")
    elif "MiniGrid" in args.env_name:
        env = make_minigrid_env(args.env_name)
        has_continuous_action_space = False
        is_atari = False
        logger.info("Environment type: MiniGrid")
    else:
        env = make_pybullet_env(args.env_name)
        has_continuous_action_space = True
        is_atari = False
        logger.info("Environment type: PyBullet")

    state_dim = env.observation_space.shape
    
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # Initialize all components
    if is_atari:
        episodic_memory = None
        curiosity_module = None 
    else:
        # Calculate the flattened state dimension
        if len(state_dim) == 3:  # Image-based states (H, W, C)
            state_size = state_dim[0] * state_dim[1] * state_dim[2]
            logger.info(f"Using flattened state size: {state_size} for image-based input")
        else:
            state_size = state_dim[0]
            logger.info(f"Using direct state size: {state_size} for vector input")
        
        episodic_memory = EpisodicMemory(args.memory_capacity, state_size)
        curiosity_module = CuriosityModule(state_size, action_dim, episodic_memory, continuous=has_continuous_action_space)

    meta_adaptation = MetaAdaptation(state_dim[0] if not is_atari else 256, 128, 1)
    reward_system = HybridRewardSystem(curiosity_module, beta_initial=args.beta_start)
    
    ppo_agent = PPO(state_dim, action_dim, args.lr_actor, args.lr_critic, 
                    args.gamma, args.k_epochs, args.eps_clip, has_continuous_action_space)

    # Training loop
    logger.info("Starting training loop")
    episode_rewards = []
    best_reward = float('-inf')

    for episode in range(args.max_episodes):
        logger.info(f"\nStarting Episode {episode}/{args.max_episodes}")
        # state = env.reset()
        reset_out = env.reset()
        if isinstance(reset_out, tuple):  # gymnasium API
            state, info = reset_out
        else:  # old gym API
            state = reset_out
        info = {}
        done = False
        episode_reward = 0
        step_count = 0

        logger.debug(f"Initial state type: {type(state)}, shape: {np.array(state).shape if isinstance(state, (np.ndarray, tuple)) else 'N/A'}")

        for t in range(args.max_timesteps):
            step_count += 1
            action = ppo_agent.select_action(state)
            logger.debug(f"Step {t}: Selected action: {action}")

            # FIX: Unpack 5 values from env.step for gymnasium API
            next_state, extrinsic_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated # Combine terminated and truncated for the 'done' signal
            
            logger.debug(f"Step {t}: Reward: {extrinsic_reward}, Done: {done}")
            
            total_reward = extrinsic_reward
            if curiosity_module:
                try:
                    # Convert and flatten state
                    if isinstance(state, tuple):
                        state_tensor = torch.FloatTensor(state[0])
                    else:
                        state_tensor = torch.FloatTensor(state)
                    state_tensor = state_tensor.view(-1)  # Flatten the state
                    
                    # Convert action to tensor
                    action_tensor = torch.FloatTensor(action) if has_continuous_action_space else torch.LongTensor([action])
                    
                    logger.debug(f"\nStep {t} - Reward Calculation:")
                    logger.debug(f"  State shape: {state_tensor.shape}")
                    logger.debug(f"  Action: {action}")
                    logger.debug(f"  Extrinsic reward: {extrinsic_reward}")
                    
                    # Calculate curiosity-adjusted reward
                    total_reward = reward_system.get_total_reward(
                        state_tensor,
                        action_tensor,
                        extrinsic_reward
                    )
                    logger.debug(f"  Final reward = {total_reward} (extrinsic: {extrinsic_reward}, curiosity bonus: {total_reward - extrinsic_reward})")
                except Exception as e:
                    logger.error(f"Error in curiosity calculation: {e}")
                    total_reward = extrinsic_reward
                    logger.error(f"Using fallback extrinsic reward: {total_reward}")

            ppo_agent.buffer.rewards.append(total_reward)
            ppo_agent.buffer.is_terminals.append(done)
            
            if episodic_memory:
                try:
                    # Process current state
                    if isinstance(state, tuple):
                        state_tensor = torch.FloatTensor(state[0])
                    else:
                        state_tensor = torch.FloatTensor(state)
                    
                    # Process next state
                    if isinstance(next_state, tuple):
                        next_state_tensor = torch.FloatTensor(next_state[0])
                    else:
                        next_state_tensor = torch.FloatTensor(next_state)
                    
                    # Ensure proper flattening for both states
                    state_flat = state_tensor.view(-1)  # Use view instead of flatten for better error messages
                    next_state_flat = next_state_tensor.view(-1)
                    
                    logger.debug(f"Step {t} - Memory Update:")
                    logger.debug(f"  State: original shape {state_tensor.shape}, flattened shape {state_flat.shape}")
                    logger.debug(f"  Next state: original shape {next_state_tensor.shape}, flattened shape {next_state_flat.shape}")
                    
                    # Create value tensor with next state and reward
                    value = torch.cat([next_state_flat, torch.FloatTensor([total_reward])])  # Use total_reward instead of extrinsic_reward
                    logger.debug(f"  Value tensor shape: {value.shape} (includes state {next_state_flat.shape} and reward)")
                    
                    # Log shapes for debugging
                    logger.debug(f"Step {t}: Memory key shape: {state_flat.shape}, "
                               f"value shape: {value.shape}")
                    
                    # Add to memory and log details
                    episodic_memory.add(state_flat, value)
                    logger.debug(f"  Memory status: size {episodic_memory.size}/{episodic_memory.capacity}")
                    logger.debug(f"  Successfully added state->value mapping to memory")
                except Exception as e:
                    logger.error(f"Error in episodic memory update: {e}")
                    logger.error(f"Current state shape: {state_tensor.shape}, Flattened: {state_flat.shape}")
                    logger.error(f"Current value shape: {value.shape}")

            state = next_state
            episode_reward += total_reward  # Use total_reward instead of extrinsic_reward

            if done:
                logger.info(f"Episode finished after {step_count} steps")
                break
        
        # Update policy
        try:
            loss_stats = ppo_agent.update()
            logger.info(f"Policy updated - Actor Loss: {loss_stats['actor_loss']:.4f}, Critic Loss: {loss_stats['critic_loss']:.4f}")
        except Exception as e:
            logger.error(f"Error in policy update: {e}")

        if episode % 10 == 0:
            reward_system.update_beta()
            logger.info(f"Updated beta to: {reward_system.beta}")

        episode_rewards.append(episode_reward)
        avg_reward = sum(episode_rewards[-100:]) / len(episode_rewards[-100:])
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            logger.info(f"New best reward achieved: {best_reward}")

        logger.info(f"Episode {episode}: Reward: {episode_reward:.2f}, Avg(100): {avg_reward:.2f}, Beta: {reward_system.beta:.2f}")
        
        if args.log_file:
            log_data(args.log_file, {
                'episode': episode,
                'reward': episode_reward,
                'avg_reward': avg_reward,
                'beta': reward_system.beta,
                'steps': step_count
            })

        if episode % args.save_interval == 0:
            ppo_agent.save(f"ppo_{args.env_name}_{episode}.pth")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MACE-RL')
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

    args = parser.parse_args()
    train(args)
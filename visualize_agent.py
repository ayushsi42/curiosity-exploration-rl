#!/usr/bin/env python3
"""
MACE-RL Agent Visualization Script
==================================
Visualize trained agents playing in GUI environments.

Usage:
    python visualize_agent.py --model-path results/experiment_20250910_003015/models/CartPole-v1_mace_rl_full_seed42_final.pth
    python visualize_agent.py --model-path path/to/model.pth --env CartPole-v1 --episodes 5
    python visualize_agent.py --compare --baseline-path path/to/baseline.pth --mace-path path/to/mace.pth
"""

import argparse
import os
import torch
import gym
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

from mace_rl.agents.ppo import PPO
from mace_rl.modules.episodic_memory import EpisodicMemory
from mace_rl.modules.curiosity import CuriosityModule
from mace_rl.modules.meta_adaptation import MetaAdaptation
from mace_rl.utils.reward_system import HybridRewardSystem
from mace_rl.envs.minigrid_env import make_minigrid_env
from mace_rl.envs.pybullet_env import make_pybullet_env


class AgentVisualizer:
    """Visualize trained agents in GUI environments."""
    
    def __init__(self, model_path, env_name="CartPole-v1", algorithm="mace_rl_full"):
        self.model_path = model_path
        self.env_name = env_name
        self.algorithm = algorithm
        self.env = None
        self.agent = None
        self.components = {}
        
        self._load_environment()
        self._load_model()
    
    def _load_environment(self):
        """Load the environment."""
        print(f"🎮 Loading environment: {self.env_name}")
        
        try:
            if 'MiniGrid' in self.env_name:
                self.env = make_minigrid_env(self.env_name)
            elif any(bullet_env in self.env_name for bullet_env in ['Ant', 'Humanoid', 'Walker']):
                self.env = make_pybullet_env(self.env_name)
            else:
                self.env = gym.make(self.env_name, render_mode='human')
            print(f"✅ Environment loaded successfully")
            print(f"   Observation space: {self.env.observation_space}")
            print(f"   Action space: {self.env.action_space}")
        except Exception as e:
            print(f"❌ Failed to load environment: {e}")
            # Fallback to basic gym environment
            self.env = gym.make(self.env_name, render_mode='human')
    
    def _load_model(self):
        """Load the trained model and components."""
        print(f"🤖 Loading model: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Get environment dimensions
        if isinstance(self.env.observation_space, gym.spaces.Box):
            state_dim = self.env.observation_space.shape
            if len(state_dim) == 1:
                state_dim = state_dim[0]
        else:
            state_dim = self.env.observation_space.n
            
        if isinstance(self.env.action_space, gym.spaces.Box):
            action_dim = self.env.action_space.shape[0]
            has_continuous_action_space = True
        else:
            action_dim = self.env.action_space.n
            has_continuous_action_space = False
        
        # Initialize agent
        self.agent = PPO(
            state_dim, action_dim,
            lr_actor=0.0003, lr_critic=0.001,
            gamma=0.99, K_epochs=4, eps_clip=0.2,
            has_continuous_action_space=has_continuous_action_space
        )
        
        # Load model weights
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.agent.load(self.model_path)
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("🤖 Using random policy instead")
        
        # Initialize MACE-RL components if needed
        if self.algorithm != 'ppo_baseline' and not self.is_atari:
            try:
                state_size = state_dim if isinstance(state_dim, int) else np.prod(state_dim)
                
                self.components['episodic_memory'] = EpisodicMemory(1000, state_size)
                self.components['curiosity_module'] = CuriosityModule(
                    state_size, action_dim, self.components['episodic_memory'],
                    continuous=has_continuous_action_space
                )
                self.components['reward_system'] = HybridRewardSystem(
                    self.components['curiosity_module'], beta_initial=0.1
                )
                print(f"✅ MACE-RL components initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize MACE-RL components: {e}")
    
    def play_episode(self, record=False, save_path=None, evaluation_mode=True):
        """Play one episode with visualization.
        
        Args:
            record: Whether to record frames for video
            save_path: Path to save video
            evaluation_mode: If True, use only extrinsic rewards for fair comparison
        """
        print(f"\n🎬 Playing episode...")
        if evaluation_mode:
            print("📊 Evaluation mode: Using only extrinsic rewards for fair comparison")
        
        frames = []
        episode_reward = 0
        episode_reward_extrinsic = 0  # Track extrinsic separately
        curiosity_rewards = []
        
        # Reset environment
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple):
            state, info = reset_out
        else:
            state = reset_out
        
        done = False
        step_count = 0
        
        while not done and step_count < 1000:  # Prevent infinite episodes
            # Render environment
            if hasattr(self.env, 'render'):
                frame = self.env.render()
                if record and frame is not None:
                    frames.append(frame)
            
            # Get action from agent
            action = self.agent.select_action(state)
            
            # Environment step
            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_state, extrinsic_reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, extrinsic_reward, done, info = step_result
            
            # Calculate curiosity if available
            curiosity_bonus = 0
            total_step_reward = extrinsic_reward  # Default to extrinsic only
            
            if 'reward_system' in self.components:
                try:
                    if isinstance(state, tuple):
                        state_tensor = torch.FloatTensor(state[0])
                    else:
                        state_tensor = torch.FloatTensor(state)
                    state_tensor = state_tensor.view(-1)
                    
                    action_tensor = torch.FloatTensor([action]) if isinstance(action, (int, float)) else torch.FloatTensor(action)
                    
                    total_step_reward = self.components['reward_system'].get_total_reward(
                        state_tensor, action_tensor, extrinsic_reward
                    )
                    curiosity_bonus = total_step_reward - extrinsic_reward
                    curiosity_rewards.append(curiosity_bonus)
                except Exception as e:
                    pass  # Continue without curiosity
            
            state = next_state
            episode_reward_extrinsic += extrinsic_reward  # Always track extrinsic
            
            # Choose what to accumulate based on evaluation mode
            if evaluation_mode:
                episode_reward += extrinsic_reward  # Fair comparison: extrinsic only
            else:
                episode_reward += total_step_reward  # Training-like: total reward
                
            step_count += 1
            
            # Add small delay for better visualization
            time.sleep(0.05)
        
        print(f"✅ Episode completed!")
        print(f"   Total steps: {step_count}")
        
        if evaluation_mode:
            print(f"   📊 Evaluation reward (extrinsic only): {episode_reward:.2f}")
            if curiosity_rewards:
                total_curiosity = sum(curiosity_rewards)
                avg_curiosity = np.mean(curiosity_rewards)
                training_reward = episode_reward_extrinsic + total_curiosity
                print(f"   🎓 Training reward would have been: {training_reward:.2f}")
                print(f"      - Extrinsic: {episode_reward_extrinsic:.2f}")
                print(f"      - Curiosity: {total_curiosity:.2f} (avg: {avg_curiosity:.4f}/step)")
        else:
            print(f"   Total reward: {episode_reward:.2f}")
            if curiosity_rewards:
                avg_curiosity = np.mean(curiosity_rewards)
                total_curiosity = sum(curiosity_rewards)
                extrinsic_total = episode_reward - total_curiosity
                print(f"   Reward breakdown:")
                print(f"     - Extrinsic: {extrinsic_total:.2f}")
                print(f"     - Curiosity: {total_curiosity:.2f} (avg: {avg_curiosity:.4f}/step)")
                print(f"     - Total: {episode_reward:.2f}")
            else:
                print(f"   Reward type: Extrinsic only (no curiosity)")
        
        # Save video if requested
        if record and frames and save_path:
            self._save_video(frames, save_path)
        
        return episode_reward, curiosity_rewards
    
    def _save_video(self, frames, save_path):
        """Save recorded frames as video."""
        print(f"💾 Saving video to: {save_path}")
        
        if len(frames) == 0:
            print("⚠️ No frames to save")
            return
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        video_writer = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"✅ Video saved successfully")
    
    def compare_agents(self, baseline_path, episodes=3, evaluation_mode=True):
        """Compare baseline vs MACE-RL agent.
        
        Args:
            baseline_path: Path to baseline model
            episodes: Number of episodes to compare
            evaluation_mode: If True, use only extrinsic rewards for fair comparison
        """
        print(f"\n🆚 Comparing agents over {episodes} episodes...")
        if evaluation_mode:
            print("📊 Fair evaluation: Both agents evaluated using extrinsic rewards only")
        
        # Load baseline agent
        baseline_visualizer = AgentVisualizer(baseline_path, self.env_name, "ppo_baseline")
        
        baseline_rewards = []
        mace_rewards = []
        
        for episode in range(episodes):
            print(f"\n--- Episode {episode + 1} ---")
            
            # Baseline agent
            print("🤖 Baseline PPO:")
            baseline_reward, _ = baseline_visualizer.play_episode(evaluation_mode=evaluation_mode)
            baseline_rewards.append(baseline_reward)
            
            time.sleep(2)  # Brief pause between agents
            
            # MACE-RL agent
            print("🚀 MACE-RL:")
            mace_reward, _ = self.play_episode(evaluation_mode=evaluation_mode)
            mace_rewards.append(mace_reward)
            
            time.sleep(2)  # Brief pause between episodes
        
        # Print comparison
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"   Baseline PPO - Mean: {np.mean(baseline_rewards):.2f} ± {np.std(baseline_rewards):.2f}")
        print(f"   MACE-RL     - Mean: {np.mean(mace_rewards):.2f} ± {np.std(mace_rewards):.2f}")
        
        improvement = np.mean(mace_rewards) - np.mean(baseline_rewards)
        print(f"   Improvement: {improvement:.2f} ({improvement/np.mean(baseline_rewards)*100:.1f}%)")
    
    def interactive_mode(self):
        """Interactive mode for continuous visualization."""
        print(f"\n🎮 Interactive Mode - Press 'q' to quit")
        print("Commands:")
        print("  'n' - Play next episode")
        print("  'r' - Record next episode")
        print("  'q' - Quit")
        
        episode_count = 0
        
        while True:
            command = input("\nEnter command (n/r/q): ").lower().strip()
            
            if command == 'q':
                print("👋 Goodbye!")
                break
            elif command == 'n':
                episode_count += 1
                print(f"\n🎬 Episode {episode_count}")
                self.play_episode()
            elif command == 'r':
                episode_count += 1
                save_path = f"episode_{episode_count}_{self.algorithm}.mp4"
                print(f"\n🎬📹 Recording Episode {episode_count}")
                self.play_episode(record=True, save_path=save_path)
            else:
                print("❌ Invalid command. Use 'n', 'r', or 'q'")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MACE-RL trained agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic visualization
  python visualize_agent.py --model-path results/experiment_20250910_003015/models/CartPole-v1_mace_rl_full_seed42_final.pth
  
  # Specify environment and number of episodes
  python visualize_agent.py --model-path path/to/model.pth --env CartPole-v1 --episodes 5
  
  # Compare two agents
  python visualize_agent.py --compare --baseline-path path/to/baseline.pth --mace-path path/to/mace.pth
  
  # Interactive mode
  python visualize_agent.py --model-path path/to/model.pth --interactive
  
  # Record episodes
  python visualize_agent.py --model-path path/to/model.pth --record --save-dir ./videos/
        """
    )
    
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--algorithm', type=str, default='mace_rl_full', 
                       choices=['ppo_baseline', 'mace_rl_full', 'mace_rl_no_memory', 'mace_rl_no_curiosity'],
                       help='Algorithm type')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to play')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--record', action='store_true', help='Record episodes as videos')
    parser.add_argument('--save-dir', type=str, default='./videos/', help='Directory to save videos')
    parser.add_argument('--evaluation-mode', action='store_true', default=True, 
                       help='Use only extrinsic rewards for fair evaluation (default: True)')
    parser.add_argument('--training-mode', action='store_true', 
                       help='Use total rewards (extrinsic + curiosity) like during training')
    
    # Comparison mode
    parser.add_argument('--compare', action='store_true', help='Compare two agents')
    parser.add_argument('--baseline-path', type=str, help='Path to baseline model')
    parser.add_argument('--mace-path', type=str, help='Path to MACE-RL model')
    
    args = parser.parse_args()
    
    # Determine evaluation mode
    evaluation_mode = not args.training_mode  # Default to evaluation mode unless training_mode specified
    
    if args.compare:
        if not args.baseline_path or not args.mace_path:
            print("❌ For comparison mode, provide both --baseline-path and --mace-path")
            return
        
        visualizer = AgentVisualizer(args.mace_path, args.env, 'mace_rl_full')
        visualizer.compare_agents(args.baseline_path, args.episodes, evaluation_mode)
        
    elif args.model_path:
        visualizer = AgentVisualizer(args.model_path, args.env, args.algorithm)
        
        if args.interactive:
            visualizer.interactive_mode()
        else:
            # Create save directory if recording
            if args.record:
                os.makedirs(args.save_dir, exist_ok=True)
            
            total_rewards = []
            for episode in range(args.episodes):
                print(f"\n🎬 Episode {episode + 1}/{args.episodes}")
                
                save_path = None
                if args.record:
                    save_path = os.path.join(args.save_dir, f"episode_{episode+1}_{args.algorithm}.mp4")
                
                reward, _ = visualizer.play_episode(record=args.record, save_path=save_path, evaluation_mode=evaluation_mode)
                total_rewards.append(reward)
            
            print(f"\n📊 SUMMARY:")
            print(f"   Episodes played: {len(total_rewards)}")
            print(f"   Mean reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
            print(f"   Best episode: {max(total_rewards):.2f}")
    else:
        print("❌ Please provide --model-path or use --compare mode")
        parser.print_help()


if __name__ == "__main__":
    main()

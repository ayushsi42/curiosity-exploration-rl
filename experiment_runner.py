#!/usr/bin/env python3
"""
SIMPLIFIED MACE-RL EXPERIMENT RUNNER
====================================

Features:
✅ Multiple algorithm comparisons (PPO baseline, MACE-RL variants)
✅ Multiple environments (Basic gym environments)
✅ Simplified ablation studies (memory size, beta values, beta schedulers)
✅ Statistical analysis with significance testing
✅ Publication-quality plots and visualizations
✅ Automated report generation
✅ Model checkpointing and result organization
✅ Parallel execution support
✅ Comprehensive logging and monitoring

Usage Examples:
    # Run full experiment suite
    python experiment_runner.py --run-experiments --run-ablations --analyze-results
    
    # Quick test mode
    python experiment_runner.py --quick-test
    
    # Specific ablation studies
    python experiment_runner.py --run-ablations --ablation-type memory_size
    
    # Publication mode (extended runs with more seeds)
    python experiment_runner.py --publication-mode
"""

import warnings
# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, message='.*missing from font.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Mean of empty slice.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*labels.*parameter.*boxplot.*')
warnings.filterwarnings('ignore', message='.*Polyfit may be poorly conditioned.*')

import argparse
import os
import json
import pickle
import subprocess
import time
import logging
import warnings
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import itertools
import numpy as np
import torch
import gym
from scipy import stats
from tqdm import tqdm

# Suppress gym deprecation warnings for np.bool8
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

# Import MACE-RL components
from mace_rl.utils.utils import set_seed, log_data
from mace_rl.utils.logger import setup_logger
from mace_rl.agents.ppo import PPO
from mace_rl.modules.episodic_memory import EpisodicMemory
from mace_rl.modules.curiosity import CuriosityModule
from mace_rl.utils.reward_system import HybridRewardSystem
from mace_rl.utils.beta_schedulers import create_beta_scheduler
from mace_rl.envs.minigrid_env import make_minigrid_env
from mace_rl.envs.pybullet_env import make_pybullet_env

# Global configuration for experiments
EXPERIMENT_CONFIG = {
    'environments': [
        # ✅ ACTIVE ENVIRONMENTS (4 total)
        # Basic gym environments (no additional dependencies required)
        'CartPole-v1',                    # Simple control task
        'Acrobot-v1',                     # More complex control task
        # # MiniGrid environments (require: pip install gym-minigrid or minigrid)  
        # # 'HopperBulletEnv-v0',       # Navigation with key-door mechanics
        # # PyBullet environments (require: pip install pybullet)
        'HalfCheetahBulletEnv-v0',       # Continuous control locomotion
        
        # # 🚫 DISABLED ENVIRONMENTS
        # # 'MountainCar-v0',               # Removed - replaced with more diverse set
        # # Atari environments require: pip install ale_py
        # # 'PongNoFrameskip-v4',
        # # Additional MiniGrid options
        # # 'HopperBulletEnv-v0',
        # 'MiniGrid-FourRooms-v0',
        # 'MiniGrid-MultiRoom-N2-S4-v0',
        # Additional PyBullet options
        # 'AntBulletEnv-v0',
        'HopperBulletEnv-v0'
    ],
    'algorithms': [
        # 'ppo_baseline',
        'mace_rl_full',
        # 'mace_rl_no_memory',
        # 'mace_rl_no_curiosity',
        # 'mace_rl_no_meta',
        # 'mace_rl_curiosity_only',
        # 'mace_rl_memory_only'
    ],
    'seeds': [42],  # Single seed for all experiments
    'max_episodes': 2000,
    'max_timesteps': 10000,
    # Note: save_interval removed - only save final models
    
    # Environment-specific episode configurations
    'env_specific_episodes': {
    'CartPole-v1': 1000,
    'Acrobot-v1': 2000,
    'HopperBulletEnv-v0': 3000,
    'HalfCheetahBulletEnv-v0': 8000,
    },

    'env_specific_timesteps': {
    'CartPole-v1': 500,
    'Acrobot-v1': 500,
    'HopperBulletEnv-v0': 1000,
    'HalfCheetahBulletEnv-v0': 1000,
    },
    
    # SIMPLIFIED ABLATION STUDY CONFIGURATIONS
    # --- Meta network usage flag ---
    'use_meta_network_for_beta': False,  # If True, use meta network for beta; else use scheduler
    'ablation_studies': {
        # Memory ablations
        'memory_sizes': [200, 500, 1000],
        
        # Curiosity ablations  
        'beta_values': [0.75, 1.0, 1.5],
        
        # Simplified beta scheduler ablations (reduced from 8 to 3)
        'beta_schedulers': [
            'constant',
            'linear_decay', 
            'exponential_decay'
        ]
    },
    
    # 🎯 PUBLICATION MODE SETTINGS
    'publication_mode': {
        'seeds': [42],  # Single seed for publication experiments too
        'max_episodes': 2000,
        'max_timesteps': 20000,
        'validation_episodes': 100
    }
}

# Helper function for defaultdict to avoid pickle issues
def create_nested_dict():
    return defaultdict(list)

import numpy as np

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class ExperimentRunner:
    """Main class for running and analyzing RL experiments."""
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.suite_name = f'mace_rl_suite_{self.timestamp}'
        self.experiment_dir = os.path.join(results_dir, self.suite_name)
        
        # Create main suite directory structure 
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'individual_runs'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'suite_analysis'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, 'suite_logs'), exist_ok=True)
        
        # Setup suite-level logging
        self.logger = setup_logger(
            'experiment_suite',
            os.path.join(self.experiment_dir, 'suite_logs', 'suite.log'),
            level=logging.INFO,
            console_output=True
        )
        
        self.results = defaultdict(create_nested_dict)
        
        # Validate environments on startup
        self._validate_environments()
        
    def _validate_environments(self):
        """Validate that all configured environments can be loaded."""
        self.logger.info("Validating configured environments...")
        failed_envs = []
        
        for env_name in EXPERIMENT_CONFIG['environments']:
            try:
                if "NoFrameskip" in env_name:
                    env = make_atari_env(env_name)
                    env = wrap_deepmind(env, frame_stack=True, clip_rewards=True)
                elif "MiniGrid" in env_name:
                    env = make_minigrid_env(env_name)
                elif "Bullet" in env_name:
                    env = make_pybullet_env(env_name)
                else:
                    env = gym.make(env_name)
                
                # Quick test to ensure environment works
                _ = env.reset()
                _ = env.action_space.sample()
                env.close()
                self.logger.info(f"✅ Environment {env_name} validated successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Environment {env_name} failed validation: {e}")
                failed_envs.append(env_name)
        
        if failed_envs:
            self.logger.warning(f"Failed environments: {failed_envs}")
            self.logger.warning("These environments will be skipped during experiments")
        else:
            self.logger.info("✅ All environments validated successfully!")
        
    def run_single_experiment(self, env_name, algorithm, seed, episode_limit=None, ablation_config=None):
        """Run a single experiment configuration."""
        # Create comprehensive experiment ID that includes all configuration parameters
        exp_id_parts = [env_name, algorithm, f"seed{seed}"]
        
        if ablation_config:
            # Add all ablation parameters to ensure unique naming
            config_parts = []
            for key, value in sorted(ablation_config.items()):  # Sort for consistent ordering
                if isinstance(value, dict):
                    # Handle nested configs like beta_scheduler
                    if key == 'beta_scheduler' and 'scheduler_type' in value:
                        config_parts.append(f"sched{value['scheduler_type']}")
                        # Add scheduler-specific parameters
                        for sub_key, sub_value in sorted(value.items()):
                            if sub_key != 'scheduler_type':
                                config_parts.append(f"{sub_key}{sub_value}")
                    else:
                        # Flatten other nested configs
                        for sub_key, sub_value in sorted(value.items()):
                            config_parts.append(f"{sub_key}{sub_value}")
                else:
                    # Simple key-value pairs
                    config_parts.append(f"{key}{value}")
            
            if config_parts:
                exp_id_parts.append("_".join(config_parts))
        
        exp_id = "_".join(exp_id_parts)
        
        # Sanitize experiment ID to avoid filesystem issues
        exp_id = exp_id.replace(".", "p")  # Replace decimal points
        exp_id = exp_id.replace("-", "n")  # Replace negative signs
        exp_id = exp_id[:200]  # Limit length to avoid filesystem issues
        
        self.logger.info(f"Starting experiment: {exp_id}")
        
        # Set random seed
        set_seed(seed)
        
        # Initialize environment with error handling
        try:
            if "NoFrameskip" in env_name:
                env = make_atari_env(env_name)
                env = wrap_deepmind(env, frame_stack=True, clip_rewards=True)
                has_continuous_action_space = False
                is_atari = True
            elif "MiniGrid" in env_name:
                env = make_minigrid_env(env_name)
                has_continuous_action_space = False
                is_atari = False
            elif "Bullet" in env_name:
                env = make_pybullet_env(env_name)
                has_continuous_action_space = True
                is_atari = False
            else:
                # Basic gym environments (CartPole, Acrobot, etc.)
                env = gym.make(env_name)
                has_continuous_action_space = hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0
                is_atari = False
        except Exception as e:
            self.logger.error(f"Failed to create environment {env_name}: {e}")
            if "NoFrameskip" in env_name:
                self.logger.error("Atari environments require ale_py. Install with: pip install ale_py")
            elif "MiniGrid" in env_name:
                self.logger.error("MiniGrid environments require gym-minigrid. Install with: pip install gym-minigrid")
                self.logger.error("Alternative: pip install minigrid")
            elif "Bullet" in env_name:
                self.logger.error("PyBullet environments require pybullet. Install with: pip install pybullet")
            else:
                self.logger.error(f"Basic gym environment {env_name} failed to load. Check if gym is properly installed.")
            return {
                'config': {'env_name': env_name, 'algorithm': algorithm, 'seed': seed, 'exp_id': exp_id},
                'error': str(e),
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'final_reward': 0.0,
                'episode_rewards': [],
                'timesteps': []
            }
        
        state_dim = env.observation_space.shape
        # For 1D observations, pass the dimension as an integer
        if len(state_dim) == 1:
            state_dim = state_dim[0]
        
        if has_continuous_action_space:
            action_dim = env.action_space.shape[0]
        else:
            action_dim = env.action_space.n
        
        # Initialize algorithm components
        components = self._initialize_algorithm_components(
            algorithm, state_dim, action_dim, has_continuous_action_space, is_atari, ablation_config
        )

        # --- MetaAdaptation integration (controlled by config flag) ---
        meta_net = None
        meta_beta_history = []
        meta_beta = None
        use_meta_net = EXPERIMENT_CONFIG.get('use_meta_network_for_beta', False)
        if algorithm == 'mace_rl_full' and not is_atari and use_meta_net:
            from mace_rl.modules.meta_adaptation import MetaAdaptation
            # Example input_dim: state_dim + action_dim + 2 (reward, curiosity)
            meta_input_dim = (state_dim if isinstance(state_dim, int) else state_dim[0]) + action_dim + 2
            meta_hidden_dim = 128
            meta_output_dim = 1
            meta_net = MetaAdaptation(meta_input_dim, meta_hidden_dim, meta_output_dim)
        
        # Training loop
        episode_rewards = []  # For plotting - will store EXTRINSIC rewards only
        episode_training_rewards = []  # For training metrics - will store TOTAL rewards (extrinsic + curiosity)
        episode_steps = []
        curiosity_rewards = []
        total_steps = 0  # Track total steps across all episodes for beta scheduler
        
        # Use environment-specific episodes if available, otherwise use global or episode_limit
        if episode_limit:
            max_eps = episode_limit
        else:
            max_eps = EXPERIMENT_CONFIG['env_specific_episodes'].get(env_name, EXPERIMENT_CONFIG['max_episodes'])
        
        # Create progress bar for episodes (only if not in multiprocessing)
        try:
            # Check if we're in a multiprocessing context
            import multiprocessing
            if multiprocessing.current_process().name == 'MainProcess':
                # Clean experiment header (only for main process with tqdm)
                print(f"\n{'='*60}")
                print(f"🚀 EXPERIMENT: {exp_id}")
                print(f"   Environment: {env_name} | Algorithm: {algorithm} | Seed: {seed}")
                if ablation_config:
                    print(f"   Ablation: {ablation_config}")
                print(f"{'='*60}")
                
                # Suppress debug logging for components to avoid tqdm interference
                logging.getLogger('HybridRewardSystem').setLevel(logging.WARNING)
                logging.getLogger('CuriosityModule').setLevel(logging.WARNING)
                logging.getLogger('EpisodicMemory').setLevel(logging.WARNING)
                logging.getLogger('PPO').setLevel(logging.WARNING)
                
                pbar = tqdm(range(max_eps), desc=f"Training {algorithm}", 
                           unit="episode", ncols=100, 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
                use_pbar = True
            else:
                use_pbar = False
                pbar = None
        except:
            use_pbar = False
            pbar = None
        
        episode_range = pbar if use_pbar else range(max_eps)
        for episode in episode_range:
            
            # Reset environment
            reset_out = env.reset()
            if isinstance(reset_out, tuple):
                state, info = reset_out
            else:
                state = reset_out
            
            done = False
            episode_extrinsic_reward = 0  # For plotting - pure environment reward
            episode_total_reward = 0      # For training metrics - includes curiosity
            episode_curiosity = 0
            step_count = 0
            
            meta_beta_value = None
            meta_input_seq = []
            for t in range(EXPERIMENT_CONFIG['env_specific_timesteps'].get(env_name, EXPERIMENT_CONFIG['max_timesteps'])):
                step_count += 1
                total_steps += 1  # Track total steps across episodes
                action = components['agent'].select_action(state)
                
                # Environment step
                next_state, extrinsic_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Calculate total reward
                total_reward = extrinsic_reward
                curiosity_bonus = 0
                # --- MetaAdaptation: collect input for meta_net (if enabled) ---
                if meta_net is not None:
                    # Prepare meta input: [state, action(one-hot if discrete), extrinsic_reward, curiosity_bonus]
                    if isinstance(state, tuple):
                        state_tensor = torch.FloatTensor(state[0])
                    else:
                        state_tensor = torch.FloatTensor(state)
                    state_tensor = state_tensor.view(-1)
                    if has_continuous_action_space:
                        action_tensor = torch.FloatTensor(action)
                    else:
                        # One-hot encode discrete action
                        action_tensor = torch.zeros(action_dim)
                        action_tensor[int(action)] = 1.0
                    meta_input = torch.cat([
                        state_tensor,
                        action_tensor,
                        torch.tensor([extrinsic_reward], dtype=torch.float32),
                        torch.tensor([curiosity_bonus], dtype=torch.float32)
                    ])
                    meta_input_seq.append(meta_input)

                if components['curiosity_module'] and components['reward_system']:
                    try:
                        # Prepare state and action tensors
                        if isinstance(state, tuple):
                            state_tensor = torch.FloatTensor(state[0])
                        else:
                            state_tensor = torch.FloatTensor(state)
                        state_tensor = state_tensor.view(-1)
                        action_tensor = torch.FloatTensor(action) if has_continuous_action_space else torch.LongTensor([action])
                        # Get curiosity-adjusted reward
                        total_reward = components['reward_system'].get_total_reward(
                            state_tensor, action_tensor, extrinsic_reward
                        )
                        curiosity_bonus = total_reward - extrinsic_reward
                    except Exception as e:
                        self.logger.warning(f"Curiosity calculation failed: {e}")
                        total_reward = extrinsic_reward
                
                # Store experience
                components['agent'].buffer.rewards.append(total_reward)
                components['agent'].buffer.is_terminals.append(done)
                
                # Update episodic memory
                if components['episodic_memory']:
                    try:
                        # Prepare state tensors
                        if isinstance(state, tuple):
                            state_tensor = torch.FloatTensor(state[0])
                        else:
                            state_tensor = torch.FloatTensor(state)
                        
                        if isinstance(next_state, tuple):
                            next_state_tensor = torch.FloatTensor(next_state[0])
                        else:
                            next_state_tensor = torch.FloatTensor(next_state)
                        
                        state_flat = state_tensor.view(-1)
                        next_state_flat = next_state_tensor.view(-1)
                        value = torch.cat([next_state_flat, torch.FloatTensor([total_reward])])
                        
                        components['episodic_memory'].add(state_flat, value)
                    except Exception as e:
                        self.logger.warning(f"Memory update failed: {e}")
                
                state = next_state
                episode_extrinsic_reward += extrinsic_reward  # Track pure environment reward for plotting
                episode_total_reward += total_reward         # Track total reward for training metrics
                episode_curiosity += curiosity_bonus
                
                if done:
                    break
            
            # --- MetaAdaptation: after episode, update beta using meta_net (if enabled) ---
            if meta_net is not None and len(meta_input_seq) > 0:
                meta_seq_tensor = torch.stack(meta_input_seq).unsqueeze(0)  # (1, seq_len, input_dim)
                with torch.no_grad():
                    meta_beta_out, _ = meta_net(meta_seq_tensor)
                    # meta_beta_out shape: [1, output_dim] (output_dim=1)
                    meta_beta_value = float(torch.sigmoid(meta_beta_out[0, 0]).item())
                meta_beta_history.append(meta_beta_value)
                # Set beta in reward system directly
                if components['reward_system']:
                    components['reward_system'].beta = meta_beta_value
                    if hasattr(components['reward_system'], 'beta_scheduler'):
                        components['reward_system'].beta_scheduler.beta = meta_beta_value
            else:
                meta_beta_history.append(None)

            # Update policy
            try:
                loss_stats = components['agent'].update()
            except Exception as e:
                self.logger.warning(f"Policy update failed: {e}")

            # Update beta scheduler for curiosity (if not using meta_net)
            if (meta_net is None or not use_meta_net) and episode % 10 == 0 and components['reward_system']:
                components['reward_system'].update_beta(episode=episode, step=total_steps)
            
            # Store episode data
            episode_rewards.append(episode_extrinsic_reward)  # Store extrinsic rewards for plotting
            episode_training_rewards.append(episode_total_reward)  # Store total rewards for training metrics
            episode_steps.append(step_count)
            curiosity_rewards.append(episode_curiosity)
            
            # Update progress bar or log progress
            if use_pbar and pbar:
                if len(episode_rewards) >= 10:
                    recent_avg = np.mean(episode_rewards[-10:])  # Show extrinsic reward average
                    pbar.set_postfix({"Extrinsic": f"{recent_avg:.2f}", "Total": f"{episode_total_reward:.2f}"})
                else:
                    pbar.set_postfix({"Extrinsic": f"{episode_extrinsic_reward:.2f}", "Total": f"{episode_total_reward:.2f}"})
            # No print statements when using tqdm to avoid interference
            
            # Note: Model saving removed from training loop - only save final model at the end
        
        # Close progress bar if it was used
        if use_pbar and pbar:
            pbar.close()
        
        # Save final model
        final_model_path = os.path.join(
            self.experiment_dir, 'models', f"{exp_id}_final.pth"
        )
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        components['agent'].save(final_model_path)
        
        # Store results
        result_data = {
            'episode_rewards': episode_rewards,  # Extrinsic rewards for plotting
            'episode_training_rewards': episode_training_rewards,  # Total rewards for training metrics
            'episode_steps': episode_steps,
            'curiosity_rewards': curiosity_rewards,
            'meta_beta_history': meta_beta_history if meta_net is not None else None,
            'final_reward': episode_rewards[-1] if episode_rewards else 0,  # Use extrinsic for final
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,  # Use extrinsic for mean
            'std_reward': np.std(episode_rewards) if episode_rewards else 0,   # Use extrinsic for std
            'final_training_reward': episode_training_rewards[-1] if episode_training_rewards else 0,  # Total for training
            'mean_training_reward': np.mean(episode_training_rewards) if episode_training_rewards else 0,  # Total for training
            'std_training_reward': np.std(episode_training_rewards) if episode_training_rewards else 0,   # Total for training
            'config': {
                'env_name': env_name,
                'algorithm': algorithm,
                'seed': seed,
                'max_episodes': max_eps,
                'ablation_config': ablation_config,
                'exp_id': exp_id
            }
        }
        
        # 🎛️ Add beta scheduler information if available
        if components['reward_system'] and hasattr(components['reward_system'], 'get_scheduler_info'):
            try:
                scheduler_info = components['reward_system'].get_scheduler_info()
                result_data['beta_scheduler_info'] = scheduler_info
                
                # Track beta values over time if we can extract them
                if hasattr(components['reward_system'].beta_scheduler, 'step_count'):
                    result_data['beta_evolution'] = {
                        'final_beta': scheduler_info['current_beta'],
                        'scheduler_type': scheduler_info['scheduler_info']['scheduler_type']
                    }
            except Exception as e:
                self.logger.warning(f"Failed to extract beta scheduler info: {e}")
        
        # Save experiment data
        data_path = os.path.join(self.experiment_dir, 'data', f"{exp_id}.pkl")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'wb') as f:
            pickle.dump(result_data, f)
        
        # 🎯 NEW: Immediate evaluation after training
        self.logger.info(f"Starting immediate evaluation for {exp_id}")
        eval_metrics = self.evaluate_trained_model(
            env_name, algorithm, seed, final_model_path, 
            num_eval_episodes=10, ablation_config=ablation_config
        )
        
        # 🎯 NEW: Create structured results directory for this experiment
        experiment_results_dir = os.path.join(self.experiment_dir, 'individual_runs', exp_id)
        os.makedirs(experiment_results_dir, exist_ok=True)
        
        # Save comprehensive metadata for this specific experiment
        comprehensive_metadata = {
            'experiment_id': exp_id,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'environment': env_name,
                'algorithm': algorithm,
                'seed': seed,
                'max_episodes': max_eps,
                'ablation_config': ablation_config or {}
            },
            'training_results': {
                'episode_rewards': episode_rewards,
                'episode_training_rewards': episode_training_rewards,
                'episode_steps': episode_steps,
                'curiosity_rewards': curiosity_rewards,
                'meta_beta_history': meta_beta_history if meta_net is not None else None,
                'final_reward': result_data['final_reward'],
                'mean_reward': result_data['mean_reward'],
                'std_reward': result_data['std_reward'],
                'final_training_reward': result_data['final_training_reward'],
                'mean_training_reward': result_data['mean_training_reward'],
                'std_training_reward': result_data['std_training_reward']
            },
            'evaluation_results': eval_metrics,
            'model_paths': {
                'final_model': final_model_path,
                'experiment_data': data_path
            },
            'performance_analysis': {
                'convergence_episode': self._analyze_convergence(episode_rewards),
                'stability_score': self._calculate_stability_score(episode_rewards),
                'sample_efficiency': self._calculate_sample_efficiency(episode_rewards),
                'exploration_effectiveness': np.mean(curiosity_rewards) if curiosity_rewards else 0.0
            }
        }
        
        # Add beta scheduler info if available
        if 'beta_scheduler_info' in result_data:
            comprehensive_metadata['beta_scheduler_info'] = result_data['beta_scheduler_info']
        if 'beta_evolution' in result_data:
            comprehensive_metadata['beta_evolution'] = result_data['beta_evolution']
        
        # Save comprehensive metadata as JSON
        metadata_path = os.path.join(experiment_results_dir, 'metadata.json')

        with open(metadata_path, 'w') as f:
            json.dump(comprehensive_metadata, f, indent=2, default=convert_numpy)
        
        # Save training data as pickle for detailed analysis
        training_data_path = os.path.join(experiment_results_dir, 'training_data.pkl')
        with open(training_data_path, 'wb') as f:
            pickle.dump(result_data, f)
        
        # Save evaluation results separately
        if eval_metrics:
            eval_path = os.path.join(experiment_results_dir, 'evaluation_results.json')
            with open(eval_path, 'w') as f:
                json.dump(eval_metrics, f, indent=2, default=convert_numpy)
        
        # Create experiment summary text file
        summary_path = os.path.join(experiment_results_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"MACE-RL Experiment Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Experiment ID: {exp_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Environment: {env_name}\n")
            f.write(f"Algorithm: {algorithm}\n")
            f.write(f"Seed: {seed}\n")
            if ablation_config:
                f.write(f"Ablation Config: {ablation_config}\n")
            f.write(f"\nTraining Results:\n")
            f.write(f"  Episodes: {len(episode_rewards)}\n")
            f.write(f"  Mean Extrinsic Reward: {result_data['mean_reward']:.4f} ± {result_data['std_reward']:.4f}\n")
            f.write(f"  Final Extrinsic Reward: {result_data['final_reward']:.4f}\n")
            f.write(f"  Mean Training Reward: {result_data['mean_training_reward']:.4f} ± {result_data['std_training_reward']:.4f}\n")
            if curiosity_rewards and any(curiosity_rewards):
                avg_curiosity = np.mean([c for c in curiosity_rewards if c != 0])
                f.write(f"  Average Curiosity Bonus: {avg_curiosity:.6f}\n")
            
            if eval_metrics:
                f.write(f"\nEvaluation Results:\n")
                f.write(f"  Episodes: {eval_metrics['total_episodes']}\n")
                f.write(f"  Mean Reward: {eval_metrics['mean_reward']:.4f} ± {eval_metrics['std_reward']:.4f}\n")
                f.write(f"  Min/Max Reward: {eval_metrics['min_reward']:.4f} / {eval_metrics['max_reward']:.4f}\n")
                f.write(f"  Median Reward: {eval_metrics['median_reward']:.4f}\n")
                f.write(f"  Mean Steps per Episode: {eval_metrics['mean_steps']:.1f}\n")
        
        # Update result_data with evaluation metrics for backward compatibility
        if eval_metrics:
            result_data['evaluation_metrics'] = eval_metrics
            result_data['experiment_directory'] = experiment_results_dir
        
        # Clean experiment summary (only for main process to avoid interference with tqdm)
        if use_pbar:
            print(f"\n✅ COMPLETED: {exp_id}")
            print(f"   Extrinsic Reward (for evaluation): {result_data['mean_reward']:.2f} ± {result_data['std_reward']:.2f}")
            print(f"   Training Reward (extrinsic + curiosity): {result_data['mean_training_reward']:.2f} ± {result_data['std_training_reward']:.2f}")
            if eval_metrics:
                print(f"   Evaluation Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            print(f"   Total Episodes: {len(episode_rewards)}")
            if curiosity_rewards and any(curiosity_rewards):
                avg_curiosity = np.mean([c for c in curiosity_rewards if c != 0])
                print(f"   Avg Curiosity Bonus: {avg_curiosity:.4f}")
            print(f"   Results saved to: {experiment_results_dir}")
            print(f"{'='*60}\n")
        
        return result_data
    
    def evaluate_trained_model(self, env_name, algorithm, seed, model_path, num_eval_episodes=10, ablation_config=None):
        """Evaluate a trained model and return evaluation metrics."""
        exp_id = f"{env_name}_{algorithm}_seed{seed}"
        if ablation_config:
            ablation_suffix = "_".join([f"{k}{v}" for k, v in ablation_config.items()])
            exp_id += f"_ablation_{ablation_suffix}"
        
        self.logger.info(f"Evaluating model: {exp_id}")
        
        # Set random seed for evaluation
        set_seed(seed + 1000)  # Different seed for evaluation
        
        try:
            # Initialize environment (same as training)
            if "NoFrameskip" in env_name:
                env = make_atari_env(env_name)
                env = wrap_deepmind(env, frame_stack=True, clip_rewards=True)
                has_continuous_action_space = False
                is_atari = True
            elif "MiniGrid" in env_name:
                env = make_minigrid_env(env_name)
                has_continuous_action_space = False
                is_atari = False
            elif "Bullet" in env_name:
                env = make_pybullet_env(env_name)
                has_continuous_action_space = True
                is_atari = False
            else:
                env = gym.make(env_name)
                has_continuous_action_space = hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0
                is_atari = False
                
            state_dim = env.observation_space.shape
            if len(state_dim) == 1:
                state_dim = state_dim[0]
            
            if has_continuous_action_space:
                action_dim = env.action_space.shape[0]
            else:
                action_dim = env.action_space.n
            
            # Initialize algorithm components (same as training)
            components = self._initialize_algorithm_components(
                algorithm, state_dim, action_dim, has_continuous_action_space, is_atari, ablation_config
            )
            
            # Load the trained model
            try:
                components['agent'].load(model_path)
                self.logger.info(f"Loaded model from: {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load model {model_path}: {e}")
                return None
            
            # Run evaluation episodes
            eval_rewards = []
            eval_steps = []
            
            for episode in tqdm(range(num_eval_episodes), desc=f"Evaluating {exp_id}", leave=False):
                reset_out = env.reset()
                if isinstance(reset_out, tuple):
                    state, info = reset_out
                else:
                    state = reset_out
                
                done = False
                episode_reward = 0
                episode_step_count = 0
                
                for t in range(EXPERIMENT_CONFIG['max_timesteps']):
                    episode_step_count += 1
                    
                    # Use the agent to select action (try eval_mode, fallback to normal)
                    try:
                        action = components['agent'].select_action(state, eval_mode=True)
                    except TypeError:
                        # Agent doesn't support eval_mode parameter
                        action = components['agent'].select_action(state)
                    
                    # Environment step
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward  # Only extrinsic reward for evaluation
                    state = next_state
                    
                    if done:
                        break
                
                eval_rewards.append(episode_reward)
                eval_steps.append(episode_step_count)
            
            # Calculate evaluation metrics
            eval_metrics = {
                'mean_reward': np.mean(eval_rewards),
                'std_reward': np.std(eval_rewards),
                'min_reward': np.min(eval_rewards),
                'max_reward': np.max(eval_rewards),
                'median_reward': np.median(eval_rewards),
                'mean_steps': np.mean(eval_steps),
                'total_episodes': num_eval_episodes,
                'eval_rewards': eval_rewards,
                'eval_steps': eval_steps,
                'exp_id': exp_id,
                'model_path': model_path
            }
            
            env.close()
            return eval_metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed for {exp_id}: {e}")
            return None
    
    def _analyze_convergence(self, episode_rewards, window_size=50):
        """Analyze when the algorithm converged."""
        if len(episode_rewards) < window_size * 2:
            return len(episode_rewards)  # Not enough data
        
        # Calculate moving average
        rewards = np.array(episode_rewards)
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        
        # Find convergence point (where improvement becomes minimal)
        for i in range(window_size, len(moving_avg) - window_size):
            if i > len(moving_avg) * 0.3:  # Look after 30% of training
                slope = (moving_avg[i + window_size] - moving_avg[i]) / window_size
                if abs(slope) < 0.05:  # Convergence threshold
                    return i
        
        return len(episode_rewards)  # No convergence detected
    
    def _calculate_stability_score(self, episode_rewards, window_size=50):
        """Calculate stability score based on variance in final episodes."""
        if len(episode_rewards) < window_size:
            return 0.0
        
        final_rewards = episode_rewards[-window_size:]
        mean_reward = np.mean(final_rewards)
        if mean_reward == 0:
            return 0.0
        
        # Stability is inversely related to coefficient of variation
        cv = np.std(final_rewards) / abs(mean_reward)
        stability = max(0.0, 1.0 - cv)
        return stability
    
    def _calculate_sample_efficiency(self, episode_rewards, target_percentile=75):
        """Calculate sample efficiency (episodes to reach target performance)."""
        if len(episode_rewards) < 10:
            return len(episode_rewards)
        
        # Define target as percentile of final performance
        final_performance = np.mean(episode_rewards[-10:])
        target = final_performance * (target_percentile / 100.0)
        
        # Find first episode to reach target
        for i, reward in enumerate(episode_rewards):
            if reward >= target:
                return i + 1
        
        return len(episode_rewards)  # Never reached target
    
    def _calculate_state_size(self, state_dim):
        """Calculate state size from state dimension."""
        if isinstance(state_dim, tuple):
            if len(state_dim) == 3:
                return state_dim[0] * state_dim[1] * state_dim[2]
            else:
                return state_dim[0]
        else:
            return state_dim
    
    def _initialize_algorithm_components(self, algorithm, state_dim, action_dim, has_continuous_action_space, is_atari, ablation_config=None):
        """Initialize algorithm-specific components."""
        components = {
            'agent': None,
            'episodic_memory': None,
            'curiosity_module': None,
            'reward_system': None
        }
        
        # Set default hyperparameters or use ablation values
        memory_size = ablation_config.get('memory_size', 1000) if ablation_config else 1000
        # Robust beta extraction: check for all possible keys
        beta_keys = ['beta_initial', 'beta', 'beta_value']
        beta_val = None
        if ablation_config:
            for k in beta_keys:
                if k in ablation_config:
                    beta_val = ablation_config[k]
                    break
        if beta_val is None:
            beta_val = 0.4  # Default changed from 0.4 to 0.6
        # For legacy code, keep beta_initial for most schedulers
        beta_initial = beta_val
        lr_actor = ablation_config.get('lr_actor', 0.0003) if ablation_config else 0.0003
        lr_critic = ablation_config.get('lr_critic', 0.001) if ablation_config else 0.001
        hidden_size = ablation_config.get('hidden_size', 128) if ablation_config else 128
        gamma = ablation_config.get('gamma', 0.99) if ablation_config else 0.99
        curiosity_lr_mult = ablation_config.get('curiosity_lr_mult', 1.0) if ablation_config else 1.0

        # Simplified beta scheduler configuration
        beta_scheduler_config = None
        if ablation_config and 'beta_scheduler' in ablation_config:
            # Only support direct scheduler type specification
            # If scheduler is a string, pass as is
            if isinstance(ablation_config['beta_scheduler'], str):
                beta_scheduler_config = ablation_config['beta_scheduler']
            # If dict, patch in correct beta key
            elif isinstance(ablation_config['beta_scheduler'], dict):
                beta_scheduler_config = ablation_config['beta_scheduler'].copy()
                # Patch beta value for constant scheduler
                if beta_scheduler_config.get('scheduler_type', '') == 'constant':
                    beta_scheduler_config['beta_value'] = beta_val
                else:
                    beta_scheduler_config['beta_initial'] = beta_val
            else:
                beta_scheduler_config = ablation_config['beta_scheduler']
        # If not using a scheduler, pass beta_initial as before
        
        # Initialize PPO agent (common to all algorithms)
        components['agent'] = PPO(
            state_dim, action_dim, 
            lr_actor=lr_actor, lr_critic=lr_critic,
            gamma=gamma, K_epochs=4, eps_clip=0.2,
            has_continuous_action_space=has_continuous_action_space
        )
        
        if algorithm == 'ppo_baseline':
            # Only PPO, no additional components
            pass
        
        elif algorithm == 'mace_rl_full':
            # Full MACE-RL with all components
            if not is_atari:
                # Handle both tuple and integer state dimensions
                if isinstance(state_dim, tuple):
                    state_size = state_dim[0] * state_dim[1] * state_dim[2] if len(state_dim) == 3 else state_dim[0]
                else:
                    state_size = state_dim
                    
                components['episodic_memory'] = EpisodicMemory(memory_size, state_size)
                components['curiosity_module'] = CuriosityModule(
                    state_size, action_dim, components['episodic_memory'], 
                    continuous=has_continuous_action_space
                )
                
                # Initialize reward system with beta scheduler
                if beta_scheduler_config is not None:
                    components['reward_system'] = HybridRewardSystem(
                        components['curiosity_module'], 
                        beta_scheduler=beta_scheduler_config
                    )
                else:
                    components['reward_system'] = HybridRewardSystem(
                        components['curiosity_module'],
                        beta_initial=beta_initial
                    )
        
        elif algorithm == 'mace_rl_no_memory':
            # MACE-RL without episodic memory
            if not is_atari:
                state_size = self._calculate_state_size(state_dim)
                components['curiosity_module'] = CuriosityModule(
                    state_size, action_dim, None,
                    continuous=has_continuous_action_space
                )
                
                # Initialize reward system with beta scheduler
                if beta_scheduler_config is not None:
                    components['reward_system'] = HybridRewardSystem(
                        components['curiosity_module'], 
                        beta_scheduler=beta_scheduler_config
                    )
                else:
                    components['reward_system'] = HybridRewardSystem(
                        components['curiosity_module'],
                        beta_initial=beta_initial
                    )
        
        elif algorithm == 'mace_rl_no_curiosity':
            # MACE-RL without curiosity module
            if not is_atari:
                state_size = self._calculate_state_size(state_dim)
                components['episodic_memory'] = EpisodicMemory(memory_size, state_size)
                
                # Initialize reward system with no curiosity module
                if beta_scheduler_config is not None:
                    components['reward_system'] = HybridRewardSystem(
                        curiosity_module=None,
                        beta_scheduler=beta_scheduler_config
                    )
                else:
                    components['reward_system'] = HybridRewardSystem(
                        curiosity_module=None,
                        beta_initial=0.0
                    )
        
        elif algorithm == 'mace_rl_no_meta':
            # MACE-RL without meta-adaptation
            if not is_atari:
                state_size = self._calculate_state_size(state_dim)
                components['episodic_memory'] = EpisodicMemory(memory_size, state_size)
                components['curiosity_module'] = CuriosityModule(
                    state_size, action_dim, components['episodic_memory'],
                    continuous=has_continuous_action_space
                )
                
                # Initialize reward system with beta scheduler
                if beta_scheduler_config is not None:
                    components['reward_system'] = HybridRewardSystem(
                        components['curiosity_module'], 
                        beta_scheduler=beta_scheduler_config
                    )
                else:
                    components['reward_system'] = HybridRewardSystem(
                        components['curiosity_module'],
                        beta_initial=beta_initial
                    )
        
        elif algorithm == 'mace_rl_curiosity_only':
            # MACE-RL with only curiosity (no memory, no meta-adaptation)
            if not is_atari:
                state_size = self._calculate_state_size(state_dim)
                components['curiosity_module'] = CuriosityModule(
                    state_size, action_dim, None,
                    continuous=has_continuous_action_space
                )
                
                # Initialize reward system with beta scheduler
                if beta_scheduler_config is not None:
                    components['reward_system'] = HybridRewardSystem(
                        components['curiosity_module'], 
                        beta_scheduler=beta_scheduler_config
                    )
                else:
                    components['reward_system'] = HybridRewardSystem(
                        components['curiosity_module'],
                        beta_initial=beta_initial
                    )
        
        elif algorithm == 'mace_rl_memory_only':
            # MACE-RL with only episodic memory (no curiosity, no meta-adaptation)
            if not is_atari:
                state_size = self._calculate_state_size(state_dim)
                components['episodic_memory'] = EpisodicMemory(memory_size, state_size)
                
                # Initialize reward system with no curiosity module for consistency
                if beta_scheduler_config is not None:
                    components['reward_system'] = HybridRewardSystem(
                        curiosity_module=None,
                        beta_scheduler=beta_scheduler_config
                    )
                else:
                    components['reward_system'] = HybridRewardSystem(
                        curiosity_module=None,
                        beta_initial=0.0
                    )
        
        return components
    
    def run_all_experiments(self, parallel=True):
        """Run all experiment configurations."""
        experiment_tasks = []
        for env_name in EXPERIMENT_CONFIG['environments']:
            for algorithm in EXPERIMENT_CONFIG['algorithms']:
                for seed in EXPERIMENT_CONFIG['seeds']:
                    experiment_tasks.append((env_name, algorithm, seed))
        
        # Clean suite header
        print(f"\n{'='*80}")
        print(f"🧪 MACE-RL COMPREHENSIVE EXPERIMENT SUITE")
        print(f"   Total Experiments: {len(experiment_tasks)}")
        print(f"   Environments: {len(EXPERIMENT_CONFIG['environments'])}")
        print(f"   Algorithms: {len(EXPERIMENT_CONFIG['algorithms'])}")
        print(f"   Seeds: {len(EXPERIMENT_CONFIG['seeds'])}")
        print(f"   Execution Mode: {'Parallel' if parallel and len(experiment_tasks) > 1 else 'Sequential'}")
        print(f"{'='*80}\n")
        
        if parallel and len(experiment_tasks) > 1:
            # Run experiments in parallel
            print(f"🚀 Running experiments in parallel using {min(cpu_count(), 4)} processes\n")
            with Pool(processes=min(cpu_count(), 4)) as pool:
                results = pool.starmap(self.run_single_experiment, experiment_tasks)
        else:
            # Run experiments sequentially
            print("🔄 Running experiments sequentially\n")
            results = []
            for i, (env_name, algorithm, seed) in enumerate(experiment_tasks, 1):
                print(f"📊 Experiment {i}/{len(experiment_tasks)}")
                result = self.run_single_experiment(env_name, algorithm, seed)
                results.append(result)
        
        # Organize results
        for result in results:
            env_name = result['config']['env_name']
            algorithm = result['config']['algorithm']
            self.results[env_name][algorithm].append(result)
        
        self.logger.info("All experiments completed!")
        return self.results
    
    def run_ablation_studies(self, ablation_type='all', parallel=True):
        """Run simplified ablation studies."""
        self.logger.info(f"Starting ablation studies: {ablation_type}")
        
        ablation_tasks = []
        base_env = 'HopperBulletEnv-v0'  # Use HopperBulletEnv-v0 for ablations
        base_algorithm = 'mace_rl_full'
        

        
        if ablation_type in ['all', 'beta_values']:
            # Beta value ablation
            for beta in EXPERIMENT_CONFIG['ablation_studies']['beta_values']:
                for seed in EXPERIMENT_CONFIG['seeds']:
                    ablation_config = {'beta_initial': beta}
                    ablation_tasks.append((base_env, base_algorithm, seed, None, ablation_config))
        
        if ablation_type in ['all', 'beta_schedulers']:
            # Beta scheduler ablation (simplified)
            for scheduler_type in EXPERIMENT_CONFIG['ablation_studies']['beta_schedulers']:
                for seed in EXPERIMENT_CONFIG['seeds']:
                    ablation_config = {'beta_scheduler': scheduler_type}
                    ablation_tasks.append((base_env, base_algorithm, seed, None, ablation_config))
        
        self.logger.info(f"Total ablation experiments: {len(ablation_tasks)}")
        
        if parallel and len(ablation_tasks) > 1:
            # Run ablation studies in parallel
            with Pool(processes=min(cpu_count(), 4)) as pool:
                results = pool.starmap(self.run_single_experiment, ablation_tasks)
        else:
            # Run ablation studies sequentially
            results = []
            for task in ablation_tasks:
                result = self.run_single_experiment(*task)
                results.append(result)
        
        # Organize ablation results
        ablation_results = defaultdict(create_nested_dict)
        for result in results:
            config = result['config']
            if 'ablation' in config.get('exp_id', ''):
                ablation_key = f"{ablation_type}_ablation"
                ablation_results[ablation_key][str(config.get('ablation_config', {}))].append(result)
        
        # Save ablation results
        ablation_path = os.path.join(self.experiment_dir, f'ablation_results_{ablation_type}.json')
        with open(ablation_path, 'w') as f:
            json.dump(dict(ablation_results), f, indent=2, default=convert_numpy)
        
        self.logger.info(f"Ablation studies completed! Results saved to {ablation_path}")
        return ablation_results
    
    def run_publication_experiments(self, parallel=True, ablation=False):
        """📚 Run publication-quality experiments with extended settings."""
        self.logger.info("Starting publication-quality experiments")
        
        # Use publication settings
        pub_config = EXPERIMENT_CONFIG['publication_mode']
        
        # Override global config temporarily
        original_seeds = EXPERIMENT_CONFIG['seeds']
        original_episodes = EXPERIMENT_CONFIG['max_episodes']
        original_timesteps = EXPERIMENT_CONFIG['max_timesteps']
        
        EXPERIMENT_CONFIG['seeds'] = pub_config['seeds']
        EXPERIMENT_CONFIG['max_episodes'] = pub_config['max_episodes'] 
        EXPERIMENT_CONFIG['max_timesteps'] = pub_config['max_timesteps']
        
        # Calculate total experiments for comprehensive reporting
        main_experiments = len(EXPERIMENT_CONFIG['environments']) * len(EXPERIMENT_CONFIG['algorithms']) * len(EXPERIMENT_CONFIG['seeds'])
        
        # Calculate ablation experiments

        key_ablations = ['beta_values']
        ablation_experiments = 0
        if ablation:
            for ablation_type in key_ablations:
                if ablation_type == 'beta_values':
                    ablation_experiments += len(EXPERIMENT_CONFIG['ablation_studies']['beta_values']) * len(EXPERIMENT_CONFIG['seeds'])

        total_experiments = main_experiments + ablation_experiments
        
        # Enhanced publication mode header
        print(f"\n{'='*80}")
        print(f"📚 PUBLICATION-QUALITY EXPERIMENT SUITE")
        print(f"   Main Experiments: {main_experiments}")
        print(f"   Ablation Studies: {ablation_experiments}")
        print(f"   Total Experiments: {total_experiments}")
        print(f"   Environments: {len(EXPERIMENT_CONFIG['environments'])}")
        print(f"   Algorithms: {len(EXPERIMENT_CONFIG['algorithms'])}")
        print(f"   Seeds: {len(EXPERIMENT_CONFIG['seeds'])}")
        print(f"   Episodes per Experiment: {EXPERIMENT_CONFIG['max_episodes']}")
        print(f"   Max Timesteps: {EXPERIMENT_CONFIG['max_timesteps']}")
        print(f"   Execution Mode: {'Parallel' if parallel else 'Sequential'}")
        print(f"{'='*80}\n")
        
        try:
            # Run main experiments with publication settings
            print("🚀 Phase 1: Running Main Experiments")
            results = self.run_all_experiments(parallel=parallel)
            
            # Run key ablations with publication settings
            if ablation:
                print("\n🔬 Phase 2: Running Ablation Studies")
                for ablation in key_ablations:
                    print(f"   Running {ablation} ablation...")
                    self.run_ablation_studies(ablation_type=ablation, parallel=parallel)
            
            self.logger.info("Publication experiments completed!")
            
        finally:
            # Restore original config
            EXPERIMENT_CONFIG['seeds'] = original_seeds
            EXPERIMENT_CONFIG['max_episodes'] = original_episodes
            EXPERIMENT_CONFIG['max_timesteps'] = original_timesteps
        
        return results
    
    def analyze_results(self):
        """🔍 Comprehensive analysis of experimental results - Rich metadata only."""
        self.logger.info("Starting comprehensive result analysis (metadata only)")
        
        # Load existing results if not in memory
        if not self.results:
            self._load_existing_results()
        
        if not self.results:
            self.logger.warning("No results found to analyze!")
            return
        
        # Generate statistical analysis
        stats_results = self._statistical_analysis()
        
        # Export results to different formats
        self.export_results_to_csv()
        self.create_latex_table()
        
        # Generate comprehensive summary report
        self._generate_summary_report(stats_results)
        
        # Generate detailed analysis files with rich metadata
        self._generate_detailed_analysis()
        
        # Generate experiment summary with key metrics
        self._generate_experiment_summary()
        
        self.logger.info("🎉 Comprehensive analysis completed! (No plots generated)")
    
    def _generate_experiment_summary(self):
        """Generate a comprehensive experiment summary with rich metadata."""
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'experiment_info': {
                'total_experiments': 0,
                'environments_tested': list(self.results.keys()),
                'algorithms_tested': set(),
                'total_runtime_episodes': 0
            },
            'performance_summary': {},
            'algorithm_rankings': {},
            'environment_difficulty': {},
            'convergence_metrics': {},
            'exploration_metrics': {}
        }
        
        total_experiments = 0
        all_algorithms = set()
        total_episodes = 0
        
        # Collect comprehensive metadata
        for env_name in self.results:
            summary_data['performance_summary'][env_name] = {}
            env_algorithms = []
            
            for algorithm in self.results[env_name]:
                all_algorithms.add(algorithm)
                env_algorithms.append(algorithm)
                results = self.results[env_name][algorithm]
                
                if not results:
                    continue
                
                total_experiments += len(results)
                
                # Extract performance metrics
                rewards = [r['mean_reward'] for r in results]
                episode_counts = [len(r['episode_rewards']) for r in results]
                total_episodes += sum(episode_counts)
                
                # Rich performance metadata
                perf_data = {
                    'runs': len(results),
                    'mean_performance': float(np.mean(rewards)),
                    'std_performance': float(np.std(rewards)),
                    'min_performance': float(np.min(rewards)),
                    'max_performance': float(np.max(rewards)),
                    'median_performance': float(np.median(rewards)),
                    'q25_performance': float(np.percentile(rewards, 25)),
                    'q75_performance': float(np.percentile(rewards, 75)),
                    'coefficient_of_variation': float(np.std(rewards) / np.mean(rewards)) if np.mean(rewards) != 0 else float('inf'),
                    'total_episodes': sum(episode_counts),
                    'avg_episodes_per_run': float(np.mean(episode_counts)),
                    'successful_runs': len([r for r in rewards if r > 0]),
                    'failure_rate': float(len([r for r in rewards if r <= 0]) / len(rewards))
                }
                
                # Add learning curve analysis
                if results[0].get('episode_rewards'):
                    all_learning_curves = [r['episode_rewards'] for r in results]
                    
                    # Convergence analysis
                    convergence_episodes = []
                    final_performances = []
                    
                    for curve in all_learning_curves:
                        if len(curve) > 100:
                            # Calculate when algorithm converged (last 10% of episodes)
                            final_10_percent = curve[-len(curve)//10:]
                            final_performance = np.mean(final_10_percent)
                            final_performances.append(final_performance)
                            
                            # Find convergence point (where performance stabilizes)
                            window_size = max(50, len(curve) // 20)
                            rolling_mean = np.convolve(curve, np.ones(window_size)/window_size, mode='valid')
                            
                            # Find where variance becomes small (converged)
                            rolling_std = []
                            for i in range(len(rolling_mean) - window_size):
                                rolling_std.append(np.std(rolling_mean[i:i+window_size]))
                            
                            if rolling_std:
                                convergence_idx = np.argmin(rolling_std) + window_size
                                convergence_episodes.append(convergence_idx)
                    
                    if convergence_episodes:
                        perf_data['convergence_analysis'] = {
                            'mean_convergence_episode': float(np.mean(convergence_episodes)),
                            'std_convergence_episode': float(np.std(convergence_episodes)),
                            'fastest_convergence': int(np.min(convergence_episodes)),
                            'slowest_convergence': int(np.max(convergence_episodes))
                        }
                    
                    if final_performances:
                        perf_data['final_performance_analysis'] = {
                            'mean_final_performance': float(np.mean(final_performances)),
                            'std_final_performance': float(np.std(final_performances)),
                            'best_final_performance': float(np.max(final_performances)),
                            'worst_final_performance': float(np.min(final_performances))
                        }
                
                # Add curiosity metrics if available
                curiosity_rewards = []
                for result in results:
                    if 'curiosity_rewards' in result and result['curiosity_rewards']:
                        curiosity_rewards.extend(result['curiosity_rewards'])
                
                if curiosity_rewards:
                    perf_data['curiosity_analysis'] = {
                        'total_curiosity_samples': len(curiosity_rewards),
                        'mean_curiosity_bonus': float(np.mean(curiosity_rewards)),
                        'std_curiosity_bonus': float(np.std(curiosity_rewards)),
                        'max_curiosity_bonus': float(np.max(curiosity_rewards)),
                        'curiosity_contribution_ratio': float(np.mean(curiosity_rewards) / (np.mean(rewards) + 1e-8))
                    }
                
                summary_data['performance_summary'][env_name][algorithm] = perf_data
            
            # Environment-level analysis
            if env_algorithms:
                env_performances = []
                for alg in env_algorithms:
                    if alg in summary_data['performance_summary'][env_name]:
                        env_performances.append(summary_data['performance_summary'][env_name][alg]['mean_performance'])
                
                if env_performances:
                    summary_data['environment_difficulty'][env_name] = {
                        'mean_performance_across_algorithms': float(np.mean(env_performances)),
                        'std_performance_across_algorithms': float(np.std(env_performances)),
                        'best_algorithm_performance': float(np.max(env_performances)),
                        'worst_algorithm_performance': float(np.min(env_performances)),
                        'performance_range': float(np.max(env_performances) - np.min(env_performances)),
                        'difficulty_score': float(1.0 / (np.mean(env_performances) + 1e-8))  # Higher score = more difficult
                    }
        
        # Update experiment info
        summary_data['experiment_info']['total_experiments'] = total_experiments
        summary_data['experiment_info']['algorithms_tested'] = list(all_algorithms)
        summary_data['experiment_info']['total_runtime_episodes'] = total_episodes
        
        # Generate algorithm rankings across all environments
        algorithm_scores = defaultdict(list)
        for env_name in summary_data['performance_summary']:
            for algorithm in summary_data['performance_summary'][env_name]:
                score = summary_data['performance_summary'][env_name][algorithm]['mean_performance']
                algorithm_scores[algorithm].append(score)
        
        for algorithm in algorithm_scores:
            summary_data['algorithm_rankings'][algorithm] = {
                'overall_mean_score': float(np.mean(algorithm_scores[algorithm])),
                'overall_std_score': float(np.std(algorithm_scores[algorithm])),
                'environments_tested': len(algorithm_scores[algorithm]),
                'best_environment_score': float(np.max(algorithm_scores[algorithm])),
                'worst_environment_score': float(np.min(algorithm_scores[algorithm])),
                'consistency_score': float(1.0 / (np.std(algorithm_scores[algorithm]) + 1e-8))  # Higher = more consistent
            }
        
        # Save comprehensive summary
        summary_path = os.path.join(self.experiment_dir, 'comprehensive_experiment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=convert_numpy)
        
        self.logger.info(f"Comprehensive experiment summary saved to: {summary_path}")
        
        # Also save a human-readable summary
        readable_path = os.path.join(self.experiment_dir, 'experiment_summary_readable.txt')
        with open(readable_path, 'w') as f:
            f.write("MACE-RL EXPERIMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {summary_data['timestamp']}\n")
            f.write(f"Total Experiments: {total_experiments}\n")
            f.write(f"Total Episodes: {total_episodes}\n")
            f.write(f"Environments: {len(summary_data['experiment_info']['environments_tested'])}\n")
            f.write(f"Algorithms: {len(all_algorithms)}\n\n")
            
            f.write("ALGORITHM RANKINGS (Overall Mean Score):\n")
            f.write("-" * 40 + "\n")
            sorted_algorithms = sorted(summary_data['algorithm_rankings'].items(), 
                                     key=lambda x: x[1]['overall_mean_score'], reverse=True)
            for i, (alg, data) in enumerate(sorted_algorithms, 1):
                f.write(f"{i}. {alg}: {data['overall_mean_score']:.3f} ± {data['overall_std_score']:.3f}\n")
            
            f.write("\nENVIRONMENT DIFFICULTY (Higher = More Difficult):\n")
            f.write("-" * 50 + "\n")
            sorted_envs = sorted(summary_data['environment_difficulty'].items(), 
                               key=lambda x: x[1]['difficulty_score'], reverse=True)
            for i, (env, data) in enumerate(sorted_envs, 1):
                f.write(f"{i}. {env}: Difficulty {data['difficulty_score']:.3f} (Mean Performance: {data['mean_performance_across_algorithms']:.3f})\n")
        
        self.logger.info(f"Human-readable summary saved to: {readable_path}")
        return summary_data
    
    def _generate_detailed_analysis(self):
        """Generate detailed analysis with advanced metrics."""
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'summary_statistics': {},
            'convergence_analysis': {},
            'robustness_metrics': {},
            'computational_metrics': {}
        }
        
        # Calculate detailed metrics for each environment-algorithm pair
        for env_name in self.results:
            analysis_data['summary_statistics'][env_name] = {}
            
            for algorithm in self.results[env_name]:
                results = self.results[env_name][algorithm]
                if not results:
                    continue
                
                # Extract all rewards
                all_rewards = [r['mean_reward'] for r in results]
                all_episode_rewards = [r['episode_rewards'] for r in results]
                
                # Calculate advanced metrics
                metrics = {
                    'mean_performance': np.mean(all_rewards),
                    'std_performance': np.std(all_rewards),
                    'min_performance': np.min(all_rewards),
                    'max_performance': np.max(all_rewards),
                    'median_performance': np.median(all_rewards),
                    'q25_performance': np.percentile(all_rewards, 25),
                    'q75_performance': np.percentile(all_rewards, 75),
                    'coefficient_of_variation': np.std(all_rewards) / np.mean(all_rewards) if np.mean(all_rewards) != 0 else float('inf'),
                    'sample_efficiency_scores': [self.calculate_sample_efficiency(rewards) for rewards in all_episode_rewards],
                    'n_runs': len(results)
                }
                
                # Add sample efficiency summary
                sample_effs = [s for s in metrics['sample_efficiency_scores'] if s is not None]
                if sample_effs:
                    metrics['mean_sample_efficiency'] = np.mean(sample_effs)
                    metrics['std_sample_efficiency'] = np.std(sample_effs)
                
                analysis_data['summary_statistics'][env_name][algorithm] = metrics
        
        # Save detailed analysis
        analysis_path = os.path.join(self.experiment_dir, 'detailed_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=convert_numpy)
        
        self.logger.info(f"Detailed analysis saved to: {analysis_path}")
    
    def _load_existing_results(self):
        """Load results from data files."""
        data_dir = os.path.join(self.experiment_dir, 'data')
        if not os.path.exists(data_dir):
            self.logger.warning("No data directory found. Run experiments first.")
            return
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'rb') as f:
                    result = pickle.load(f)
                
                env_name = result['config']['env_name']
                algorithm = result['config']['algorithm']
                self.results[env_name][algorithm].append(result)
    
    def _statistical_analysis(self):
        """Perform statistical analysis on results."""
        self.logger.info("Performing statistical analysis")
        
        stats_results = {}
        
        for env_name in self.results:
            stats_results[env_name] = {}
            
            # Collect data for each algorithm
            algorithm_data = {}
            for algorithm in self.results[env_name]:
                rewards = [result['mean_reward'] for result in self.results[env_name][algorithm]]
                algorithm_data[algorithm] = rewards
            
            # Compute statistics
            for algorithm in algorithm_data:
                rewards = algorithm_data[algorithm]
                if rewards:
                    stats_results[env_name][algorithm] = {
                        'mean': np.mean(rewards),
                        'std': np.std(rewards),
                        'median': np.median(rewards),
                        'min': np.min(rewards),
                        'max': np.max(rewards),
                        'n_runs': len(rewards)
                    }
            
            # Perform pairwise t-tests
            algorithms = list(algorithm_data.keys())
            p_values = {}
            for i in range(len(algorithms)):
                for j in range(i+1, len(algorithms)):
                    alg1, alg2 = algorithms[i], algorithms[j]
                    if algorithm_data[alg1] and algorithm_data[alg2]:
                        _, p_val = stats.ttest_ind(algorithm_data[alg1], algorithm_data[alg2])
                        p_values[f"{alg1}_vs_{alg2}"] = p_val
            
            stats_results[env_name]['p_values'] = p_values
        
        # Save statistical results
        stats_path = os.path.join(self.experiment_dir, 'statistical_analysis.json')
        with open(stats_path, 'w') as f:
            json.dump(stats_results, f, indent=2, default=convert_numpy)
        
        return stats_results
    
    def _generate_plots(self):
        """Plotting disabled - using rich metadata storage instead."""
        self.logger.info("Plotting disabled - rich metadata stored in JSON files")
        pass
    
    def _plot_training_vs_evaluation_rewards(self):
        """Plotting function removed - using metadata collection only."""
        self.logger.info("Plotting function _plot_training_vs_evaluation_rewards removed - using metadata collection only")
        pass

    def _plot_significance_heatmap(self):
        """Plotting function removed - using metadata collection only."""
        self.logger.info("Plotting function _plot_significance_heatmap removed - using metadata collection only")
        pass

    def _plot_ablation_studies(self):
        """Plotting function removed - using metadata collection only."""
        self.logger.info("Plotting function _plot_ablation_studies removed - using metadata collection only")
        pass

    def _plot_convergence_analysis(self):
        """Plotting function removed - using metadata collection only."""
        self.logger.info("Plotting function _plot_convergence_analysis removed - using metadata collection only")
        pass

    def _plot_algorithm_comparison_radar(self):
        """Plotting function removed - using metadata collection only."""
        self.logger.info("Plotting function _plot_algorithm_comparison_radar removed - using metadata collection only")
        pass

    def _plot_environment_difficulty_analysis(self):
        """Plotting function removed - using metadata collection only."""
        self.logger.info("Plotting function _plot_environment_difficulty_analysis removed - using metadata collection only")
        pass

    def _generate_summary_report(self, stats_results):
        """Generate comprehensive summary report."""
        self.logger.info("Generating summary report")
        
        report_path = os.path.join(self.experiment_dir, 'experiment_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# MACE-RL Experimental Results Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Experiment Configuration\n\n")
            f.write(f"- **Environments**: {', '.join(EXPERIMENT_CONFIG['environments'])}\n")
            f.write(f"- **Algorithms**: {', '.join(EXPERIMENT_CONFIG['algorithms'])}\n")
            f.write(f"- **Random Seeds**: {EXPERIMENT_CONFIG['seeds']}\n")
            f.write(f"- **Max Episodes**: {EXPERIMENT_CONFIG['max_episodes']}\n")
            f.write(f"- **Max Timesteps**: {EXPERIMENT_CONFIG['max_timesteps']}\n\n")
            
            f.write("## 🎯 Important: Reward System Explanation\n\n")
            f.write("**This experiment uses a FAIR EVALUATION methodology:**\n\n")
            f.write("- **Training Rewards**: Algorithms with curiosity modules use hybrid rewards (extrinsic + curiosity bonus) for learning\n")
            f.write("- **Evaluation Rewards**: ALL plots and comparisons show ONLY extrinsic rewards (pure environment performance)\n")
            f.write("- **Research Validity**: This ensures fair comparison between algorithms on actual task performance\n")
            f.write("- **Curiosity Bonus**: Used only to guide exploration during training, not for final evaluation\n\n")
            f.write("*Note: Plots labeled 'Extrinsic Rewards' show the true environment task performance for research comparison.*\n\n")
            
            f.write("## Results Summary\n\n")
            
            for env_name in stats_results:
                f.write(f"### {env_name}\n\n")
                f.write("| Algorithm | Mean Reward | Std Dev | Median | Min | Max | Runs |\n")
                f.write("|-----------|-------------|---------|--------|-----|-----|------|\n")
                
                for algorithm in stats_results[env_name]:
                    if algorithm != 'p_values':
                        stats = stats_results[env_name][algorithm]
                        f.write(f"| {algorithm} | {stats['mean']:.2f} | {stats['std']:.2f} | "
                               f"{stats['median']:.2f} | {stats['min']:.2f} | "
                               f"{stats['max']:.2f} | {stats['n_runs']} |\n")
                
                f.write("\n")
                
                # Statistical significance
                if 'p_values' in stats_results[env_name]:
                    f.write("#### Statistical Significance (p-values)\n\n")
                    for comparison, p_val in stats_results[env_name]['p_values'].items():
                        significance = "**Significant**" if p_val < 0.05 else "Not significant"
                        f.write(f"- {comparison}: p = {p_val:.4f} ({significance})\n")
                    f.write("\n")
            
            f.write("## Key Findings\n\n")
            
            # Automatically generate insights
            best_algorithm_per_env = {}
            for env_name in stats_results:
                best_mean = float('-inf')
                best_alg = None
                for algorithm in stats_results[env_name]:
                    if algorithm != 'p_values':
                        mean_reward = stats_results[env_name][algorithm]['mean']
                        if mean_reward > best_mean:
                            best_mean = mean_reward
                            best_alg = algorithm
                best_algorithm_per_env[env_name] = (best_alg, best_mean)
            
            for env_name, (best_alg, best_mean) in best_algorithm_per_env.items():
                f.write(f"- **{env_name}**: Best performing algorithm is {best_alg} "
                       f"with mean reward {best_mean:.2f}\n")
            
            f.write("\n## Files Generated\n\n")
            f.write("- **Models**: `models/` directory contains trained model checkpoints\n")
            f.write("- **Data**: `data/` directory contains raw experimental results\n")
            f.write("- **Plots**: `plots/` directory contains visualization plots\n")
            f.write("- **Logs**: `logs/` directory contains detailed training logs\n")
            f.write("- **Analysis**: `statistical_analysis.json` contains detailed statistics\n")
            
            # Add advanced insights
            f.write("\n## 🎯 Advanced Analysis\n\n")
            
            # Sample efficiency analysis
            f.write("### Sample Efficiency\n\n")
            f.write("| Algorithm | Episodes to Convergence | Sample Efficiency Score |\n")
            f.write("|-----------|------------------------|-------------------------|\n")
            
            for env_name in stats_results:
                for algorithm in stats_results[env_name]:
                    if algorithm != 'p_values':
                        # Placeholder for sample efficiency calculation
                        f.write(f"| {algorithm} | TBD | TBD |\n")
            
            f.write("\n### Computational Cost Analysis\n\n")
            f.write("- **Training Time**: Analysis of computational requirements\n")
            f.write("- **Memory Usage**: Peak memory consumption per algorithm\n")
            f.write("- **Scalability**: Performance scaling with problem size\n")
            
            f.write("\n### Robustness Analysis\n\n") 
            f.write("- **Seed Variance**: Consistency across different random seeds\n")
            f.write("- **Hyperparameter Sensitivity**: Stability to parameter changes\n")
            f.write("- **Environment Transfer**: Generalization across domains\n")
    
    # Additional utility methods
    def calculate_sample_efficiency(self, rewards, convergence_threshold=0.9):
        """Calculate sample efficiency metrics."""
        if not rewards:
            return None
        
        final_performance = np.mean(rewards[-100:]) if len(rewards) >= 100 else rewards[-1]
        target = convergence_threshold * final_performance
        
        # Find first episode where performance reaches target
        for i, reward in enumerate(rewards):
            if reward >= target:
                return i
        
        return len(rewards)  # Never converged
    
    def compute_statistical_power(self, group1, group2, effect_size=0.5, alpha=0.05):
        """Compute statistical power for comparison."""
        try:
            from scipy.stats import ttest_ind
            import numpy as np
            
            # Calculate observed effect size
            if len(group1) > 1 and len(group2) > 1:
                pooled_std = np.sqrt(((len(group1)-1)*np.var(group1) + (len(group2)-1)*np.var(group2)) / (len(group1)+len(group2)-2))
                observed_effect = abs(np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
                
                # Simple power approximation
                power = min(1.0, observed_effect / effect_size)
                return power
            
        except ImportError:
            pass
        
        return None
    
    def generate_experiment_summary_table(self):
        """Generate a comprehensive summary table of all experiments."""
        summary_data = []
        
        for env_name in self.results:
            for algorithm in self.results[env_name]:
                for result in self.results[env_name][algorithm]:
                    summary_data.append({
                        'Environment': env_name,
                        'Algorithm': algorithm,
                        'Seed': result['config']['seed'],
                        'Mean Reward': result['mean_reward'],
                        'Std Reward': result['std_reward'],
                        'Final Reward': result['final_reward'],
                        'Episodes': len(result['episode_rewards']),
                        'Sample Efficiency': self.calculate_sample_efficiency(result['episode_rewards'])
                    })
        
        return summary_data
    
    def export_results_to_csv(self):
        """Export all results to JSON format for external analysis."""
        try:
            summary_data = self.generate_experiment_summary_table()
            
            json_path = os.path.join(self.experiment_dir, 'experiment_results.json')
            with open(json_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=convert_numpy)
            
            self.logger.info(f"Results exported to JSON: {json_path}")
            return json_path
            
        except Exception as e:
            self.logger.warning(f"Failed to export results: {e}")
            return None
    
    def create_latex_table(self):
        """Generate LaTeX table for publication."""
        latex_content = []
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{MACE-RL Experimental Results}")
        latex_content.append("\\label{tab:results}")
        latex_content.append("\\begin{tabular}{lcccc}")
        latex_content.append("\\toprule")
        latex_content.append("Environment & Algorithm & Mean Reward & Std Dev & p-value \\\\")
        latex_content.append("\\midrule")
        
        # Add data rows (placeholder)
        for env_name in self.results:
            for algorithm in self.results[env_name]:
                if self.results[env_name][algorithm]:
                    mean_reward = np.mean([r['mean_reward'] for r in self.results[env_name][algorithm]])
                    std_reward = np.std([r['mean_reward'] for r in self.results[env_name][algorithm]])
                    latex_content.append(f"{env_name} & {algorithm} & {mean_reward:.2f} & {std_reward:.2f} & - \\\\")
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        latex_path = os.path.join(self.experiment_dir, 'results_table.tex')
        with open(latex_path, 'w') as f:
            f.write('\n'.join(latex_content))
        
        self.logger.info(f"LaTeX table generated: {latex_path}")
        return latex_path
    
    def _plot_sample_efficiency_analysis(self):
        """Plotting function removed - using metadata collection only."""
        self.logger.info("Plotting function _plot_sample_efficiency_analysis removed - using metadata collection only")
        pass

    def _plot_exploration_exploitation_balance(self):
        """Plotting function removed - using metadata collection only."""
        self.logger.info("Plotting function _plot_exploration_exploitation_balance removed - using metadata collection only")
        pass

    def _plot_beta_scheduler_evolution(self):
        """Plotting function removed - using metadata collection only."""
        self.logger.info("Plotting function _plot_beta_scheduler_evolution removed - using metadata collection only")
        pass

    def _plot_learning_stability_analysis(self):
        """Plotting function removed - using metadata collection only."""
        self.logger.info("Plotting function _plot_learning_stability_analysis removed - using metadata collection only")
        pass

    def _plot_algorithmic_efficiency_comparison(self):
        """Plotting function removed - using metadata collection only."""
        self.logger.info("Plotting function _plot_algorithmic_efficiency_comparison removed - using metadata collection only")
        pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MACE-RL Experiment Runner')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run quick test with reduced episodes')
    parser.add_argument('--run-experiments', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--run-ablation', action='store_true',
                       help='Run ablation studies')
    parser.add_argument('--publication-mode', action='store_true',
                       help='Run publication-quality experiments')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel execution')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to store results')
    
    args = parser.parse_args()
    
    # Initialize experiment runner
    runner = ExperimentRunner(results_dir=args.results_dir)
    
    print("🚀 MACE-RL Experiment Suite")
    print("=" * 50)
    print(f"📁 Results will be saved to: {runner.experiment_dir}")
    print(f"📋 Configuration:")
    print(f"   • Environments: {EXPERIMENT_CONFIG['environments']}")
    print(f"   • Algorithms: {EXPERIMENT_CONFIG['algorithms']}")
    print(f"   • Seeds: {EXPERIMENT_CONFIG['seeds']}")
    print(f"   • Max Episodes: {EXPERIMENT_CONFIG['max_episodes']}")
    print("=" * 50)
    
    # Determine what to run
    if args.quick_test:
        print("🔬 Running Quick Test Mode")
        print("Testing all environments with all 7 algorithm types")
        
        # Test with all environments and ALL algorithms
        test_environments = EXPERIMENT_CONFIG['environments']
        test_algorithms = EXPERIMENT_CONFIG['algorithms']  # All 7 algorithms
        test_seed = EXPERIMENT_CONFIG['seeds'][0]
        
        total_experiments = len(test_environments) * len(test_algorithms)
        print(f"Running {total_experiments} quick test experiments:")
        print(f"   • Environments: {test_environments}")
        print(f"   • Algorithms: {test_algorithms}")
        print(f"   • Seed: {test_seed}")
        print(f"   • Episodes per experiment: 100")
        print("=" * 50)
        
        results = []
        for i, env in enumerate(test_environments):
            for j, algorithm in enumerate(test_algorithms):
                exp_num = i * len(test_algorithms) + j + 1
                print(f"\n[{exp_num}/{total_experiments}] {env} with {algorithm}")
                
                result = runner.run_single_experiment(env, algorithm, test_seed, episode_limit=100)
                
                if result and 'evaluation_metrics' in result:
                    eval_metrics = result['evaluation_metrics']
                    results.append({
                        'env': env,
                        'algorithm': algorithm,
                        'training_reward': f"{result['mean_reward']:.2f} ± {result['std_reward']:.2f}",
                        'eval_reward': f"{eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}",
                        'directory': result.get('experiment_directory', 'N/A')
                    })
                else:
                    print(f"❌ Experiment failed: {env} with {algorithm}")
        
        # Summary of quick test results
        print("\n" + "=" * 70)
        print("📊 QUICK TEST SUMMARY")
        print("=" * 70)
        for result in results:
            print(f"🔹 {result['env']} + {result['algorithm']}")
            print(f"   Training:   {result['training_reward']}")
            print(f"   Evaluation: {result['eval_reward']}")
        print("=" * 70)
        print(f"✅ Quick test completed: {len(results)}/{total_experiments} experiments successful")
    
    elif args.publication_mode:
        print("📊 Running Publication Mode")
        use_parallel = not args.no_parallel
        runner.run_publication_experiments(parallel=use_parallel,ablation=args.run_ablation)
            
    elif args.run_experiments:
        print("🏃 Running All Experiments")
        use_parallel = not args.no_parallel
        runner.run_all_experiments(parallel=use_parallel)
        
    elif args.run_ablation:
        print("� Running Ablation Studies")
        use_parallel = not args.no_parallel
        runner.run_ablation_studies(parallel=use_parallel)
        
    else:
        print("ℹ️  No experiment type specified. Use --help for options.")
        print("Available options:")
        print("  --quick-test      : Run a single quick experiment")
        print("  --run-experiments : Run all configured experiments")
        print("  --run-ablation    : Run ablation studies")
        print("  --publication-mode: Run publication-quality experiments")
        
    print("\n🎯 Experiment suite completed!")
    print(f"📁 Results saved in: {runner.experiment_dir}")

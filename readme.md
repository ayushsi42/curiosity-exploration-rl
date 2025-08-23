# MACE-RL: Meta-Adaptive Curiosity-Driven Exploration with Episodic Memory

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Environment Setup](#environment-setup)
5. [Training Scripts](#training-scripts)
6. [Utility Functions](#utility-functions)
7. [How Components Interconnect](#how-components-interconnect)
8. [Usage Examples](#usage-examples)
9. [Experimental Setup](#experimental-setup)
10. [Troubleshooting](#troubleshooting)

## Overview

MACE-RL is a reinforcement learning framework that combines meta-learning, curiosity-driven exploration, and episodic memory to achieve sample-efficient learning in sparse-reward environments. The system addresses the exploration problem in RL by adaptively adjusting curiosity parameters based on past experiences stored in an episodic memory buffer.

### Key Innovations:
- **Meta-Adaptive Curiosity**: Dynamically adjusts exploration strategies using LSTM-based meta-learning
- **Episodic Memory Integration**: Stores and retrieves past experiences using attention mechanisms
- **Hybrid Reward System**: Combines extrinsic rewards with intrinsic curiosity bonuses
- **Multi-Environment Support**: Works with Atari, MiniGrid, and PyBullet environments

## Project Structure

```
mace_rl/
├── mace_rl/                    # Core library package
│   ├── __init__.py            # Package initialization
│   ├── agents/                # RL agents implementation
│   │   ├── __init__.py
│   │   ├── ppo.py            # Proximal Policy Optimization agent
│   │   └── a2c.py            # Advantage Actor-Critic agent
│   ├── envs/                  # Environment wrappers
│   │   ├── __init__.py
│   │   ├── atari_env.py      # Atari game wrappers
│   │   ├── minigrid_env.py   # MiniGrid environment wrappers
│   │   └── pybullet_env.py   # PyBullet physics simulation wrappers
│   ├── modules/               # Core MACE-RL components
│   │   ├── __init__.py
│   │   ├── curiosity.py      # Curiosity-driven exploration module
│   │   ├── episodic_memory.py # Episodic memory system
│   │   └── meta_adaptation.py # Meta-learning adaptation network
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── reward_system.py  # Hybrid reward calculation
│       └── utils.py          # General utilities (logging, profiling, etc.)
├── train.py                   # Main training script for MACE-RL
├── ablation_train.py         # Ablation study training script
├── baseline_ppo.py           # PPO baseline training
├── baseline_a2c.py           # A2C baseline training
├── plot_results.py           # Results visualization script
├── requirements.txt          # Python dependencies
└── plan.md                   # Research plan and methodology
```

## Core Components

### 1. Agents (`mace_rl/agents/`)

#### PPO Agent (`ppo.py`)
The Proximal Policy Optimization agent is the base RL algorithm used in MACE-RL.

**Key Classes:**
- `RolloutBuffer`: Stores trajectory data during episodes
  - `actions`: List of actions taken
  - `states`: List of observed states
  - `logprobs`: Log probabilities of actions
  - `rewards`: List of rewards received
  - `is_terminals`: Episode termination flags

- `ActorCritic`: Neural network architecture
  - **For image observations**: CNN feature extractor (Conv2d layers with ReLU)
  - **For vector observations**: Fully connected layers
  - **Actor network**: Outputs action probabilities (discrete) or action means (continuous)
  - **Critic network**: Outputs state value estimates
  - **Action handling**: Categorical distribution for discrete, MultivariateNormal for continuous

- `PPO`: Main PPO algorithm implementation
  - **Initialization**: Creates policy networks, optimizers, and buffers
  - **Action selection**: Uses old policy for action sampling during rollouts
  - **Update mechanism**: 
    - Computes Monte Carlo returns with advantage estimation
    - Performs K epochs of policy updates using clipped objective
    - Updates both actor and critic networks
    - Implements entropy regularization for exploration

**Memory Requirements**: ~50-100MB for network parameters, ~10MB for rollout buffer

#### A2C Agent (`a2c.py`)
Advantage Actor-Critic baseline implementation that reuses PPO's ActorCritic architecture.

**Key Differences from PPO**:
- Single-step updates instead of mini-batch updates
- No clipped objective function
- Direct policy gradient with baseline subtraction
- Lower computational overhead but potentially less stable training

### 2. Environments (`mace_rl/envs/`)

#### Atari Environments (`atari_env.py`)
Comprehensive Atari game preprocessing pipeline following DeepMind's methodology.

**Environment Wrappers** (applied in sequence):
1. `NoopResetEnv`: Performs 1-30 random no-op actions at episode start
2. `MaxAndSkipEnv`: Skips 4 frames, taking max over last 2 frames
3. `EpisodicLifeEnv`: Treats life loss as episode termination
4. `FireResetEnv`: Automatically fires at episode start for applicable games
5. `WarpFrame`: Resizes frames to 84x84 grayscale
6. `ScaledFloatFrame`: Normalizes pixel values to [0,1] range
7. `ClipRewardEnv`: Clips rewards to {-1, 0, 1}
8. `FrameStack`: Stacks 4 consecutive frames for temporal information

**Usage Pattern**:
```python
env = make_atari_env('PongNoFrameskip-v4')
env = wrap_deepmind(env, frame_stack=True, clip_rewards=True)
```

**Output**: Observations are (84, 84, 4) uint8 arrays representing stacked grayscale frames

#### MiniGrid Environments (`minigrid_env.py`)
Wrapper for procedurally generated grid-world environments.

**MiniGridWrapper**:
- Extracts image component from MiniGrid's dictionary observation
- Flattens 3D image to 1D vector for compatibility with standard RL algorithms
- Observation space becomes 1D array of pixel values

**Supported Environments**:
- Navigation tasks (Empty, FourRooms, etc.)
- Object manipulation (DoorKey, Unlock, etc.)
- Memory-dependent tasks (Memory, RedBlueDoors, etc.)

#### PyBullet Environments (`pybullet_env.py`)
Simple wrapper for physics-based continuous control tasks.

**Characteristics**:
- Continuous action spaces (typically 6-12 dimensional)
- 3D physics simulation with realistic dynamics
- Common tasks: locomotion, manipulation, navigation
- Observation spaces vary by task (joint positions, velocities, contact forces)

### 3. Core Modules (`mace_rl/modules/`)

#### Curiosity Module (`curiosity.py`)
Implements curiosity-driven intrinsic motivation using the formula:
```
C(s,a) = η(s,a) × M(s,a) × T(s,a)
```

**Components**:
- **Novelty Score η(s,a)**:
  - *Discrete spaces*: Count-based novelty using visitation frequency
    - `novelty = 1.0 / (count + 1)`
  - *Continuous spaces*: k-NN density estimation
    - Maintains buffer of 10,000 recent states
    - Uses k=10 nearest neighbors for density estimation

- **Memory Relevance M(s,a)**:
  - Queries episodic memory for similar past experiences
  - Uses cosine similarity for relevance scoring
  - Returns mean relevance score across retrieved memories

- **Temporal Score T(s,a)**:
  - Currently simplified to constant 1.0
  - Placeholder for temporal importance weighting

**Memory Footprint**: ~40MB for state buffer in continuous environments

#### Episodic Memory (`episodic_memory.py`)
Differentiable Neural Dictionary implementing episodic memory storage and retrieval.

**Architecture**:
- **Capacity**: Configurable maximum number of memories (default: 1000)
- **Key-Value Storage**:
  - Keys: State embeddings (64-dimensional vectors)
  - Values: Arbitrary tensors (rewards, next states, etc.)
- **Retrieval Mechanism**:
  - Attention-based similarity computation using cosine similarity
  - Softmax over similarities for weighted retrieval
  - Top-k retrieval for multiple relevant memories

**Memory Management**:
- **LRU Policy**: Replaces least recently used memories when at capacity
- **Usage Tracking**: Updates usage counters on each retrieval
- **FIFO Alternative**: Optional first-in-first-out replacement

**Implementation Details**:
- `keys`: (capacity, key_dim) tensor storing all memory keys
- `values`: List storing associated values
- `usage`: (capacity,) tensor tracking usage for LRU
- `pointer`: Circular buffer pointer for insertions
- `size`: Current number of stored memories

**Memory Requirements**: ~50MB for 1000 memories with 64-dim keys

#### Meta-Adaptation Network (`meta_adaptation.py`)
LSTM-based meta-learning network that adapts curiosity parameters based on exploration outcomes.

**Architecture**:
- **Input**: Exploration outcome sequences (reward patterns, curiosity effectiveness)
- **LSTM Layer**: 128 hidden units for temporal pattern recognition
- **Output Layer**: Fully connected layer producing curiosity weighting parameters
- **Dual Learning Rates**:
  - Fast adaptation: 0.1 (for quick strategy changes)
  - Slow meta-learning: 0.01 (for stable meta-parameter updates)

**Training Process**:
1. Collect exploration outcomes over multiple episodes
2. Feed sequences to LSTM for temporal pattern analysis
3. Output curiosity parameters (e.g., β values for reward weighting)
4. Meta-update network based on agent performance feedback

**Memory Requirements**: ~5MB for network parameters

### 4. Utility Systems (`mace_rl/utils/`)

#### Hybrid Reward System (`reward_system.py`)
Combines extrinsic and intrinsic rewards with adaptive weighting.

**Formula**: `R_total = R_extrinsic + β × R_curiosity`

**Adaptive β Scheduling**:
- Initial value: Configurable (default: 1.0)
- Decay schedule: Exponential decay with configurable rate (default: 0.999)
- Minimum value: Prevents β from becoming too small (default: 0.1)
- Update frequency: Typically every episode or batch of episodes

**Usage Pattern**:
```python
reward_system = HybridRewardSystem(curiosity_module, beta_initial=1.0)
total_reward = reward_system.get_total_reward(state, action, extrinsic_reward)
reward_system.update_beta()  # Decay β over time
```

#### General Utilities (`utils.py`)
Essential utility functions for the entire codebase.

**Functions**:
- `set_seed(seed)`: Ensures reproducible results across PyTorch, NumPy, and Python random
  - Sets deterministic CUDA operations
  - Disables CUDA benchmarking for consistency
  
- `log_data(log_file, data)`: CSV logging with automatic header detection
  - Appends data to existing files
  - Creates headers on first write
  
- `profile(func)`: Decorator for timing function execution
  - Measures wall-clock time
  - Prints execution duration
  - Useful for performance optimization

## Environment Setup

### System Requirements
- **Python**: 3.7-3.9 (3.8 recommended)
- **GPU**: Optional but recommended (CUDA-compatible)
- **RAM**: Minimum 8GB, recommended 16GB
- **Storage**: ~2GB for dependencies and datasets

### Installation Steps

1. **Clone the repository** (if from git):
```bash
git clone <repository-url>
cd mace_rl
```

2. **Create virtual environment**:
```bash
python -m venv mace_env
source mace_env/bin/activate  # Linux/Mac
# or
mace_env\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Dependency Details
- `torch==1.13.1`: Core deep learning framework
- `gym==0.26.2`: OpenAI Gym for RL environments
- `gym-minigrid==1.2.1`: Procedural grid-world environments
- `pybullet==3.2.5`: Physics simulation environments
- `atari-py==0.3.0`: Atari 2600 game environments
- `numpy==1.24.2`: Numerical computing
- `matplotlib==3.6.3`: Plotting and visualization
- `tensorboard==2.11.2`: Training visualization (optional)
- `opencv-python`: Image processing for environment wrappers

### Platform-Specific Notes

**Linux**: All environments supported, recommended for development

**Windows**: May require additional setup for Atari environments:
```bash
pip install atari-py[all]  # Install ROM dependencies
```

**Mac**: PyBullet may require additional dependencies:
```bash
# For M1/M2 Macs, use conda for better compatibility
conda install pybullet -c conda-forge
```

## Training Scripts

### Main Training Script (`train.py`)
The primary entry point for training MACE-RL agents with all components enabled.

**Command Line Arguments**:
- `--seed`: Random seed for reproducibility (default: 42)
- `--env-name`: Environment identifier (default: 'PongNoFrameskip-v4')
- `--max-episodes`: Total training episodes (default: 1000)
- `--max-timesteps`: Maximum steps per episode (default: 10000)
- `--memory-capacity`: Episodic memory size (default: 1000)
- `--lr-actor`: Actor network learning rate (default: 0.0003)
- `--lr-critic`: Critic network learning rate (default: 0.001)
- `--gamma`: Reward discount factor (default: 0.99)
- `--k-epochs`: PPO update epochs (default: 4)
- `--eps-clip`: PPO clipping parameter (default: 0.2)
- `--beta-start`: Initial curiosity weight (default: 1.0)
- `--save-interval`: Model checkpoint frequency (default: 100 episodes)
- `--log-file`: CSV log file path (optional)

**Training Flow**:
1. **Environment Setup**: Automatically detects environment type and applies appropriate wrappers
2. **Component Initialization**: Creates all MACE-RL components based on environment characteristics
3. **Episode Loop**:
   - Reset environment and collect trajectory
   - Calculate intrinsic rewards using curiosity module
   - Store experiences in episodic memory
   - Update PPO agent with hybrid rewards
   - Adapt curiosity parameters every 10 episodes
4. **Logging and Checkpointing**: Save training progress and model weights

**Memory Considerations**:
- Atari: Disables episodic memory and curiosity (CNN feature complexity)
- MiniGrid/PyBullet: Full MACE-RL pipeline enabled
- Total memory usage: 8-12GB during training

### Ablation Study Script (`ablation_train.py`)
Systematic evaluation of individual MACE-RL components through controlled ablation.

**Additional Arguments**:
- `--no-episodic-memory`: Disables episodic memory component
- `--no-curiosity`: Disables curiosity-driven exploration
- `--no-meta-adaptation`: Disables meta-learning adaptation

**Ablation Configurations**:
1. **Full MACE-RL**: All components enabled (baseline)
2. **No Memory**: Curiosity without episodic memory context
3. **No Curiosity**: Standard PPO with meta-adaptation only
4. **No Meta-Adaptation**: Static curiosity parameters
5. **PPO Only**: Pure PPO without any MACE-RL components

**Experimental Design**:
- Each configuration runs with identical hyperparameters
- Multiple seeds for statistical significance
- Consistent evaluation protocol across ablations

### Baseline Training Scripts

#### PPO Baseline (`baseline_ppo.py`)
Pure PPO implementation without MACE-RL enhancements.

**Purpose**: Establishes performance baseline for comparison
**Features**:
- Standard PPO algorithm implementation
- No curiosity or memory augmentation
- Identical network architecture to MACE-RL for fair comparison
- Same hyperparameter ranges for controlled experiments

#### A2C Baseline (`baseline_a2c.py`)
Advantage Actor-Critic baseline with simplified update mechanism.

**Purpose**: Evaluates impact of PPO-specific improvements
**Characteristics**:
- Online learning (no experience replay)
- Single-step policy updates
- Lower computational requirements
- Faster training but potentially less stable

**Hyperparameter Differences**:
- Lower learning rates (actor: 0.0001, critic: 0.0005)
- No K-epoch updates or clipping
- Direct entropy regularization

## How Components Interconnect

### Data Flow Architecture

```
Environment → State → [PPO Agent] → Action → Environment
                ↓
         [Episodic Memory] ← State + Experience
                ↓
         [Curiosity Module] ← Memory Retrieval + Novelty
                ↓
         [Hybrid Reward] ← Extrinsic + Intrinsic Rewards
                ↓
         [PPO Update] ← Enhanced Reward Signal
                ↓
    [Meta-Adaptation] ← Performance Feedback
                ↓
    [Parameter Update] → Curiosity Weighting
```

### Detailed Interaction Flow

1. **Episode Initialization**:
   ```python
   state = env.reset()
   episode_reward = 0
   ```

2. **Action Selection**:
   ```python
   action = ppo_agent.select_action(state)  # Uses current policy
   ```

3. **Environment Interaction**:
   ```python
   next_state, extrinsic_reward, done, info = env.step(action)
   ```

4. **Curiosity Calculation** (if enabled):
   ```python
   # Curiosity module computes intrinsic motivation
   curiosity_bonus = curiosity_module(state, action)
   # Combines novelty, memory relevance, and temporal factors
   ```

5. **Hybrid Reward Computation**:
   ```python
   total_reward = reward_system.get_total_reward(state, action, extrinsic_reward)
   # total_reward = extrinsic_reward + β × curiosity_bonus
   ```

6. **Experience Storage**:
   ```python
   # Store in PPO rollout buffer
   ppo_agent.buffer.rewards.append(total_reward)
   ppo_agent.buffer.is_terminals.append(done)
   
   # Store in episodic memory
   memory_value = torch.cat([next_state, extrinsic_reward])
   episodic_memory.add(state, memory_value)
   ```

7. **Policy Update** (end of episode):
   ```python
   ppo_agent.update()  # Multi-epoch policy optimization
   ```

8. **Meta-Adaptation** (periodic):
   ```python
   if episode % 10 == 0:
       reward_system.update_beta()  # Adapt curiosity weighting
   ```

### Memory Management Strategy

**During Training**:
- **PPO Buffer**: Cleared after each episode update (~10MB)
- **Episodic Memory**: Persistent across episodes, managed by LRU (~50MB)
- **Curiosity State Buffer**: Rolling window of recent states (~40MB)
- **Network Parameters**: Static during episode, updated during learning (~100MB)

**Memory Allocation Timeline**:
```
Episode Start:  Allocate trajectory buffers
Episode Middle: Fill buffers, query memories
Episode End:    Update networks, clear trajectory buffers
Every 10 Episodes: Meta-adaptation, memory consolidation
Every 100 Episodes: Model checkpointing
```

## Usage Examples

### Basic Training

**1. Train MACE-RL on MiniGrid**:
```bash
python train.py \
    --env-name MiniGrid-Empty-8x8-v0 \
    --max-episodes 2000 \
    --memory-capacity 500 \
    --log-file minigrid_training.csv
```

**2. Train on PyBullet Continuous Control**:
```bash
python train.py \
    --env-name HalfCheetahBulletEnv-v0 \
    --max-episodes 1500 \
    --memory-capacity 1000 \
    --lr-actor 0.0003 \
    --lr-critic 0.001 \
    --log-file halfcheetah_training.csv
```

**3. Train PPO Baseline for Comparison**:
```bash
python baseline_ppo.py \
    --env-name MiniGrid-Empty-8x8-v0 \
    --max-episodes 2000 \
    --log-file ppo_baseline.csv
```

### Ablation Studies

**Test Individual Components**:
```bash
# Without episodic memory
python ablation_train.py \
    --env-name MiniGrid-DoorKey-5x5-v0 \
    --no-episodic-memory \
    --log-file ablation_no_memory.csv

# Without curiosity
python ablation_train.py \
    --env-name MiniGrid-DoorKey-5x5-v0 \
    --no-curiosity \
    --log-file ablation_no_curiosity.csv

# Without meta-adaptation
python ablation_train.py \
    --env-name MiniGrid-DoorKey-5x5-v0 \
    --no-meta-adaptation \
    --log-file ablation_no_meta.csv
```

### Hyperparameter Sweeps

**Learning Rate Sensitivity**:
```bash
for lr in 0.0001 0.0003 0.001 0.003; do
    python train.py \
        --env-name MiniGrid-Empty-6x6-v0 \
        --lr-actor $lr \
        --lr-critic $(echo "$lr * 3" | bc) \
        --log-file lr_${lr}_training.csv \
        --seed 42
done
```

**Memory Capacity Analysis**:
```bash
for capacity in 100 500 1000 2000; do
    python train.py \
        --env-name MiniGrid-FourRooms-v0 \
        --memory-capacity $capacity \
        --log-file memory_${capacity}_training.csv \
        --seed 42
done
```

### Multi-Seed Evaluation

**Statistical Significance Testing**:
```bash
#!/bin/bash
for seed in 42 123 456 789 999; do
    python train.py \
        --env-name MiniGrid-DoorKey-6x6-v0 \
        --seed $seed \
        --max-episodes 1500 \
        --log-file mace_seed_${seed}.csv &
    
    python baseline_ppo.py \
        --env-name MiniGrid-DoorKey-6x6-v0 \
        --seed $seed \
        --max-episodes 1500 \
        --log-file ppo_seed_${seed}.csv &
done
wait
```

### Results Visualization

**Compare Training Curves**:
```bash
python plot_results.py \
    --log-files mace_seed_42.csv ppo_seed_42.csv \
    --labels "MACE-RL" "PPO Baseline" \
    --title "Training Comparison: MiniGrid DoorKey"
```

## Experimental Setup

### Recommended Environment Progression

**Phase 1: Discrete Environments (Week 1-2)**
1. `MiniGrid-Empty-5x5-v0`: Basic navigation
2. `MiniGrid-DoorKey-5x5-v0`: Simple exploration
3. `MiniGrid-FourRooms-v0`: Memory-dependent navigation
4. `MiniGrid-KeyCorridorS3R1-v0`: Complex exploration

**Phase 2: Continuous Control (Week 3-4)**
1. `InvertedPendulumBulletEnv-v0`: Simple control
2. `HalfCheetahBulletEnv-v0`: Locomotion
3. `AntBulletEnv-v0`: Multi-limb coordination
4. `HumanoidBulletEnv-v0`: High-dimensional control

**Phase 3: Atari Games (Week 5-6)**
1. `PongNoFrameskip-v4`: Dense rewards (baseline)
2. `BreakoutNoFrameskip-v4`: Moderate exploration
3. `MontezumaRevengeNoFrameskip-v4`: Extreme sparse rewards
4. `PrivateEyeNoFrameskip-v4`: Complex exploration

### Hyperparameter Recommendations

**MiniGrid Environments**:
```
--max-episodes 2000
--max-timesteps 500
--memory-capacity 500
--lr-actor 0.0003
--lr-critic 0.001
--beta-start 1.0
```

**PyBullet Environments**:
```
--max-episodes 1500
--max-timesteps 1000
--memory-capacity 1000
--lr-actor 0.0003
--lr-critic 0.001
--beta-start 0.5
```

**Atari Environments**:
```
--max-episodes 5000
--max-timesteps 10000
--memory-capacity 0  # Disabled for Atari
--lr-actor 0.0001
--lr-critic 0.0005
--beta-start 0.1
```

### Performance Metrics

**Primary Metrics**:
- **Sample Efficiency**: Episodes to reach performance threshold
- **Final Performance**: Average reward over last 100 episodes
- **Training Stability**: Variance in performance across seeds

**Secondary Metrics**:
- **Memory Usage**: Peak RAM consumption during training
- **Training Time**: Wall-clock time per episode
- **Exploration Coverage**: Unique states visited (environment-specific)

**Evaluation Protocol**:
1. Train for specified episodes
2. Evaluate policy (no exploration) for 100 episodes
3. Report mean and standard deviation
4. Statistical significance testing (t-test, p<0.05)

### Computational Resources

**Minimum Configuration**:
- CPU: 4 cores, 2.5GHz
- RAM: 8GB
- GPU: Optional (1-2GB VRAM)
- Training time: 2-8 hours per experiment

**Recommended Configuration**:
- CPU: 8+ cores, 3.0GHz+
- RAM: 16GB+
- GPU: NVIDIA GTX 1080 / RTX 2060 or better (4-6GB VRAM)
- Training time: 30 minutes - 2 hours per experiment

**Memory Usage by Component**:
- Base PPO: ~200MB
- Episodic Memory: ~50MB per 1000 entries
- Curiosity Module: ~40MB for continuous environments
- Meta-Adaptation: ~5MB
- Environment Wrappers: ~100-500MB (varies by environment)

## Troubleshooting

### Common Installation Issues

**Issue**: Atari environments not working
```bash
# Solution: Install ROM dependencies
pip install gym[atari,accept-rom-license]
# or
ale-import-roms /path/to/rom/files
```

**Issue**: PyBullet compilation errors on Mac M1/M2
```bash
# Solution: Use conda instead of pip
conda install pybullet -c conda-forge
```

**Issue**: CUDA out of memory errors
```bash
# Solution: Reduce memory-intensive components
python train.py --memory-capacity 100  # Reduce memory size
# or use CPU-only training
export CUDA_VISIBLE_DEVICES=""
```

### Training Issues

**Issue**: Agent not learning (flat reward curves)
**Debugging Steps**:
1. Check reward scaling: `print(f"Reward range: {min_reward} to {max_reward}")`
2. Verify environment setup: `print(f"Action space: {env.action_space}")`
3. Monitor curiosity bonuses: Enable debug logging in `curiosity.py`
4. Validate memory retrieval: Check if episodic memory is being populated

**Issue**: Training too slow
**Solutions**:
1. Reduce episode length: `--max-timesteps 200`
2. Decrease memory capacity: `--memory-capacity 100`
3. Reduce PPO epochs: `--k-epochs 2`
4. Use GPU acceleration: Ensure CUDA is available

**Issue**: Memory usage growing over time
**Causes**:
1. Memory leaks in curiosity state buffer
2. Episodic memory not replacing old entries
3. PPO buffer not clearing properly

**Solution**:
```python
# Add explicit memory management
import gc
if episode % 100 == 0:
    gc.collect()
    torch.cuda.empty_cache()  # If using GPU
```

### Environment-Specific Issues

**MiniGrid**:
- **Issue**: Observation shape mismatch
- **Solution**: Verify wrapper is flattening observations correctly

**PyBullet**:
- **Issue**: Environment hanging or crashing
- **Solution**: Check PyBullet GUI mode (`p.connect(p.DIRECT)`)

**Atari**:
- **Issue**: Frame stacking errors
- **Solution**: Ensure frame_stack=True in wrap_deepmind call

### Performance Debugging

**Low Sample Efficiency**:
1. Increase curiosity weight: `--beta-start 2.0`
2. Larger memory capacity: `--memory-capacity 2000`
3. Lower learning rates: `--lr-actor 0.0001`

**Unstable Training**:
1. Reduce clipping: `--eps-clip 0.1`
2. More PPO epochs: `--k-epochs 8`
3. Higher discount factor: `--gamma 0.995`

**Memory Errors**:
1. Monitor memory usage: `nvidia-smi` (GPU), `htop` (CPU)
2. Reduce batch sizes in PPO update
3. Use gradient checkpointing for large networks

### Validation and Testing

**Unit Tests** (run individual components):
```python
# Test episodic memory
python -m mace_rl.modules.episodic_memory

# Test curiosity module
python -m mace_rl.modules.curiosity

# Test PPO agent
python -m mace_rl.agents.ppo
```

**Integration Testing**:
```bash
# Short training run to verify pipeline
python train.py --max-episodes 10 --max-timesteps 50 --env-name MiniGrid-Empty-5x5-v0
```

This completes the comprehensive documentation of the MACE-RL codebase. Each component has been explained in detail, including its purpose, implementation, memory requirements, and interaction with other components. The usage examples and troubleshooting guide should enable successful deployment and experimentation with the system.
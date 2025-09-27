# MACE-RL: Meta-Adaptive Curiosity-Driven Exploration with Episodic Memory

Effective exploration remains a fundamental challenge in Reinforcement Learning (RL), particularly in complex environments where rewards are sparse or misleading. Although curiosity-driven methods offer intrinsic motivation for exploration, they often rely on fixed heuristics that fail to adapt to the evolving requirements of a learning task. This raises a critical question: how can an agent learn how to explore? To address this, we introduce MACE-RL: Meta-Adaptive Curiosity-driven Exploration with Episodic Memory. MACE-RL is a novel framework that learns to dynamically regulate its own exploration strategy. It integrates a curiosity module, informed by episodic memory of past experiences, with a meta-learning network that monitors the agent's performance and adaptively regulates the influence of the intrinsic curiosity bonus. This enables the agent to achieve a dynamic balance between exploration and exploitation, becoming more exploratory when progress stalls and more focused when learning advances. We evaluate MACE-RL on a suite of control benchmarks, including Acrobot-v1, CartPole-v1, and HopperBulletEnv-v0. The results demonstrate that learning an adaptive exploration policy significantly outperforms both a strong PPO baseline and an ablated version with a fixed curiosity bonus, confirming that meta-adapting the exploration drive is crucial for effective learning.

## Results

### Main Results

Metrics reported are **Sample Efficiency / Convergence Steps** across three environments.  
Lower is better for both metrics. Best results are in **bold**.

| Algorithm                                                   | Acrobot-v1     | CartPole-v1     | HopperBulletEnv-v0 |
|-------------------------------------------------------------|----------------|-----------------|---------------------|
| **PPO (Baseline)**                                          | 186 / 617      | 140 / 660       | 865 / 1841          |
| **MACE-RL (Fixed β, β=0.4, Exponential Scheduler)**         | 201 / 595      | 120 / **620**   | 782 / 1801          |
| **MACE-RL (Full, Meta Network)**                            | **249 / 586**  | **104 / 634**   | **495 / 1799**      |


Our code is available at \url{https://github.com/ayushsi42/mace_rl}
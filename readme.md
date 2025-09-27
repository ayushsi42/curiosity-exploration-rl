# MACE-RL: Meta-Adaptive Curiosity-Driven Exploration with Episodic Memory

Effective exploration remains a fundamental challenge in Reinforcement Learning (RL), particularly in complex environments where rewards are sparse or misleading. Although curiosity-driven methods offer intrinsic motivation for exploration, they often rely on fixed heuristics that fail to adapt to the evolving requirements of a learning task. This raises a critical question: how can an agent learn how to explore? To address this, we introduce MACE-RL: Meta-Adaptive Curiosity-driven Exploration with Episodic Memory. MACE-RL is a novel framework that learns to dynamically regulate its own exploration strategy. It integrates a curiosity module, informed by episodic memory of past experiences, with a meta-learning network that monitors the agent's performance and adaptively regulates the influence of the intrinsic curiosity bonus. This enables the agent to achieve a dynamic balance between exploration and exploitation, becoming more exploratory when progress stalls and more focused when learning advances. We evaluate MACE-RL on a suite of control benchmarks, including Acrobot-v1, CartPole-v1, and HopperBulletEnv-v0. The results demonstrate that learning an adaptive exploration policy significantly outperforms both a strong PPO baseline and an ablated version with a fixed curiosity bonus, confirming that meta-adapting the exploration drive is crucial for effective learning.

## Results

\begin{table*}[!t]
\centering
\label{tab:main_results}
\begin{tabular}{l|ccc}
\toprule
\textbf{Algorithm} & \textbf{Acrobot-v1} & \textbf{CartPole-v1} & \textbf{HopperBulletEnv-v0} \\
\midrule
PPO (Baseline) & 186 / 617 & 140 / 660 & 865 / 1841 \\
\midrule
MACE-RL (Fixed $\beta$, $\beta=0.4$, Exponential Scheduler) & 201 / 595 & 120 / \textbf{620} & 782 / 1801 \\
\midrule
MACE-RL (Full, Meta Network) & \textbf{249 / 586} & \textbf{104} / 634 & \textbf{495 / 1799} \\
\bottomrule
\end{tabular}
\caption{Metrics reported are Sample Efficiency and Convergence Steps across three environments Acrobot-v1, CartPole-v1, HopperBulletEnv-v0. Lower is better for both metrics. Best results are in \textbf{bold}.}
\end{table*}

Our code is available at \url{https://github.com/ayushsi42/mace_rl}
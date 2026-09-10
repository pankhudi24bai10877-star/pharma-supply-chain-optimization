# PPO Study & Setup Notes
**Owner**: M4 (Prakhar Gupta)
**Week**: 1 (September 1–7, 2026)
**Deliverable references**: C3 (PPO agent), C4 (CVaR reward), C5 (Hyperparameter tuning), C6 (Baselines)

## 1. Environment Setup
- **Python version requirement**: 3.10+
- **Virtual environment setup instructions**:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
  ```
- **Package installation**:
  ```bash
  pip install gymnasium stable-baselines3 torch tensorboard numpy scipy matplotlib seaborn pandas pyyaml pytest
  ```
- **Verification steps**: Ensure imports succeed, verify package versions, and check CUDA availability.
  ```python
  import gymnasium as gym
  import stable_baselines3
  import torch
  print(f"Gymnasium: {gym.__version__}")
  print(f"Stable-Baselines3: {stable_baselines3.__version__}")
  print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
  ```
- Document that this was completed in Week 1.

## 2. PPO Algorithm Study Notes
Comprehensive study notes on PPO based on Paper 4 (Schulman et al., 2017).

### 2.1 Background: Policy Gradient Methods
- **REINFORCE and vanilla policy gradient**: Methods that directly optimize the policy by estimating gradients of expected returns.
- **Trust Region Policy Optimization (TRPO)**: Predecessor to PPO, introduced trust regions to prevent destructively large policy updates, but complex to implement and compute.
- **Motivation for PPO**: Simpler, more general, and exhibits better empirical performance than TRPO while retaining its reliability.

### 2.2 PPO Core Algorithm
- **Clipped Surrogate Objective**:
  $$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta) \cdot A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t)]$$
  where $$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$
- **Why clipping works**: Prevents destructively large policy updates by bounding the probability ratio, ensuring the new policy does not stray too far from the old policy.
- **Alternative**: KL penalty version (less commonly used).

### 2.3 Advantage Estimation
- **Generalized Advantage Estimation (GAE)**:
  $$A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \cdot \delta_{t+l}$$
  where $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
- **Trade-off**: $\lambda=0$ (high bias, low variance) vs $\lambda=1$ (low bias, high variance).
- **Default**: $\lambda=0.95$ optimally balances this trade-off.

### 2.4 Actor-Critic Architecture
- **Shared vs. separate feature extractors**: Depending on the problem, the actor and critic can share base networks or have completely separate ones.
- **Policy head**: Outputs action distribution (Gaussian for continuous action spaces).
- **Value head**: Outputs state value $V(s)$.
- **Loss**: $$L = L^{CLIP} - c_1 \cdot L^{VF} + c_2 \cdot H[\pi]$$ (includes an entropy bonus to encourage exploration).

## 3. Stable-Baselines3 PPO Design

### 3.1 SB3 PPO API
SB3's PPO will be integrated as follows:
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
```

### 3.2 Planned Hyperparameters
Table of hyperparameters we plan to use (to be tuned in Week 12):

| Hyperparameter | Default | Range to Explore | Notes |
|---|---|---|---|
| learning_rate | 3e-4 | [1e-5, 1e-3] | Adam optimizer |
| n_steps | 2048 | [512, 4096] | Steps per rollout |
| batch_size | 64 | [32, 256] | Mini-batch size |
| n_epochs | 10 | [3, 30] | PPO update epochs |
| gamma | 0.99 | [0.95, 0.999] | Discount factor |
| gae_lambda | 0.95 | [0.9, 0.99] | GAE lambda |
| clip_range | 0.2 | [0.1, 0.3] | PPO clip parameter |
| ent_coef | 0.01 | [0.0, 0.05] | Entropy coefficient |
| vf_coef | 0.5 | [0.25, 1.0] | Value function coefficient |
| max_grad_norm | 0.5 | [0.3, 1.0] | Gradient clipping |
| policy_net | [64,64] | [64,64], [128,128], [256,256] | MLP hidden layers |

### 3.3 Planned Training Pipeline
- **Vectorized environments**: `DummyVecEnv` for debugging, `SubprocVecEnv` for speed.
- **Callbacks**: `EvalCallback`, `CheckpointCallback`, `TensorBoard`.
- **Model saving/loading**: For checkpointing.
- **Logging**: TensorBoard for tracking training curves.

### 3.4 Risk-Aware Reward Design (Preview)
- CVaR augmentation of reward signal (to be implemented in C4, Week 10).
- $$r_{risk} = (1-\lambda) \cdot r_{base} + \lambda \cdot CVaR_\alpha(r_{base})$$
- $\lambda$ controls risk-awareness: $\lambda=0 \rightarrow$ risk-neutral, $\lambda=1 \rightarrow$ fully risk-averse.
- $\alpha$ is the CVaR confidence level (typically 0.05 for 5% worst-case).
- This will be the key differentiator of our approach.

## 4. CartPole Validation Plan (Week 2)
Describe the plan to validate SB3 installation by training PPO on CartPole:
- **Environment**: `CartPole-v1`
- **Goal**: Achieve reward >= 475 (near-optimal).
- This validates that SB3 + PyTorch + Gymnasium work correctly together.
- Not yet executed — scheduled for Week 2.

## 5. Week 1 Status and Next Steps
- ✅ Python RL environment set up (Gymnasium, SB3, PyTorch installed)
- ✅ Dependencies verified (all imports work)
- ✅ PPO algorithm studied (Schulman et al., 2017)
- ✅ PPO hyperparameter plan designed
- ✅ SB3 API studied
- ⬜ Week 2: Run PPO on CartPole to validate full pipeline
- ⬜ Week 2: Write Section 3.2 PPO Agent Design for paper
- ⬜ Week 9-10: Implement PPO agent for HealthcareSCEnv
- ⬜ Week 10: Implement CVaR reward weighting
- ⬜ Week 12: Hyperparameter tuning pipeline

## 6. References
- **Paper 4**: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347
- **Paper 6**: Rolf, B. et al. — RL in Supply Chain Management review
- **Stable-Baselines3 docs**: https://stable-baselines3.readthedocs.io/
- **Gymnasium docs**: https://gymnasium.farama.org/
# Section 3.2.2 — PPO Agent Design (Draft)

## 3.2.2 Proximal Policy Optimization Agent

We employ Proximal Policy Optimization (PPO) [Schulman et al., 2017] as the reinforcement learning algorithm for operational supply chain decisions. PPO is selected for its strong empirical performance on continuous control tasks, implementation stability, and compatibility with our hybrid framework.

**Policy Architecture.** The agent employs an actor-critic architecture with separate multi-layer perceptron (MLP) networks for the policy (actor) and value function (critic). Both networks consist of two hidden layers with 64 units each and tanh activation functions. The policy network outputs a Gaussian distribution over the continuous action space $a_t \in [0,1]^{P \times K}$, parameterized by a mean vector and a learned diagonal covariance matrix. The value network estimates the state value function $V(s_t)$ used for advantage computation.

**Training Objective.** PPO maximizes the clipped surrogate objective:
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio, $\hat{A}_t$ is the estimated advantage, and $\epsilon = 0.2$ is the clipping parameter. This objective prevents destructively large policy updates while maintaining first-order optimization simplicity.

**Advantage Estimation.** We use Generalized Advantage Estimation (GAE) [Schulman et al., 2016]:
$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the temporal difference error, $\gamma = 0.99$ is the discount factor, and $\lambda = 0.95$ balances bias and variance in advantage estimation.

**Combined Loss.** The total training objective combines the policy surrogate, value function, and entropy terms:
$$L(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 H[\pi_\theta]$$

where $L^{VF}$ is the mean squared value function error, $H[\pi_\theta]$ is the policy entropy bonus encouraging exploration, $c_1 = 0.5$ is the value function coefficient, and $c_2 = 0.01$ is the entropy coefficient.

**Training Procedure.** The agent collects $N = 2048$ environment steps per rollout using the current policy. Each rollout buffer is then used for $K = 10$ epochs of mini-batch gradient descent with batch size 64 and learning rate $3 \times 10^{-4}$ (Adam optimizer). Gradient norms are clipped at 0.5 to ensure training stability.

**Risk-Aware Extension.** In the full hybrid framework, the base reward signal will be augmented with a Conditional Value-at-Risk (CVaR) component (Section 3.3), producing a risk-sensitive training signal. This extension is implemented in the reward function rather than the PPO algorithm itself, preserving the standard policy optimization procedure.

*Table I: PPO Hyperparameters*

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning rate | $3 \times 10^{-4}$ | Adam optimizer step size |
| Rollout steps | 2048 | Steps collected per update |
| Mini-batch size | 64 | SGD mini-batch size |
| Training epochs | 10 | Epochs per rollout |
| Discount factor ($\gamma$) | 0.99 | Future reward discounting |
| GAE $\lambda$ | 0.95 | Advantage estimation bias-variance |
| Clip range ($\epsilon$) | 0.2 | PPO surrogate clipping |
| Entropy coefficient | 0.01 | Exploration bonus weight |
| Value function coefficient | 0.5 | Value loss weight |
| Max gradient norm | 0.5 | Gradient clipping threshold |
| Network architecture | [64, 64] | Hidden layer sizes (actor & critic) |

[References: Schulman et al. 2017 — PPO; Schulman et al. 2016 — GAE]

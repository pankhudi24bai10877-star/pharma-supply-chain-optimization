# Section 4 — Experimental Setup

## 4. Experimental Setup

### 4.1 Dataset Description

We use a synthetic pharmaceutical supply chain dataset modeling three pharmaceutical products managed through two supplier channels over a planning horizon of T = 365 days. Demand for each product follows a Negative Binomial distribution, D_{t,p} ~ NegBin(n_p, p_p), with product-specific parameters calibrated from pharmaceutical demand literature [ref]. The default parameters are:

| Product | n (shape) | p (probability) | Mean Daily Demand |
|---------|-----------|-----------------|-------------------|
| Product 1 | 5 | 0.30 | ~11.7 units |
| Product 2 | 8 | 0.40 | ~12.0 units |
| Product 3 | 3 | 0.25 | ~9.0 units |

The Negative Binomial distribution is chosen over Poisson as it captures the overdispersion commonly observed in pharmaceutical demand patterns, where variance exceeds the mean due to irregular prescription patterns and seasonal effects.

Supply disruptions will be modeled as Bernoulli processes with supplier-specific disruption probabilities (to be implemented). Lead times follow Log-normal distributions (to be implemented). These uncertainty sources will be calibrated using parameters from Group B's data pipeline.

### 4.2 Environment Configuration

The HealthcareSCEnv implements a Gymnasium-compatible MDP with the following configuration:

**State Space.** The observation vector s_t ∈ R^{15} comprises:
- Inventory levels per product (3 dimensions, normalized by max_inventory = 500)
- Pipeline inventory per product (3 dimensions)
- Most recent demand per product (3 dimensions, normalized by max_demand = 100)
- Shelf life indicators per product (3 dimensions, placeholder = 1.0)
- Supplier availability status (2 dimensions, binary)
- Normalized time index (1 dimension)

**Action Space.** The agent selects order quantities a_t ∈ [0,1]^6, representing normalized orders for each product-supplier pair (3 products × 2 suppliers). Actions are scaled to physical quantities by max_order_qty = 500.

**Cost Parameters.**

| Parameter | Value | Description |
|-----------|-------|-------------|
| c_hold | 0.5 | Holding cost per unit per day |
| c_stock | 10.0 | Stockout penalty per unit of unmet demand |
| c_order | 1.0 | Ordering cost per unit ordered |
| c_waste | 5.0 | Wastage cost per expired unit (future) |
| Initial inventory | 100 | Starting inventory per product |

The stockout penalty is set significantly higher than the holding cost (ratio 20:1) to reflect the critical nature of pharmaceutical products where stockouts may impact patient outcomes.

### 4.3 Baselines

We compare our hybrid CVaR-augmented PPO framework against the following baselines:

**1. (s, S) Inventory Policy.** A classical operations research inventory control policy parameterized by reorder point s and order-up-to level S for each product. When the inventory level of product p drops to or below s_p, an order is placed to restore inventory to S_p. Orders are distributed equally across available suppliers. Default parameters are s_p = 50 and S_p = 200 for all products, representing a conservative replenishment strategy.

**2. Risk-Neutral PPO.** A standard PPO agent trained with the base cost-minimization reward without CVaR augmentation (λ = 0). This ablation baseline isolates the contribution of the risk-aware component by using the same policy architecture and training procedure but without risk-sensitive reward shaping.

### 4.4 Implementation Details

The PPO agent is implemented using Stable-Baselines3 with the hyperparameters listed in Table I (Section 3.2.2). The actor-critic network uses two hidden layers of 64 units each with tanh activations. Training uses the Adam optimizer with learning rate 3 × 10^{-4}, collecting 2048 environment steps per rollout and performing 10 epochs of mini-batch updates with batch size 64.

All experiments are implemented in Python 3.13 using PyTorch 2.10.0 for neural network computations, Gymnasium 1.3.0 for the environment interface, and Stable-Baselines3 2.9.0 for the PPO implementation. Hardware specifications will be reported with final results.

### 4.5 Evaluation Protocol

All methods are evaluated over N = 500 episodes with 5 independent random seeds [42, 123, 456, 789, 1024] for reproducibility. We report mean ± standard deviation with 95% bootstrap confidence intervals (10,000 resamples). Statistical significance is assessed using Welch's t-test at significance level α = 0.05 with Bonferroni correction for multiple comparisons. Effect sizes are quantified using Cohen's d, with thresholds of d = 0.2 (small), d = 0.5 (medium), and d = 0.8 (large).

### 4.6 Metrics

We evaluate performance across three categories:

**Cost Metrics:**
- Total cost: sum of holding, stockout, ordering, and wastage costs over the episode
- Component-wise costs for detailed analysis

**Service Metrics:**
- Service level (fill rate): fraction of total demand fulfilled, 1 - (stockout units / total demand)
- Stockout frequency: fraction of time steps with any stockout event
- Inventory turnover: total demand / average inventory level

**Risk Metrics:**
- CVaR_{0.05}: expected cost in the worst 5% of episodes
- VaR_{0.05}: cost threshold exceeded in 5% of episodes
- Cost variance: Var(total cost) across evaluation episodes
- Worst-case cost: maximum total cost observed

Results will be reported after full training and evaluation are completed.
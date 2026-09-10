# HealthcareSCEnv — MDP Design Document
**Owner:** M3 (Devraj Srivastava)  
**Week:** 1 (September 1–7, 2026)  
**Deliverable references:** C1 (HealthcareSCEnv), C2 (Environment unit tests)  

### 1. Overview
- This document presents the Markov Decision Process (MDP) design for the HealthcareSCEnv, a Gymnasium-compatible RL environment for pharmaceutical supply chain optimization.
- The environment models a multi-product, multi-supplier pharmaceutical supply chain subject to stochastic demand, supply disruptions, inventory perishability, and variable lead times.

### 2. MDP Formulation

#### 2.1 State Space S
Define the state vector components:
- **Inventory levels** $I_t \in \mathbb{R}^P$ (current on-hand inventory for $P$ products)
- **Pipeline inventory** $Q_t \in \mathbb{R}^P$ (orders in transit, not yet received)
- **Demand observation** $D_{t-1} \in \mathbb{R}^P$ (most recent realized demand)
- **Supplier status** $\sigma_t \in \{0,1\}^K$ (availability of $K$ suppliers; 1=available, 0=disrupted)
- **Time index** $\tau_t \in [0,1]$ (normalized current time step in episode)
- **Shelf life** $L_t \in \mathbb{R}^P$ (remaining shelf life of oldest batch per product)

Total state dimension: $4P + K + 1$
Default: $P=3$ products, $K=2$ suppliers $\rightarrow$ dim=15

All components normalized to [0,1] for neural network compatibility.

#### 2.2 Action Space A
- **Order quantities** $a_t \in [0,1]^{P \times K}$ (normalized order quantities for each product-supplier pair)
- Continuous Box space, will be scaled to actual quantities by environment
- Default dimension: $3 \times 2 = 6$

#### 2.3 Transition Dynamics T(s'|s,a)
Describe each stochastic component:
- **Demand**: $D_t \sim \text{NegBin}(r, p)$ per product (calibrated from Paper 1 data)
- **Supply disruptions**: $\sigma_{t+1,k} \sim \text{Bernoulli}(p_{\text{disrupt\_}k})$ per supplier
- **Lead times**: $\tau_{\text{lead}} \sim \text{LogNormal}(\mu, \sigma^2)$ per order
- **Inventory update**: $I_{t+1} = \max(0, I_t + R_t - D_t - E_t)$
  - $R_t$ = received orders (those past lead time)
  - $E_t$ = expired inventory (shelf life exceeded)
- **Pipeline update**: $Q_{t+1} = Q_t - R_t + \text{new orders placed}$ (subject to disruption)

#### 2.4 Reward Function R(s,a,s')
- **Base reward**: $r_t = -(c_{\text{hold}} \cdot \max(I_t - D_t, 0) + c_{\text{stock}} \cdot \max(D_t - I_t, 0) + c_{\text{order}} \cdot \|a_t\| + c_{\text{waste}} \cdot E_t)$
- Cost components:
  - Holding cost: $c_{\text{hold}}$ per unit per period
  - Stockout penalty: $c_{\text{stock}}$ per unit of unmet demand (high for critical pharmaceuticals)
  - Ordering cost: $c_{\text{order}}$ per unit ordered
  - Wastage cost: $c_{\text{waste}}$ per expired unit
- **CVaR augmentation** (Week 10, M4): $r_{\text{cvar}} = (1-\lambda) \cdot r_t + \lambda \cdot \text{CVaR}_\alpha(r_t)$ where $\lambda$ is the risk-weighting parameter and $\alpha$ is the CVaR confidence level

#### 2.5 Episode Structure
- **Horizon**: $T = 365$ time steps (1 year of daily decisions)
- **Initial state**: Inventory at nominal levels, no pipeline orders, all suppliers available
- **Termination**: Episode ends when $t \geq T$
- **No early termination** (all episodes run full horizon for fair comparison)

### 3. Gymnasium API Design
- Class: `HealthcareSCEnv(gym.Env)`
- `__init__(num_products, num_suppliers, max_steps, render_mode, config)`
- `reset(seed, options) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`
- `render() -> str or None`
- `close()`
- metadata: render_modes=['human', 'ansi'], render_fps=4

### 4. Configuration Parameters
Table of environment parameters with default values and descriptions:
| Parameter | Default | Description |
|-----------|---------|-------------|
| num_products | 3 | Number of pharmaceutical products |
| num_suppliers | 2 | Number of available suppliers |
| max_steps | 365 | Planning horizon (days) |
| c_hold | 0.5 | Holding cost per unit per day |
| c_stock | 10.0 | Stockout penalty per unit |
| c_order | 1.0 | Ordering cost per unit |
| c_waste | 5.0 | Wastage cost per expired unit |
| shelf_life | 180 | Shelf life in days |
| max_order_qty | 500 | Maximum order quantity per product-supplier |

### 5. Data Dependencies
- Demand distribution parameters $\rightarrow$ from Group B (M1), Deliverable B3
- Supply disruption probabilities $\rightarrow$ from Group B (M1)
- Lead time distribution parameters $\rightarrow$ from Group B (M1)
- SP constraints (safety stocks, capacities) $\rightarrow$ from Group A (M2), via M5's bridge

### 6. Week 1 Status and Next Steps
- ✅ MDP formulation complete (state, action, transition, reward)
- ✅ Gymnasium API design complete
- ✅ Configuration parameters defined
- ✅ Stub implementation created with space definitions
- ⬜ Week 2: Implement reset() with proper initial state, begin step() logic
- ⬜ Week 3: Add demand dynamics, disruption logic
- ⬜ Week 4: Add perishability, lead times
- ⬜ Week 8: Full environment with unit tests (C1, C2)

### 7. References
- Paper 7: Saha & Ray — MDP formulation for healthcare supply chain
- Paper 1: Saha & Rathore — MARL environment design, demand modeling
- Paper 4: Schulman et al. — PPO (relevant for env-agent interface)
- Gymnasium documentation: https://gymnasium.farama.org/
# Section 3.2 — Reinforcement Learning Environment (Draft)

### 3.2 Reinforcement Learning Environment

We formulate the pharmaceutical supply chain operational decision problem as a Markov Decision Process (MDP) and implement it as a Gymnasium-compatible environment, HealthcareSCEnv.

**State Space.** The state at time step $t$ is defined as:
$s_t = [I_t, Q_t, D_{t-1}, \sigma_t, \tau_t, L_t]$

where $I_t \in \mathbb{R}^P$ represents current inventory levels for $P$ pharmaceutical products, $Q_t \in \mathbb{R}^P$ denotes pipeline inventory (orders in transit), $D_{t-1} \in \mathbb{R}^P$ captures the most recently observed demand, $\sigma_t \in \{0,1\}^K$ indicates the availability status of $K$ suppliers, $\tau_t \in [0,1]$ is the normalized time index, and $L_t \in \mathbb{R}^P$ tracks remaining shelf life of the oldest batch per product. All state components are normalized to $[0,1]$ to facilitate neural network training. The total state dimension is $4P + K + 1$.

**Action Space.** At each time step, the agent selects order quantities $a_t \in [0,1]^{P \times K}$, representing normalized order quantities for each product-supplier pair. These are scaled to physical quantities by the environment based on maximum order capacity constraints, which may be informed by the SP module's strategic decisions (Section 3.1).

**Transition Dynamics.** The environment incorporates three sources of uncertainty:
1. *Stochastic demand*: Product demands follow Negative Binomial distributions, $D_{t,p} \sim \text{NegBin}(r_p, q_p)$, calibrated from pharmaceutical demand data [refs].
2. *Supply disruptions*: Supplier availability evolves as $\sigma_{t+1,k} \sim \text{Bernoulli}(1 - p_{\text{disrupt},k})$, modeling random supply disruptions.
3. *Variable lead times*: Order delivery times follow Log-normal distributions, $\tau_{\text{lead}} \sim \text{LogNormal}(\mu_l, \sigma_l^2)$.

Inventory transitions follow: $I_{t+1} = \max(0, I_t + R_t - D_t - E_t)$, where $R_t$ represents received orders (past lead time) and $E_t$ denotes expired inventory exceeding shelf life.

**Reward Function.** The reward signal is defined as the negative total cost:
$r_t = -(c_h \cdot H_t + c_s \cdot S_t + c_o \cdot O_t + c_w \cdot W_t)$

where $H_t$ is holding cost, $S_t$ is stockout penalty, $O_t$ is ordering cost, and $W_t$ is wastage cost from expired products. This reward is subsequently augmented with a CVaR risk measure (Section 3.3) to produce a risk-aware training signal.

**Episode Structure.** Each episode spans a planning horizon of $T = 365$ days. The agent makes daily ordering decisions, receiving the current state observation and selecting actions to minimize long-term expected (risk-adjusted) costs.

[Note: Placeholder references — will be filled with proper citations]
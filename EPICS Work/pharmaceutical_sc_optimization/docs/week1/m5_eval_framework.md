# Evaluation Framework Design Document
**Owner:** M5 (Shourya Gupta)  
**Week:** 1 (September 1–7, 2026)  
**Deliverable references:** C7 (Integration), C8 (Evaluation), C9 (Sensitivity analysis), C10 (Ablation studies)  

## 1. Overview
This document defines the evaluation framework for comparing the hybrid CVaR-PPO agent against baselines in pharmaceutical supply chain optimization. The framework ensures rigorous, reproducible, and statistically sound evaluation.

## 2. Methods to Evaluate

| Method | Description | Owner | Type |
|--------|-------------|-------|------|
| Hybrid CVaR-PPO | SP constraints + CVaR-augmented PPO reward | M4+M5 | Our method |
| Risk-Neutral PPO | Standard PPO without CVaR reward augmentation | M4 | Ablation baseline |
| (s,S) Policy | Classic reorder-point/order-up-to inventory policy | M5 | Traditional OR baseline |

## 3. Performance Metrics

### 3.1 Supply Chain Metrics
- **Total Cost**: Sum of holding, stockout, ordering, and wastage costs over episode
- **Service Level (Fill Rate)**: Fraction of demand fulfilled = 1 - (stockout_units / total_demand)
- **Stockout Frequency**: Fraction of time steps with any stockout
- **Inventory Turnover**: Total demand / average inventory
- **Wastage Rate**: Expired units / total units procured
- **Average Inventory Level**: Mean inventory across products and time steps

### 3.2 RL Training Metrics
- **Cumulative Episode Reward**: Total reward per episode (mean ± std)
- **Convergence Rate**: Number of episodes/timesteps to reach 95% of asymptotic performance
- **Training Wall-Clock Time**: Total training time
- **Policy Entropy**: Entropy of action distribution (exploration measure)

### 3.3 Risk Metrics
- **CVaR_α(Cost)**: Conditional Value-at-Risk at α=0.05 (expected cost in worst 5% scenarios)
- **VaR_α(Cost)**: Value-at-Risk at α=0.05
- **Cost Variance**: Var(total_cost) across episodes
- **Worst-Case Cost**: Maximum total cost observed
- **Cost Distribution Skewness**: Measure of tail risk

## 4. Statistical Evaluation Protocol

### 4.1 Hypothesis Testing
For each pair of methods (A, B):
- **H₀**: Mean total cost of A = Mean total cost of B
- **H₁**: Mean total cost of A ≠ Mean total cost of B
- **Primary test**: Welch's t-test (unequal variances, α=0.05)
- **Non-parametric alternative**: Mann-Whitney U test
- **Multiple comparison correction**: Bonferroni correction when comparing >2 methods

### 4.2 Effect Size
- **Cohen's d**: Standardized mean difference
  - Small: d=0.2, Medium: d=0.5, Large: d=0.8
- Report for all pairwise comparisons

### 4.3 Confidence Intervals
- **Bootstrap CI**: 95% bootstrap confidence intervals (10,000 resamples)
- Report for all primary metrics

## 5. Reproducibility Protocol

### 5.1 Random Seeds
- Primary seeds: [42, 123, 456, 789, 1024]
- For each seed, run complete training + evaluation
- Report results aggregated across seeds

### 5.2 Episode Counts
- **Training**: Total timesteps TBD (hyperparameter, likely 500K-1M)
- **Evaluation**: 500 episodes per method per seed = 2500 total per method
- **Rationale**: 500 episodes provides sufficient statistical power for detecting medium effect sizes (d=0.5) at α=0.05 with power=0.99

### 5.3 Environment Versioning
- Lock all package versions in requirements.txt
- Record environment configuration in results metadata
- Store trained model checkpoints

## 6. Sensitivity Analysis Plan (C9)
Parameters to vary:
| Parameter | Symbol | Range | Steps | Rationale |
|-----------|--------|-------|-------|----------|
| CVaR weight | λ | [0, 0.2, 0.4, 0.6, 0.8, 1.0] | 6 | Risk-return trade-off |
| CVaR confidence | α | [0.01, 0.05, 0.10, 0.20] | 4 | Tail severity |
| Disruption probability | p_disrupt | [0.01, 0.05, 0.10, 0.20] | 4 | Supply uncertainty |
| Demand variability | CV_demand | [0.2, 0.4, 0.6, 0.8] | 4 | Demand uncertainty |

For each parameter value: train agent → evaluate 500 episodes → record metrics

## 7. Ablation Study Plan (C10)

| Ablation | What's Removed | Purpose |
|----------|---------------|--------|
| No CVaR | λ=0, standard PPO reward | Isolate CVaR contribution |
| No SP constraints | RL without SP-derived bounds | Isolate SP integration value |
| No perishability | Remove expiry mechanics | Assess perishability impact |
| No disruptions | p_disrupt=0 | Assess disruption modeling value |
| Single product | P=1 | Complexity scaling |

## 8. Results Visualization Plan
- Training curves: reward vs. timesteps (with confidence bands)
- Cost comparison: box plots by method
- Risk comparison: CVaR vs. total cost scatter plot
- Sensitivity: line plots of metric vs. parameter value
- Ablation: grouped bar charts
- Cost distribution: kernel density plots per method

## 9. Week 1 Status and Next Steps
- ✅ Evaluation framework designed
- ✅ Performance metrics defined
- ✅ Statistical evaluation protocol established
- ✅ Reproducibility protocol defined (seeds, episodes)
- ✅ Sensitivity and ablation plans outlined
- ⬜ Week 2: Code (s,S) baseline
- ⬜ Week 2: Write Section 4 Experimental Setup for paper
- ⬜ Week 13-15: Implement evaluation pipeline
- ⬜ Week 15-16: Run sensitivity analysis
- ⬜ Week 16: Run ablation studies
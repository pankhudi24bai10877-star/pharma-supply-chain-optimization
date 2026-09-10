# Pharmaceutical Supply Chain Optimization — Operational Layer (Group C)

## Project Title
**Hybrid CVaR-Augmented Stochastic Programming and Deep Reinforcement Learning for Pharmaceutical Supply Chain Optimization under Mixed Uncertainty**

## Group C Members
- **M3: Devraj Srivastava** — RL Environment Architecture (`HealthcareSCEnv`)
- **M4: Prakhar Gupta** — Deep RL Agent (`PPOAgent` via Stable-Baselines3)
- **M5: Shourya Gupta** — Baseline Implementation & Statistical Evaluation Framework

---

## Overview
This package contains the operational layer simulation and reinforcement learning agents for the pharmaceutical supply chain optimization project. It features:
- **`HealthcareSCEnv`**: A Gymnasium-compatible environment modeling multi-product, multi-supplier pharmaceutical inventory dynamics with stochastic demand, disruption states, and multi-component cost rewards.
- **`PPOAgent`**: A configurable wrapper around Stable-Baselines3 PPO, supporting continuous action spaces, custom MLP architectures, and reproducible training pipelines.
- **`SSPolicy`**: A classical $(s, S)$ inventory control baseline configured for multi-supplier distribution.
- **`EvaluationFramework`**: Rigorous statistical benchmarking tools supporting Welch's t-test, Mann-Whitney U tests, Bootstrap 95% confidence intervals, and CVaR calculations across fixed seeds.

---

## Directory Layout
- `configs/`: YAML configuration files (environment, agent, training, evaluation).
- `src/`: Core Python modules:
  - `environment/`: Gymnasium supply chain simulation environment.
  - `agents/`: PPO agent implementation and training routines.
  - `baselines/`: Classical operations research inventory policies.
  - `evaluation/`: Performance metrics, statistical tests, and report generators.
  - `integration/`: Bridge translating strategic SP constraints into RL bounds.
  - `utils/`: Shared utilities (seeds, config loaders).
- `tests/`: Pytest suite (52 unit and integration tests).
- `docs/`: Academic paper drafts (IEEE format) and architectural design specifications.
- `notebooks/`, `results/`, `logs/`: Experiment tracking and analysis artifacts.

---

## Quickstart

### Installation
```bash
pip install -r requirements.txt
```

### Running Tests
Execute the comprehensive test suite:
```bash
pytest tests/ -v
```

---

## Milestone Status (Weeks 1 & 2 Completed)
- [x] **Week 1:** Mathematical MDP design, state/action space specifications, Gymnasium API stubs, PPO architecture planning, evaluation protocol design.
- [x] **Week 2:** Full `HealthcareSCEnv` dynamics implemented (stochastic Negative Binomial demand, dynamic inventory updates, cost breakdown rewards); PPO agent validated on CartPole (reward ≥ 300) and verified with `HealthcareSCEnv`; working $(s, S)$ baseline policy running complete 365-day episodes; **52/52 automated tests passing**.

---

## Academic References
- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*
- Schulman et al. (2016) — *High-Dimensional Continuous Control Using Generalized Advantage Estimation*
- Saha & Ray (2019) — *Modelling and analysis of healthcare supply chain risk management*
- Saha & Rathore (2022) — *Multi-echelon pharmaceutical supply chain coordination under disruption*
- Rolf et al. (2022) — *A review on reinforcement learning algorithms and applications in SCM*

# Pharmaceutical Supply Chain Optimization

## Project Title
Hybrid CVaR-Augmented Stochastic Programming and Deep Reinforcement Learning for Pharmaceutical Supply Chain Optimization under Mixed Uncertainty

## Team Members
Group C:
- M3: Devraj Srivastava (RL Environment)
- M4: Prakhar Gupta (PPO Agent)
- M5: Shourya Gupta (Integration & Evaluation)

## Project Overview
This project implements a hybrid CVaR-SP + Deep RL framework for pharmaceutical supply chain optimization. The Stochastic Programming (SP) module provides strategic parameters, while the Deep RL (PPO) agent makes operational decisions considering CVaR-augmented risk measures under mixed uncertainties.

## Installation
```bash
pip install -r requirements.txt
```

## Project Structure
- `configs/`: Configuration files (YAML)
- `src/`: Source code modules (Environment, Agents, Baselines, Evaluation, Integration, Utils)
- `tests/`: Unit and integration tests
- `docs/`: Design and documentation files
- `notebooks/`, `results/`, `logs/`: Operational directories

## Week 1 Status
- M3: MDP Design and Environment stub defined (`healthcare_sc_env.py`)
- M4: PPO wrapper stub and study set up (`ppo_agent.py`)
- M5: Evaluation framework and integration bridge stubs defined (`metrics.py`, `sp_rl_bridge.py`)
- Foundational project structure and configs created.

## References
- Paper 4 (Schulman — PPO)
- Paper 1 (Saha & Rathore — MARL)
- Paper 7 (Saha & Ray — MDP)
- Paper 6 (Rolf — RL in SCM review)

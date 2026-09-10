# Hybrid CVaR-Augmented Stochastic Programming and Deep Reinforcement Learning for Pharmaceutical Supply Chain Optimization under Mixed Uncertainty

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0%2B-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-v2.1%2B-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

Managing pharmaceutical supply chains is a critical balancing act where decision failures have severe consequences. Unlike conventional retail supply chains, pharmaceutical inventory involves:
1. **Life-critical service levels:** Stockouts can lead to treatment delays or loss of patient life, demanding near-zero stockout tolerance.
2. **Perishability & strict shelf life:** Medicines expire, meaning excessive inventory results in heavy disposal and wastage costs.
3. **Mixed, high-impact uncertainties:** Volatile customer and hospital demand, abrupt supplier disruptions, and stochastic transportation lead times.

This project implements an end-to-end **two-tier hybrid optimization framework**:
- **Strategic Layer:** **Stochastic Programming (SP)** augmented with **Conditional Value-at-Risk (CVaR)** to account for catastrophic tail risks, setting robust baseline capacities, safety stock levels, and supply contracts.
- **Operational Layer:** A **Gymnasium**-compatible environment (`HealthcareSCEnv`) paired with **Deep Reinforcement Learning (PPO)** to make dynamic, day-to-day replenishment and allocation decisions under non-stationary real-world disruptions.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data_Pipeline [Data Engineering & Modeling]
        D1[Synthetic Data Generator] -->|Demand, Disruptions, Lead Times| D2[Validation & Statistical Calibration]
        D2 --> D3[Interface Spec JSON]
    end

    subgraph Strategic_Tier [Strategic Layer: Group A]
        D3 --> SP[Two-Stage Stochastic Programming]
        SP --> CVaR[CVaR Tail-Risk Augmentation]
        CVaR -->|Strategic Constraints & Safety Stock| Bridge[SP-RL Integration Bridge]
    end

    subgraph Operational_Tier [Operational Layer: Group C]
        Bridge --> Env[HealthcareSCEnv Gymnasium Simulation]
        Env <-->|State Observation, Reward| Agent[PPO Reinforcement Learning Agent]
        Env <-->|State Observation, Action| Baseline[Classical (s, S) Baseline]
    end

    subgraph Evaluation_Tier [Evaluation & Benchmarking]
        Agent & Baseline --> Eval[Statistical Evaluation Framework]
        Eval --> Metrics[Cost, Service Level, Stockout Rate, CVaR_0.05]
    end
```

---

## 👥 Team Structure & Responsibilities

The project is developed in collaborative, specialized functional groups:

### 🔹 Group A — Stochastic Programming & CVaR Optimization
* **Mihir Wani (M2)**
  * Mathematical formulation of the two-stage stochastic programming model.
  * Integration of Conditional Value-at-Risk (CVaR) risk measures for tail-risk mitigation.
  * Derivation of strategic inventory targets and safety stock boundaries.

### 🔹 Group B — Data Pipeline & System Engineering
* **Pankhudi Gupta (M6) & Pavan Sabnani (M1)**
  * Architecture of synthetic pharmaceutical data generation pipelines (Negative Binomial demand, Bernoulli disruptions, Log-normal lead times).
  * Data validation, statistical property testing, and exploratory visualization.
  * System interface design (standardized JSON contracts) and GitHub CI/CD automation.
  * Research paper co-authorship (Introduction & Literature Review).

### 🔹 Group C — RL Environment, PPO Agent & Evaluation
* **Devraj Srivastava (M3)** — *RL Environment Architecture*
  * Mathematical MDP formulation and design of `HealthcareSCEnv`.
  * Gymnasium-compliant step/reset simulation dynamics, state observations, and reward functions.
* **Prakhar Gupta (M4)** — *Deep Reinforcement Learning Agent*
  * Stable-Baselines3 PPO wrapper, neural network policy configuration, and validation.
  * Implementation of hyperparameter management and training pipelines.
* **Shourya Gupta (M5)** — *Integration & Evaluation Framework*
  * Implementation of traditional operations research baselines: $(s, S)$ inventory policy.
  * Statistical evaluation framework (Welch's t-test, Mann-Whitney U, Bootstrap 95% CIs, Cohen's d).
  * Integration bridge connecting Strategic SP outputs to operational RL constraints.

---

## 📂 Repository Structure

```text
pharma-supply-chain-optimization/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated testing and linting CI pipeline
├── data/                          # Raw, processed, and synthetic datasets
├── docs/                          # Architectural documentation, paper drafts, and notes
│   ├── paper/                     # Research paper sections (IEEE format)
│   └── week1/                     # Milestone design specifications
├── EPICS Work/                    # Group deliverables, documentation & test suites
│   ├── Implementation Plan.docx   # Master project roadmap and milestones
│   ├── WEEK 1 REPORT.docx         # Formal supervisor milestone report
│   └── pharmaceutical_sc_optimization/
│       ├── configs/               # Environment & agent YAML configurations
│       ├── src/
│       │   ├── environment/       # HealthcareSCEnv Gymnasium environment
│       │   ├── agents/            # PPO Agent wrappers (Stable-Baselines3)
│       │   ├── baselines/         # Classical (s, S) policy implementations
│       │   ├── evaluation/        # Metrics, statistics & reporting pipelines
│       │   ├── integration/       # SP-to-RL constraint translation bridge
│       │   └── utils/             # Reproducibility seeds and config loaders
│       └── tests/                 # Comprehensive pytest test suite (52 tests)
├── src/
│   ├── data_pipeline/             # Synthetic generation, validation & visualization
│   ├── stochastic_programming/    # Group A mathematical models (in progress)
│   ├── rl_environment/            # Core RL environment definitions
│   ├── ppo_agent/                 # PPO training modules
│   └── evaluation/                # Performance analytics
└── README.md                      # Primary project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10 - 3.13)
- Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/pankhudi24bai10877-star/pharma-supply-chain-optimization.git
cd pharma-supply-chain-optimization

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install core dependencies
pip install -r "EPICS Work/pharmaceutical_sc_optimization/requirements.txt"
```

### 3. Generate Synthetic Data
```bash
python src/data_pipeline/synthetic_data.py
python src/data_pipeline/validate_data.py
```

### 4. Run the Test Suite
To verify the complete test suite (52 tests across environment, PPO agent, baselines, and reproducibility):
```bash
pytest "EPICS Work/pharmaceutical_sc_optimization/tests" -v
```

---

## 📊 Current Milestone Progress

| Module / Milestone | Group | Status | Key Highlights |
| :--- | :--- | :---: | :--- |
| **Data Generation Pipeline** | Group B | ✅ Completed | Negative Binomial demand, disruption & lead-time generator with validation |
| **System Architecture & CI** | Group B | ✅ Completed | GitHub Actions CI workflow, standardized JSON schema |
| **MDP & Environment Design** | Group C | ✅ Completed | 15-dim state, 6-dim action space, Gymnasium API compliant (`HealthcareSCEnv`) |
| **Environment Dynamics** | Group C | ✅ Completed | Dynamic inventory transitions, non-trivial demand sampling, cost tracking |
| **PPO Agent Implementation** | Group C | ✅ Completed | Stable-Baselines3 wrapper; CartPole baseline validation (reward ≥ 300) |
| **(s, S) Baseline Policy** | Group C | ✅ Completed | Working OR policy executing full 365-day episodes with multi-supplier orders |
| **Evaluation Framework** | Group C | ✅ Completed | Statistical metrics (cost, fill rate, stockout rate, CVaR), seeds [42..1024] |
| **Unit & Integration Tests** | Group C | ✅ Completed | **52 passing tests (0 failures)** covering all core dynamics |
| **Stochastic Programming Model**| Group A | 🔄 In Progress | Two-stage formulation and scenario generation underway |

---

## 📖 Key Literature References

1. **PPO Algorithm:** Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
2. **GAE Formulation:** Schulman, J., et al. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016.
3. **Healthcare Supply Chain MDP:** Saha, E., & Ray, P. K. (2019). *Modelling and analysis of healthcare supply chain risk management.* IJPR.
4. **Multi-Agent RL in SCM:** Saha, E., & Rathore, P. (2022). *Multi-echelon pharmaceutical supply chain coordination under disruption.*
5. **RL in SCM Survey:** Rolf, B., et al. (2022). *A review on reinforcement learning algorithms and applications in supply chain management.* IJPE.
6. **CVaR Optimization:** Rockafellar, R. T., & Uryasev, S. (2000). *Optimization of conditional value-at-risk.* Journal of Risk.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

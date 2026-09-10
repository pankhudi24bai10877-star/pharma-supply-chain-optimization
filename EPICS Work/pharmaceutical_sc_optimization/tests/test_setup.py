"""Test that all dependencies are properly installed and importable.

This test file validates the Week 1 (M4) setup task:
- Gymnasium, Stable-Baselines3, PyTorch are installed and working.
- Project modules are importable.
"""

import pytest


class TestDependencyImports:
    """Verify all required packages are installed."""
    
    def test_gymnasium_import(self):
        import gymnasium as gym
        assert hasattr(gym, 'Env')
    
    def test_stable_baselines3_import(self):
        import stable_baselines3 as sb3
        assert hasattr(sb3, 'PPO')
    
    def test_pytorch_import(self):
        import torch
        assert hasattr(torch, 'tensor')
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    
    def test_tensorboard_import(self):
        from torch.utils.tensorboard import SummaryWriter
    
    def test_numpy_import(self):
        import numpy as np
        assert hasattr(np, 'array')
    
    def test_scipy_import(self):
        import scipy
        from scipy import stats
    
    def test_matplotlib_import(self):
        import matplotlib
        import matplotlib.pyplot as plt
    
    def test_yaml_import(self):
        import yaml


class TestProjectImports:
    """Verify project modules are importable."""
    
    def test_environment_import(self):
        from src.environment.healthcare_sc_env import HealthcareSCEnv
    
    def test_evaluation_import(self):
        from src.evaluation.metrics import EvaluationFramework, EvaluationConfig
    
    def test_baselines_import(self):
        from src.baselines.s_s_policy import SSPolicy
    
    def test_utils_import(self):
        from src.utils.common import set_global_seed, load_config

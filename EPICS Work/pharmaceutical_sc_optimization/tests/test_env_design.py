"""Test HealthcareSCEnv design — space definitions and Gymnasium API compliance.

Week 1 tests for M3's environment design.
These tests validate that the environment:
1. Has properly defined observation and action spaces
2. Follows Gymnasium API (reset returns obs+info, step returns 5-tuple)
3. Can be instantiated with default parameters
4. Spaces have correct shapes and types
"""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from src.environment.healthcare_sc_env import HealthcareSCEnv


class TestEnvDesign:
    """Test environment design and space definitions."""
    
    def test_env_instantiation(self):
        env = HealthcareSCEnv()
        assert env is not None
        assert env.num_products == 3
        assert env.num_suppliers == 2
        assert env.max_steps == 365
    
    def test_observation_space_defined(self):
        env = HealthcareSCEnv()
        assert isinstance(env.observation_space, gym.spaces.Box)
        expected_dim = (3 * 4) + 2 + 1  # 4 per product + suppliers + time = 15
        assert env.observation_space.shape == (expected_dim,)
        assert env.observation_space.dtype == np.float32
    
    def test_action_space_defined(self):
        env = HealthcareSCEnv()
        assert isinstance(env.action_space, gym.spaces.Box)
        expected_dim = 3 * 2  # products * suppliers = 6
        assert env.action_space.shape == (expected_dim,)
        assert env.action_space.dtype == np.float32
    
    def test_reset_returns_correct_format(self):
        env = HealthcareSCEnv()
        result = env.reset(seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert obs.shape == env.observation_space.shape
    
    def test_step_returns_correct_format(self):
        env = HealthcareSCEnv()
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_custom_parameters(self):
        env = HealthcareSCEnv(num_products=5, num_suppliers=3, max_steps=100)
        assert env.num_products == 5
        assert env.num_suppliers == 3
        assert env.max_steps == 100
        obs_dim = (5 * 4) + 3 + 1  # 24
        assert env.observation_space.shape == (obs_dim,)
        action_dim = 5 * 3  # 15
        assert env.action_space.shape == (action_dim,)
    
    def test_gymnasium_api_compliance(self):
        """Run Gymnasium's built-in env checker."""
        env = HealthcareSCEnv()
        # This checks reset/step signatures, space membership, etc.
        check_env(env.unwrapped, skip_render_check=True)
    
    def test_episode_termination(self):
        env = HealthcareSCEnv(max_steps=5)
        env.reset(seed=42)
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if i < 4:
                assert not terminated
            else:
                assert terminated
    
    def test_info_dict_keys(self):
        env = HealthcareSCEnv()
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        expected_keys = {'step', 'holding_cost', 'stockout_cost', 'ordering_cost', 'wastage_cost'}
        assert expected_keys.issubset(info.keys())
    
    def test_seed_reproducibility(self):
        env1 = HealthcareSCEnv()
        obs1, _ = env1.reset(seed=42)
        
        env2 = HealthcareSCEnv()
        obs2, _ = env2.reset(seed=42)
        
        np.testing.assert_array_equal(obs1, obs2)


class TestEnvDynamics:
    def test_reset_inventory_initialized(self):
        env = HealthcareSCEnv()
        obs, _ = env.reset()
        # Inventory is first P elements
        P = env.num_products
        inv_normalized = obs[0:P]
        # check > 0
        assert np.all(inv_normalized > 0)
        
    def test_step_demand_generated(self):
        env = HealthcareSCEnv()
        env.reset()
        action = np.zeros(env.action_space.shape)
        obs, _, _, _, _ = env.step(action)
        P = env.num_products
        demand_normalized = obs[2*P:3*P]
        # with high probability, demand > 0
        assert np.any(demand_normalized > 0)

    def test_step_inventory_changes(self):
        env = HealthcareSCEnv()
        env.reset(seed=42)
        # Force high demand
        env.demand_n = np.array([100, 100, 100])
        env.demand_p = np.array([0.1, 0.1, 0.1])
        action = np.zeros(env.action_space.shape)
        obs, _, _, _, _ = env.step(action)
        P = env.num_products
        inv_normalized = obs[0:P]
        # Inventory should decrease, possibly to 0
        assert np.all(inv_normalized < (env.initial_inventory / env._max_inventory) + 0.001)

    def test_step_reward_is_negative(self):
        env = HealthcareSCEnv()
        env.reset()
        action = np.zeros(env.action_space.shape)
        _, reward, _, _, _ = env.step(action)
        assert reward < 0

    def test_step_ordering_costs(self):
        env = HealthcareSCEnv()
        env.reset()
        action = np.ones(env.action_space.shape) # Max order
        _, _, _, _, info = env.step(action)
        assert info['ordering_cost'] > 0

    def test_step_stockout_penalty(self):
        env = HealthcareSCEnv()
        env.reset(seed=42)
        # Force high demand and no stock to get stockout
        env._inventory = np.zeros(env.num_products)
        env.demand_n = np.array([100, 100, 100])
        env.demand_p = np.array([0.1, 0.1, 0.1])
        action = np.zeros(env.action_space.shape)
        _, _, _, _, info = env.step(action)
        assert info['stockout_cost'] > 0

    def test_observation_normalized(self):
        env = HealthcareSCEnv()
        env.reset()
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_full_episode_runs(self):
        env = HealthcareSCEnv(max_steps=365)
        env.reset(seed=42)
        terminated = False
        steps = 0
        while not terminated:
            action = env.action_space.sample()
            _, _, terminated, _, _ = env.step(action)
            steps += 1
        assert steps == 365

    def test_different_seeds_different_demand(self):
        env1 = HealthcareSCEnv()
        env1.reset(seed=42)
        action = np.zeros(env1.action_space.shape)
        obs1, _, _, _, _ = env1.step(action)
        
        env2 = HealthcareSCEnv()
        env2.reset(seed=99)
        obs2, _, _, _, _ = env2.step(action)
        
        P = env1.num_products
        demand1 = obs1[2*P:3*P]
        demand2 = obs2[2*P:3*P]
        
        assert not np.array_equal(demand1, demand2)

    def test_cost_breakdown_in_info(self):
        env = HealthcareSCEnv()
        env.reset()
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        assert 'holding_cost' in info
        assert 'stockout_cost' in info
        assert 'ordering_cost' in info
        assert 'wastage_cost' in info

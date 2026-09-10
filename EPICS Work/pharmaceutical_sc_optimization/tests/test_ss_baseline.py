"""Test (s,S) inventory policy baseline.

Week 2 tests for M5's baseline implementation.
Validates:
1. Policy can be instantiated with default and custom parameters
2. Policy correctly applies (s,S) ordering logic
3. Policy produces valid actions for HealthcareSCEnv
4. Policy can run a full episode in the environment
"""

import pytest
import numpy as np
from src.baselines.s_s_policy import SSPolicy
from src.environment.healthcare_sc_env import HealthcareSCEnv


class TestSSPolicyLogic:
    """Test (s,S) policy ordering logic."""
    
    def test_instantiation_default(self):
        policy = SSPolicy()
        assert policy.num_products == 3
        assert policy.num_suppliers == 2
        np.testing.assert_array_equal(policy.s, [50, 50, 50])
        np.testing.assert_array_equal(policy.S, [200, 200, 200])
    
    def test_instantiation_custom(self):
        s = np.array([30, 40, 20])
        S = np.array([150, 180, 100])
        policy = SSPolicy(reorder_points=s, order_up_to_levels=S)
        np.testing.assert_array_equal(policy.s, s)
        np.testing.assert_array_equal(policy.S, S)
    
    def test_invalid_params_raises(self):
        """s must be less than S."""
        with pytest.raises(AssertionError):
            SSPolicy(reorder_points=np.array([200, 200, 200]),
                     order_up_to_levels=np.array([100, 100, 100]))
    
    def test_order_when_below_reorder_point(self):
        """When inventory < s, should order up to S."""
        policy = SSPolicy(
            reorder_points=np.array([50.0, 50.0, 50.0]),
            order_up_to_levels=np.array([200.0, 200.0, 200.0]),
            max_inventory=500.0,
            max_order_qty=500.0,
        )
        # Create obs with inventory = [30, 60, 40] (normalized by 500)
        obs = np.zeros(15, dtype=np.float32)
        obs[0] = 30.0 / 500.0   # product 0: inv=30 < s=50 -> order
        obs[1] = 60.0 / 500.0   # product 1: inv=60 > s=50 -> no order
        obs[2] = 40.0 / 500.0   # product 2: inv=40 < s=50 -> order
        
        action = policy.predict(obs)
        assert action.shape == (6,)
        
        # Product 0: order = (200-30)/2 = 85 per supplier, normalized = 85/500 = 0.17
        assert action[0] > 0  # p0_s0
        assert action[1] > 0  # p0_s1
        # Product 1: no order
        assert action[2] == 0  # p1_s0
        assert action[3] == 0  # p1_s1
        # Product 2: order = (200-40)/2 = 80 per supplier
        assert action[4] > 0  # p2_s0
        assert action[5] > 0  # p2_s1
    
    def test_no_order_when_above_reorder_point(self):
        """When inventory > s for all products, no orders."""
        policy = SSPolicy()
        obs = np.zeros(15, dtype=np.float32)
        obs[0:3] = 200.0 / 500.0  # All products at 200, above s=50
        
        action = policy.predict(obs)
        np.testing.assert_array_almost_equal(action, np.zeros(6))
    
    def test_action_in_valid_range(self):
        """Actions should be in [0, 1]."""
        policy = SSPolicy()
        obs = np.zeros(15, dtype=np.float32)  # Very low inventory
        action = policy.predict(obs)
        assert np.all(action >= 0.0)
        assert np.all(action <= 1.0)
    
    def test_get_params(self):
        policy = SSPolicy()
        params = policy.get_params()
        assert 's' in params
        assert 'S' in params


class TestSSPolicyWithEnv:
    """Test (s,S) policy interaction with HealthcareSCEnv."""
    
    def test_policy_action_shape_matches_env(self):
        env = HealthcareSCEnv()
        policy = SSPolicy(
            num_products=env.num_products,
            num_suppliers=env.num_suppliers,
        )
        obs, _ = env.reset(seed=42)
        action = policy.predict(obs)
        assert action.shape == env.action_space.shape
    
    def test_policy_runs_one_step(self):
        env = HealthcareSCEnv()
        policy = SSPolicy(num_products=env.num_products, num_suppliers=env.num_suppliers)
        obs, _ = env.reset(seed=42)
        action = policy.predict(obs)
        obs2, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float))
    
    def test_policy_runs_full_episode(self):
        """Run complete episode with (s,S) policy."""
        env = HealthcareSCEnv(max_steps=100)
        policy = SSPolicy(num_products=env.num_products, num_suppliers=env.num_suppliers)
        
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        steps = 0
        
        while True:
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        
        assert steps == 100
        assert total_reward < 0  # Costs should make reward negative
        env.close()
    
    def test_policy_with_custom_env(self):
        env = HealthcareSCEnv(num_products=5, num_suppliers=3, max_steps=50)
        policy = SSPolicy(num_products=5, num_suppliers=3,
                         reorder_points=np.ones(5) * 40,
                         order_up_to_levels=np.ones(5) * 150)
        obs, _ = env.reset(seed=123)
        for _ in range(50):
            action = policy.predict(obs)
            assert action.shape == (15,)  # 5*3
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break

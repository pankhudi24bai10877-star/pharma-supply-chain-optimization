"""Test PPO agent wrapper and SB3 integration.

Week 2 tests for M4's PPO agent.
Validates:
1. PPO agent can be instantiated with a Gymnasium environment
2. PPO agent can train (briefly) and predict actions
3. CartPole validation passes
4. PPO works with HealthcareSCEnv
"""

import pytest
import numpy as np
import gymnasium as gym
from src.agents.ppo_agent import PPOAgent
from src.environment.healthcare_sc_env import HealthcareSCEnv


class TestPPOAgent:
    """Test PPO agent wrapper."""
    
    def test_agent_instantiation_without_env(self):
        """Agent can be created without an environment."""
        agent = PPOAgent()
        assert agent.model is None
        assert agent.env is None
    
    def test_agent_instantiation_with_env(self):
        """Agent creates SB3 model when given an environment."""
        env = gym.make('CartPole-v1')
        agent = PPOAgent(env=env)
        assert agent.model is not None
        env.close()
    
    def test_agent_with_config(self):
        """Agent respects configuration."""
        env = gym.make('CartPole-v1')
        config = {'learning_rate': 1e-4, 'gamma': 0.95, 'n_steps': 512}
        agent = PPOAgent(env=env, config=config)
        assert agent.learning_rate == 1e-4
        assert agent.gamma == 0.95
        assert agent.n_steps == 512
        env.close()
    
    def test_train_briefly(self):
        """Agent can train for a few steps without errors."""
        env = gym.make('CartPole-v1')
        agent = PPOAgent(env=env, config={'n_steps': 128, 'batch_size': 32})
        result = agent.train(total_timesteps=256)
        assert 'total_timesteps' in result
        env.close()
    
    def test_predict_after_train(self):
        """Agent can predict after training."""
        env = gym.make('CartPole-v1')
        agent = PPOAgent(env=env, config={'n_steps': 128, 'batch_size': 32})
        agent.train(total_timesteps=256)
        obs, _ = env.reset()
        action, states = agent.predict(obs)
        assert action is not None
        env.close()
    
    def test_predict_without_model_raises(self):
        """Predicting without a model raises RuntimeError."""
        agent = PPOAgent()
        with pytest.raises(RuntimeError):
            agent.predict(np.zeros(4))
    
    def test_agent_with_healthcare_env(self):
        """PPO agent can be created and briefly trained with HealthcareSCEnv."""
        env = HealthcareSCEnv(max_steps=50)  # Short episodes for speed
        agent = PPOAgent(env=env, config={'n_steps': 128, 'batch_size': 64})
        agent.train(total_timesteps=256)
        obs, _ = env.reset()
        action, _ = agent.predict(obs)
        assert action.shape == env.action_space.shape
        env.close()
    
    def test_evaluate_agent(self):
        """Agent evaluation returns mean and std reward."""
        env = gym.make('CartPole-v1')
        agent = PPOAgent(env=env, config={'n_steps': 128, 'batch_size': 32})
        agent.train(total_timesteps=256)
        mean_reward, std_reward = agent.evaluate(n_eval_episodes=5)
        assert isinstance(mean_reward, float)
        assert isinstance(std_reward, float)
        env.close()


class TestCartPoleValidation:
    """Validate PPO works correctly on CartPole (M4 Week 2 task)."""
    
    @pytest.mark.slow
    def test_cartpole_validation(self):
        """Full CartPole validation — may take ~30s."""
        result = PPOAgent.validate_on_cartpole(total_timesteps=25_000)
        assert result['passed'], f"CartPole validation failed: mean_reward={result['mean_reward']}"
        print(f"CartPole mean reward: {result['mean_reward']:.1f} ± {result['std_reward']:.1f}")

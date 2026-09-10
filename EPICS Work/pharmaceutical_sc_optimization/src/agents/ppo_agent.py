"""PPO Agent wrapper for pharmaceutical supply chain optimization.

Owner: M4 (Prakhar Gupta)
Deliverable: C3 — PPO agent implementation using Stable-Baselines3
Week 2: Basic PPO wrapper with SB3. Validated on CartPole.

References:
- Paper 4 (Schulman et al. — PPO)
- Stable-Baselines3 documentation
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from typing import Optional, Dict, Any, Tuple
import os


class PPOAgent:
    """Wrapper around SB3's PPO for pharmaceutical supply chain.
    
    Week 2 Implementation:
    - Basic SB3 PPO with configurable hyperparameters
    - Train/predict/save/load interface
    - Works with any Gymnasium environment
    - No CVaR reward (Week 10)
    - No hyperparameter tuning pipeline (Week 12)
    """
    
    def __init__(
        self,
        env: Optional[gym.Env] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or {}
        self.env = env
        self.model = None
        
        # PPO hyperparameters (defaults, tuning in Week 12)
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        self.n_steps = self.config.get('n_steps', 2048)
        self.batch_size = self.config.get('batch_size', 64)
        self.n_epochs = self.config.get('n_epochs', 10)
        self.gamma = self.config.get('gamma', 0.99)
        self.gae_lambda = self.config.get('gae_lambda', 0.95)
        self.clip_range = self.config.get('clip_range', 0.2)
        self.ent_coef = self.config.get('ent_coef', 0.01)
        self.vf_coef = self.config.get('vf_coef', 0.5)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
        self.policy_kwargs = self.config.get('policy_kwargs', dict(net_arch=[64, 64]))
        self.tensorboard_log = self.config.get('tensorboard_log', None)
        self.verbose = self.config.get('verbose', 0)
        
        if self.env is not None:
            self._build_model()
    
    def _build_model(self):
        """Build the SB3 PPO model."""
        self.model = PPO(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            policy_kwargs=self.policy_kwargs,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
        )
    
    def train(self, total_timesteps: int = 100_000) -> Dict[str, Any]:
        """Train the PPO agent.
        
        Args:
            total_timesteps: Total number of environment steps for training.
            
        Returns:
            Training info dict.
        """
        if self.model is None:
            raise RuntimeError("Model not built. Provide an environment.")
        self.model.learn(total_timesteps=total_timesteps)
        return {'total_timesteps': total_timesteps}
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Any]:
        """Predict action given observation.
        
        Args:
            observation: Current environment observation.
            deterministic: Whether to use deterministic actions.
            
        Returns:
            Tuple of (action, states).
        """
        if self.model is None:
            raise RuntimeError("Model not built or loaded.")
        return self.model.predict(observation, deterministic=deterministic)
    
    def evaluate(self, env: Optional[gym.Env] = None, n_eval_episodes: int = 10) -> Tuple[float, float]:
        """Evaluate the trained agent.
        
        Args:
            env: Environment to evaluate on (uses training env if None).
            n_eval_episodes: Number of evaluation episodes.
            
        Returns:
            Tuple of (mean_reward, std_reward).
        """
        if self.model is None:
            raise RuntimeError("Model not built or loaded.")
        eval_env = env if env is not None else self.env
        return evaluate_policy(self.model, eval_env, n_eval_episodes=n_eval_episodes)
    
    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(path)
    
    def load(self, path: str, env: Optional[gym.Env] = None) -> 'PPOAgent':
        """Load trained model from disk."""
        self.model = PPO.load(path, env=env)
        if env is not None:
            self.env = env
        return self
    
    @staticmethod
    def validate_on_cartpole(total_timesteps: int = 25_000, verbose: int = 0) -> Dict[str, Any]:
        """Validate PPO setup by training on CartPole-v1.
        
        This is a Week 2 validation task to confirm SB3+PyTorch+Gymnasium
        work correctly together.
        
        Args:
            total_timesteps: Training steps for validation.
            verbose: Verbosity level.
            
        Returns:
            Dict with mean_reward, std_reward, and pass/fail status.
        """
        env = gym.make('CartPole-v1')
        model = PPO('MlpPolicy', env, verbose=verbose)
        model.learn(total_timesteps=total_timesteps)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
        env.close()
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'target_reward': 475.0,
            'passed': mean_reward >= 300.0,  # Relaxed threshold for quick validation
            'total_timesteps': total_timesteps,
        }

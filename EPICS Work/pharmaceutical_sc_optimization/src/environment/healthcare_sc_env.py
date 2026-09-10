"""HealthcareSCEnv — Gymnasium-compatible RL environment for pharmaceutical supply chain.

Owner: M3 (Devraj Srivastava)
Deliverable: C1 — HealthcareSCEnv
Target: Nov (W8) for full implementation; Week 1 is design only.

This module will implement a custom Gymnasium environment modeling a multi-product,
multi-supplier pharmaceutical supply chain with:
- Stochastic demand (Negative Binomial)
- Supply disruptions (Bernoulli)
- Perishable inventory
- Variable lead times (Log-normal)

References:
- Paper 7 (Saha & Ray — MDP formulation)
- Paper 1 (Saha & Rathore — MARL environment)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple


class HealthcareSCEnv(gym.Env):
    """Pharmaceutical Supply Chain Environment.
    
    MDP Design (Week 1 — M3):
    --------------------------
    State Space:
        - Current inventory levels per product (continuous)
        - Pipeline inventory (orders in transit) per product (continuous)
        - Current demand observation per product (continuous)
        - Supplier availability status per supplier (binary)
        - Time step / day of planning horizon (discrete)
        - Remaining shelf life per product batch (discrete)
        
    Action Space:
        - Order quantities per product from each supplier (continuous)
        - Bounded by [0, max_order_quantity]
        
    Reward:
        - Negative total cost = -(holding_cost + stockout_penalty + ordering_cost + wastage_cost)
        - Will be augmented with CVaR risk measure in later weeks (C4)
        
    Transitions:
        - Demand sampled from Negative Binomial distribution
        - Supply disruptions sampled from Bernoulli distribution
        - Lead times sampled from Log-normal distribution
        - Inventory updated: new_inv = old_inv + received_orders - demand - expired
        
    Episode:
        - Fixed planning horizon (e.g., 365 days)
        - Terminates at end of horizon
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(
        self,
        num_products: int = 3,
        num_suppliers: int = 2,
        max_steps: int = 365,
        render_mode: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the HealthcareSCEnv."""
        super().__init__()
        
        self.num_products = num_products
        self.num_suppliers = num_suppliers
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.config = config or {}
        
        obs_dim = (num_products * 4) + num_suppliers + 1
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        action_dim = num_products * num_suppliers
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )
        
        # Cost parameters
        self.c_hold = self.config.get('c_hold', 0.5)
        self.c_stock = self.config.get('c_stock', 10.0)
        self.c_order = self.config.get('c_order', 1.0)
        self.c_waste = self.config.get('c_waste', 5.0)
        self.max_order_qty = self.config.get('max_order_qty', 500)
        self.initial_inventory = self.config.get('initial_inventory', 100)
        
        # Demand distribution parameters (must match num_products)
        # Defaults: NegBin(n, p) with mean = n*(1-p)/p ~ 10 units/day
        default_n = np.array([5, 8, 3])
        default_p = np.array([0.3, 0.4, 0.25])
        # Scale to num_products by cycling the default patterns
        self.demand_n = self.config.get(
            'demand_n',
            np.array([default_n[i % len(default_n)] for i in range(num_products)])
        )
        self.demand_p = self.config.get(
            'demand_p',
            np.array([default_p[i % len(default_p)] for i in range(num_products)])
        )
        
        # Normalization constants
        self._max_inventory = 500.0
        self._max_demand = 100.0
        
        self._current_step = 0
        
        # Internal state
        self._inventory = None
        self._pipeline = None
        self._last_demand = None
        self._supplier_status = None
    
    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        P = self.num_products
        K = self.num_suppliers
        
        # Inventory levels (normalized)
        obs[0:P] = np.clip(self._inventory / self._max_inventory, 0, 1)
        # Pipeline inventory (normalized) 
        obs[P:2*P] = np.clip(self._pipeline / self._max_inventory, 0, 1)
        # Last demand (normalized)
        obs[2*P:3*P] = np.clip(self._last_demand / self._max_demand, 0, 1)
        # shelf_life (normalized, 1.0 placeholder for now)
        obs[3*P:4*P] = 1.0  # shelf life placeholder
        # supplier status
        obs[4*P:4*P+K] = self._supplier_status
        # time step normalized
        obs[4*P+K] = self._current_step / self.max_steps
        
        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self._current_step = 0
        
        self._inventory = np.full(self.num_products, self.initial_inventory, dtype=np.float32)
        self._pipeline = np.zeros(self.num_products, dtype=np.float32)
        self._last_demand = np.zeros(self.num_products, dtype=np.float32)
        self._supplier_status = np.ones(self.num_suppliers, dtype=np.float32)
        
        obs = self._build_observation()
        info = {"step": self._current_step}
        
        return obs, info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step in the environment."""
        self._current_step += 1
        
        # Scale action to order quantities
        scaled_action = action * self.max_order_qty
        
        # Aggregate orders per product
        orders_per_product_supplier = scaled_action.reshape((self.num_products, self.num_suppliers))
        received_orders = orders_per_product_supplier.sum(axis=1)
        total_ordered = received_orders.sum()
        
        # Sample demand
        demand = self.np_random.negative_binomial(self.demand_n, self.demand_p)
        self._last_demand = demand.astype(np.float32)
        
        # Update inventory
        old_inventory = self._inventory.copy()
        new_inv = old_inventory + received_orders - demand
        self._inventory = np.maximum(0, new_inv)
        
        # Compute costs
        holding_cost = self.c_hold * self._inventory.sum()
        stockout_qty = np.maximum(0, demand - (old_inventory + received_orders))
        stockout_cost = self.c_stock * stockout_qty.sum()
        ordering_cost = self.c_order * total_ordered
        wastage_cost = 0.0
        
        total_cost = holding_cost + stockout_cost + ordering_cost + wastage_cost
        reward = -float(total_cost)
        
        terminated = self._current_step >= self.max_steps
        truncated = False
        
        obs = self._build_observation()
        info = {
            "step": self._current_step,
            "holding_cost": float(holding_cost),
            "stockout_cost": float(stockout_cost),
            "ordering_cost": float(ordering_cost),
            "wastage_cost": float(wastage_cost),
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment state."""
        if self.render_mode == "ansi":
            return f"Step: {self._current_step}/{self.max_steps}"
        return None
    
    def close(self):
        """Clean up resources."""
        pass

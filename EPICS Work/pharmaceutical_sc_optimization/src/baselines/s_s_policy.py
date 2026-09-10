"""(s, S) Inventory Policy Baseline.

Owner: M5 (Shourya Gupta)
Deliverable: C6 — Baseline implementations
Week 2: Implement working (s,S) policy that interacts with HealthcareSCEnv.

The (s, S) policy is a classic operations research inventory control policy:
- s = reorder point: when inventory drops to or below s, trigger an order
- S = order-up-to level: order enough to bring inventory up to S
- Order quantity = max(0, S - current_inventory) when inventory <= s

This serves as a traditional baseline to compare against RL-based approaches.
"""

import numpy as np
from typing import Optional, Dict, Any


class SSPolicy:
    """Classic (s, S) inventory policy baseline.
    
    For each product p:
        if inventory[p] <= s[p]:
            order_qty[p] = S[p] - inventory[p]
        else:
            order_qty[p] = 0
    
    Orders are distributed equally across all available suppliers.
    Action is normalized to [0, 1] for compatibility with HealthcareSCEnv.
    """
    
    def __init__(
        self,
        reorder_points: Optional[np.ndarray] = None,
        order_up_to_levels: Optional[np.ndarray] = None,
        num_products: int = 3,
        num_suppliers: int = 2,
        max_order_qty: float = 500.0,
        max_inventory: float = 500.0,
    ):
        """Initialize (s, S) policy.
        
        Args:
            reorder_points: Reorder point s per product. Default: [50, 50, 50]
            order_up_to_levels: Order-up-to level S per product. Default: [200, 200, 200]
            num_products: Number of pharmaceutical products.
            num_suppliers: Number of suppliers.
            max_order_qty: Maximum order quantity (for action normalization).
            max_inventory: Maximum inventory level (for observation denormalization).
        """
        self.num_products = num_products
        self.num_suppliers = num_suppliers
        self.max_order_qty = max_order_qty
        self.max_inventory = max_inventory
        
        # Default (s, S) parameters — can be tuned later
        self.s = reorder_points if reorder_points is not None else np.ones(num_products) * 50.0
        self.S = order_up_to_levels if order_up_to_levels is not None else np.ones(num_products) * 200.0
        
        assert len(self.s) == num_products
        assert len(self.S) == num_products
        assert np.all(self.s < self.S), "Reorder points must be less than order-up-to levels"
    
    def _extract_inventory(self, observation: np.ndarray) -> np.ndarray:
        """Extract inventory levels from normalized observation.
        
        The observation vector layout (from HealthcareSCEnv):
        [inventory(P), pipeline(P), demand(P), shelf_life(P), supplier_status(K), time(1)]
        
        Inventory levels are the first P elements, normalized by max_inventory.
        """
        normalized_inv = observation[0:self.num_products]
        return normalized_inv * self.max_inventory  # De-normalize
    
    def predict(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """Predict order quantities using (s, S) policy.
        
        Args:
            observation: Current environment observation (normalized).
            
        Returns:
            action: Normalized order quantities in [0, 1], shape (P * K,).
        """
        inventory = self._extract_inventory(observation)
        
        # Compute order quantities per product
        order_qty = np.zeros(self.num_products, dtype=np.float64)
        for p in range(self.num_products):
            if inventory[p] <= self.s[p]:
                order_qty[p] = max(0.0, self.S[p] - inventory[p])
        
        # Distribute orders equally across suppliers
        # Action shape: (num_products * num_suppliers,)
        # Layout: [p0_s0, p0_s1, p1_s0, p1_s1, p2_s0, p2_s1]
        action = np.zeros(self.num_products * self.num_suppliers, dtype=np.float32)
        for p in range(self.num_products):
            per_supplier = order_qty[p] / self.num_suppliers
            for k in range(self.num_suppliers):
                idx = p * self.num_suppliers + k
                # Normalize to [0, 1] by dividing by max_order_qty
                action[idx] = np.clip(per_supplier / self.max_order_qty, 0.0, 1.0)
        
        return action
    
    def get_params(self) -> Dict[str, Any]:
        """Return policy parameters for logging."""
        return {
            's': self.s.tolist(),
            'S': self.S.tolist(),
            'num_products': self.num_products,
            'num_suppliers': self.num_suppliers,
        }

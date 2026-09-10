"""Common utilities for the pharmaceutical supply chain optimization project."""

import numpy as np
import random
import yaml
from typing import Dict, Any

def set_global_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

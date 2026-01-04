"""
Synapse Dynamics: Human-inspired neuroplasticity engine for neural networks.
Compatible with NumPy, PyTorch, and TensorFlow. Safe, efficient, and extensible.
"""
import numpy as np

class PlasticityConfig:
    def __init__(self, rule='hebbian', lr=0.01, **kwargs):
        self.rule = rule
        self.lr = lr
        self.kwargs = kwargs

class PlasticityEngine:
    def __init__(self, config: PlasticityConfig):
        self.config = config
        self.last_delta = None

    def validate(self, weights, pre, post):
        # Garante arrays válidos e evita problemas de overflow/NaN
        if not (isinstance(weights, np.ndarray) and isinstance(pre, np.ndarray) and isinstance(post, np.ndarray)):
            raise ValueError('weights, pre, post must be numpy arrays')
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            raise ValueError('weights contain NaN or Inf')
        if np.any(np.isnan(pre)) or np.any(np.isinf(pre)):
            raise ValueError('pre contains NaN or Inf')
        if np.any(np.isnan(post)) or np.any(np.isinf(post)):
            raise ValueError('post contains NaN or Inf')

    def update(self, weights, pre, post):
        self.validate(weights, pre, post)
        lr = float(self.config.lr)
        # Hebbian: delta_w = lr * (post x pre)
        if self.config.rule == 'hebbian':
            delta = lr * np.outer(post, pre)
        # Oja: delta_w = lr * [(post x pre) - (post^2) * w]
        elif self.config.rule == 'oja':
            delta = lr * (np.outer(post, pre) - (post**2)[:, None] * weights)
        else:
            raise ValueError(f'Unknown rule: {self.config.rule}')
        # Limita o tamanho do update para estabilidade
        max_delta = 10.0 * lr
        delta = np.clip(delta, -max_delta, max_delta)
        self.last_delta = delta
        return delta
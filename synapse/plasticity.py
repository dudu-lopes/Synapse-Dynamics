"""
Synapse Dynamics: Human-inspired neuroplasticity engine for neural networks.
Minimal, portátil e seguro. Apenas núcleo essencial.
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
        lr = self.config.lr
        if self.config.rule == 'hebbian':
            delta = lr * np.outer(post, pre)
        elif self.config.rule == 'oja':
            delta = lr * (np.outer(post, pre) - (post**2)[:, None] * weights)
        else:
            raise ValueError(f'Unknown rule: {self.config.rule}')
        self.last_delta = delta
        return delta
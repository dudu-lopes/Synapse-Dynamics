import numpy as np
from synapse import Injector, PlasticityConfig


class NumpyNet:
    def __init__(self):
        self.w1 = np.random.randn(10, 20) * 0.1
        self.w2 = np.random.randn(20, 5) * 0.1

    def forward(self, x):
        h = np.maximum(0, x @ self.w1)
        return h @ self.w2


def test_injector_shape_consistency():
    w = np.zeros((10, 20), dtype=np.float32)
    inj = Injector(w, PlasticityConfig(rule='hebbian', lr=0.1))
    pre = np.ones(20, dtype=np.float32)
    post = np.ones(10, dtype=np.float32)
    updated = inj.apply_updates(pre, post)
    assert updated.shape == (10, 20)
    assert np.all(updated >= 0)

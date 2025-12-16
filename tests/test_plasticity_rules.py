import numpy as np

from synapse.plasticity import PlasticityConfig
from synapse.plasticity_rules import HebbianPlasticity, HomeostasisPlasticity


def test_homeostasis_scales_weights_toward_target():
    cfg = PlasticityConfig()
    cfg.base_learning_rate = 0.02
    cfg.min_weight = 0.0
    cfg.max_weight = 2.0
    cfg.homeostatic_target = 0.5

    homeo = HomeostasisPlasticity(cfg)

    pre = np.random.rand(4)
    post = np.random.rand(3) * 2.0  # high activity
    current = np.ones((4, 3), dtype=np.float32)

    new_w = homeo.update_weights(pre, post, current)
    # New weights should remain within min/max
    assert new_w.shape == current.shape
    assert np.all(new_w >= cfg.min_weight - 1e-6)
    assert np.all(new_w <= cfg.max_weight + 1e-6)

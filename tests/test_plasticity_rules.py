import numpy as np
from synapse import PlasticityEngine, PlasticityConfig


def test_homeostasis_like_update():
    w = np.ones((3, 3), dtype=np.float32)
    pre = np.ones(3, dtype=np.float32)
    post = np.ones(3, dtype=np.float32)
    engine = PlasticityEngine(PlasticityConfig(rule='hebbian', lr=0.01))
    delta = engine.update(w, pre, post)
    assert delta.shape == (3, 3)
    assert np.all(delta >= 0)

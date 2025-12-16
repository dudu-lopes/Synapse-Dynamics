import numpy as np
import pytest
from synapse import Injector, PlasticityEngine, PlasticityConfig

def test_imports():
    Injector
    PlasticityEngine
    PlasticityConfig

def test_hebbian_update():
    w = np.zeros((2,2), dtype=np.float32)
    inj = Injector(w, PlasticityConfig(rule='hebbian', lr=0.5))
    pre = np.array([1, 0], dtype=np.float32)
    post = np.array([0, 1], dtype=np.float32)
    updated = inj.apply_updates(pre, post)
    expected = np.array([[0,0],[0.5,0]], dtype=np.float32)
    np.testing.assert_allclose(updated, expected)

def test_safe_mode_rollback():
    w = np.zeros((1,1), dtype=np.float32)
    inj = Injector(w, PlasticityConfig(rule='hebbian', lr=1e10), safe_mode=True)
    pre = np.array([np.nan], dtype=np.float32)
    post = np.array([1], dtype=np.float32)
    with pytest.raises(ValueError):
        inj.apply_updates(pre, post)
    # Should rollback to last good
    np.testing.assert_allclose(inj.weights, w)

import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None
try:
    import tensorflow as tf
except ImportError:
    tf = None

from synapse import Injector, PlasticityConfig

def test_numpy_integration():
    w = np.zeros((2,2), dtype=np.float32)
    inj = Injector(w, PlasticityConfig(rule='hebbian', lr=0.5))
    pre = np.array([1, 0], dtype=np.float32)
    post = np.array([0, 1], dtype=np.float32)
    updated = inj.apply_updates(pre, post)
    assert isinstance(updated, np.ndarray)
    np.testing.assert_allclose(updated, np.array([[0,0],[0.5,0]], dtype=np.float32))

@pytest.mark.skipif(torch is None, reason='torch not installed')
def test_torch_integration():
    w = torch.zeros((2,2), dtype=torch.float32)
    inj = Injector(w, PlasticityConfig(rule='hebbian', lr=0.5))
    pre = torch.tensor([1, 0], dtype=torch.float32)
    post = torch.tensor([0, 1], dtype=torch.float32)
    updated = inj.apply_updates(pre, post)
    assert isinstance(updated, torch.Tensor)
    np.testing.assert_allclose(updated.cpu().numpy(), np.array([[0,0],[0.5,0]], dtype=np.float32))

@pytest.mark.skipif(tf is None, reason='tensorflow not installed')
def test_tf_integration():
    w = tf.zeros((2,2), dtype=tf.float32)
    inj = Injector(w, PlasticityConfig(rule='hebbian', lr=0.5))
    pre = tf.constant([1, 0], dtype=tf.float32)
    post = tf.constant([0, 1], dtype=tf.float32)
    updated = inj.apply_updates(pre, post)
    assert isinstance(updated, tf.Tensor)
    np.testing.assert_allclose(updated.numpy(), np.array([[0,0],[0.5,0]], dtype=np.float32))

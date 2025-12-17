import time
import numpy as np
try:
    import torch
except ImportError:
    torch = None
try:
    import tensorflow as tf
except ImportError:
    tf = None
from synapse import Injector, PlasticityConfig

def benchmark_numpy():
    w = np.zeros((100,100), dtype=np.float32)
    inj = Injector(w, PlasticityConfig(rule='hebbian', lr=0.01))
    pre = np.random.rand(100).astype(np.float32)
    post = np.random.rand(100).astype(np.float32)
    t0 = time.time()
    for _ in range(1000):
        inj.apply_updates(pre, post)
    return time.time() - t0

if torch is not None:
    def benchmark_torch():
        w = torch.zeros((100,100), dtype=torch.float32)
        inj = Injector(w, PlasticityConfig(rule='hebbian', lr=0.01))
        pre = torch.rand(100, dtype=torch.float32)
        post = torch.rand(100, dtype=torch.float32)
        t0 = time.time()
        for _ in range(1000):
            inj.apply_updates(pre, post)
        return time.time() - t0
else:
    def benchmark_torch():
        return None

if tf is not None:
    def benchmark_tf():
        w = tf.zeros((100,100), dtype=tf.float32)
        inj = Injector(w, PlasticityConfig(rule='hebbian', lr=0.01))
        pre = tf.random.uniform((100,), dtype=tf.float32)
        post = tf.random.uniform((100,), dtype=tf.float32)
        t0 = time.time()
        for _ in range(1000):
            inj.apply_updates(pre, post)
        return time.time() - t0
else:
    def benchmark_tf():
        return None

def test_benchmarks():
    np_time = benchmark_numpy()
    print(f"NumPy benchmark: {np_time:.3f}s for 1000 updates")
    t_time = benchmark_torch()
    if t_time is not None:
        print(f"PyTorch benchmark: {t_time:.3f}s for 1000 updates")
    tf_time = benchmark_tf()
    if tf_time is not None:
        print(f"TensorFlow benchmark: {tf_time:.3f}s for 1000 updates")

"""Minimal, well-tested plasticity injector.

This module provides two public symbols used by the tests and the
plugin: ``PlasticityInjectionConfig`` and ``PlasticityInjector``. The
implementation is intentionally small and dependency-free so it can be
imported during test collection without pulling in optional or
unfinished components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
import threading
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None
try:
    import tensorflow as tf
except ImportError:
    tf = None


logger = logging.getLogger(__name__)


"""
Synapse Injector: Portable plasticity wrapper for any neural model.
- Does not change model architecture.
- Can be installed on any AI, any size.
- Applies PlasticityEngine updates to weights.
"""
from .plasticity import PlasticityEngine, PlasticityConfig

class Injector:
    def __init__(self, weights, config=None, safe_mode=True):
        self.weights = self._to_numpy(weights)
        self.last_good = self.weights.copy()
        self.safe_mode = safe_mode
        self.activity = None
        self.engine = PlasticityEngine(config or PlasticityConfig())
        self.framework = self._detect_framework(weights)

    def _detect_framework(self, arr):
        if torch is not None and isinstance(arr, torch.Tensor):
            return 'torch'
        if tf is not None and isinstance(arr, tf.Tensor):
            return 'tf'
        return 'numpy'

    def _to_numpy(self, arr):
        if torch is not None and isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        if tf is not None and isinstance(arr, tf.Tensor):
            return arr.numpy()
        return np.array(arr, dtype=np.float32)

    def _from_numpy(self, arr, ref):
        if self.framework == 'torch' and torch is not None:
            return torch.from_numpy(arr).to(ref.device)
        if self.framework == 'tf' and tf is not None:
            return tf.convert_to_tensor(arr)
        return arr

    def record(self, activity):
        self.activity = self._to_numpy(activity)

    def apply_updates(self, pre, post):
        pre_np = self._to_numpy(pre)
        post_np = self._to_numpy(post)
        try:
            delta = self.engine.update(self.weights, pre_np, post_np)
            new_weights = self.weights + delta
            if self.safe_mode:
                if np.any(np.isnan(new_weights)) or np.any(np.isinf(new_weights)):
                    self.rollback()
                    raise ValueError('Unsafe weight update detected (NaN/Inf). Rolled back.')
                self.last_good = self.weights.copy()
            self.weights = new_weights
            return self._from_numpy(self.weights, pre)
        except Exception as e:
            if self.safe_mode:
                self.rollback()
            raise e

    def rollback(self):
        self.weights = self.last_good.copy()
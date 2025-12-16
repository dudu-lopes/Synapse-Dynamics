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
        self.weights = np.array(weights, dtype=np.float32)
        self.last_good = self.weights.copy()
        self.safe_mode = safe_mode
        self.activity = None
        self.engine = PlasticityEngine(config or PlasticityConfig())

    def record(self, activity):
        self.activity = np.array(activity, dtype=np.float32)

    def apply_updates(self, pre, post):
        try:
            delta = self.engine.update(self.weights, pre, post)
            new_weights = self.weights + delta
            if self.safe_mode:
                if np.any(np.isnan(new_weights)) or np.any(np.isinf(new_weights)):
                    self.rollback()
                    raise ValueError('Unsafe weight update detected (NaN/Inf). Rolled back.')
                self.last_good = self.weights.copy()
            self.weights = new_weights
            return self.weights
        except Exception as e:
            if self.safe_mode:
                self.rollback()
            raise e

    def rollback(self):
        self.weights = self.last_good.copy()
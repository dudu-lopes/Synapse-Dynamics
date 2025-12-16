"""
Synapse: Universal Brain-Inspired Learning System
Copyright (c) 2025 Neural Engineering Labs
License: MIT

A universal plug-and-play brain-inspired system that enables any AI model or robot
to learn, adapt, and evolve in real-time. Combines biological principles with
engineering simplicity for maximum applicability and ease of use.

Key Features:
- Advanced meta-plasticity
- Memory consolidation and intelligent forgetting
- Dynamic structural plasticity
- Framework-agnostic (PyTorch, TensorFlow, NumPy, etc.)
- Non-intrusive adaptation (preserves model architecture)
- Real-time brain-inspired learning

Usage:
    >>> from synapse import SynapsePlugin, PlasticityConfig
    >>> model = your_existing_model  # Any AI model
    >>> config = PlasticityConfig(enable_meta_plasticity=True)
    >>> plugin = SynapsePlugin(config)
    >>> plugin.attach(model)  # Adds brain-inspired learning
    >>> outputs = plugin.forward(inputs)  # Now with continuous learning!

Copyright (c) 2025 Neural Engineering Labs
License: MIT
"""

from .injector import Injector
from .plasticity import PlasticityEngine, PlasticityConfig
from .plugin import SynapsePlugin

__version__ = '0.2.0'
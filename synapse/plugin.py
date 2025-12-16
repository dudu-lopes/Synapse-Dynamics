"""
Synapse Plugin Shim: Minimal entrypoint for plugin/extension systems.
"""
from .injector import Injector
from .plasticity import PlasticityEngine, PlasticityConfig

class SynapsePlugin:
    def __init__(self):
        self.name = "Synapse Dynamics"
        self.version = "1.0"
        self.Injector = Injector
        self.PlasticityEngine = PlasticityEngine
        self.PlasticityConfig = PlasticityConfig

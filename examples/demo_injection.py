"""
Exemplo de como injetar o Synapse em qualquer modelo de IA.
Demonstra o uso com diferentes tipos de modelos comuns.
"""

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from typing import Any, Dict

from synapse import SynapsePlugin
from synapse.adapters import PyTorchAdapter, TensorFlowAdapter
from synapse.plasticity import PlasticityConfig

# 1. Exemplo com PyTorch
def inject_synapse_torch():
    # Criar um modelo PyTorch qualquer
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    # Criar o modelo
    model = SimpleNet()
    
    # Configurar plasticidade
    config = PlasticityConfig(
        base_learning_rate=0.01,
        adaptation_factor=0.1,
        enable_meta_plasticity=True
    )
    
    # Criar plugin Synapse
    synapse = SynapsePlugin(config)
    
    # Injetar plasticidade (torna o modelo adaptativo)
    synapse.attach(model)
    
    # Uso normal, com aprendizado contínuo automático
    x = torch.randn(32, 10)
    out = synapse.forward(x)  # Forward pass com plasticidade
    
    return synapse, model

# 2. Exemplo com TensorFlow
def inject_synapse_tensorflow():
    # Criar um modelo TensorFlow qualquer
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(5)
    ])
    
    # Configurar e injetar Synapse
    config = PlasticityConfig(
        learning_rate=0.01,
        enable_consolidation=True,
        consolidation_rate=0.1
    )
    
    synapse = SynapsePlugin(config)
    synapse.attach(model)
    
    # Uso com plasticidade
    x = tf.random.normal((32, 10))
    out = synapse.forward(x)
    
    return synapse, model

# 3. Exemplo com NumPy (qualquer modelo baseado em arrays)
def inject_synapse_numpy():
    # Criar um modelo simples com pesos numpy
    class NumpyNet:
        def __init__(self):
            self.w1 = np.random.randn(10, 20) * 0.1
            self.w2 = np.random.randn(20, 5) * 0.1
            
        def forward(self, x):
            h = np.maximum(0, x @ self.w1)  # ReLU
            return h @ self.w2
    
    model = NumpyNet()
    
    # Configurar Synapse com plasticidade estrutural
    config = PlasticityConfig(
        enable_structural_plasticity=True,
        growth_rate=0.1,
        prune_rate=0.01
    )
    
    synapse = SynapsePlugin(config)
    synapse.attach(model)
    
    # Uso com adaptação automática
    x = np.random.randn(32, 10)
    out = synapse.forward(x)
    
    return synapse, model

# Exemplo de uso com feedback de recompensa
def train_with_reward(synapse: SynapsePlugin, inputs: Any, reward: float):
    """Treinar com feedback de recompensa."""
    # Forward pass com plasticidade
    outputs = synapse.forward(inputs)
    
    # Atualizar com sinal de recompensa
    synapse.update(reward)
    
    return outputs

# Exemplo de uso com consolidação de memória
def use_with_consolidation(synapse: SynapsePlugin):
    """Demonstra uso de consolidação de memória."""
    # Salvar estado atual (checkpoint)
    synapse.save_state("checkpoint.pt")
    
    # Realizar algumas atualizações
    for _ in range(10):
        x = torch.randn(32, 10)
        out = synapse.forward(x)
        synapse.update(reward=0.8)
    
    # Consolidar memórias importantes
    synapse.consolidate()
    
    # Carregar estado anterior se necessário
    synapse.load_state("checkpoint.pt")

if __name__ == "__main__":
    # 1. Testar com PyTorch
    print("\nTestando injeção em modelo PyTorch:")
    synapse_torch, model_torch = inject_synapse_torch()
    
    # 2. Testar com TensorFlow
    print("\nTestando injeção em modelo TensorFlow:")
    synapse_tf, model_tf = inject_synapse_tensorflow()
    
    # 3. Testar com NumPy
    print("\nTestando injeção em modelo NumPy:")
    synapse_np, model_np = inject_synapse_numpy()
    
    print("\nSynapse injetado com sucesso em todos os modelos!")
"""Tiny demo showing continuous learning injection on a simple numpy model."""
import numpy as np
from time import sleep

from synapse import SynapsePlugin

# Simple model: a single linear layer y = Wx
class SimpleNumpyModel:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(out_dim, in_dim) * 0.1

    def forward(self, x):
        return self.W @ x

    # Adapter-like helpers used by the plugin/injector
    def get_layer_output(self, layer_name, x):
        # For this simple model, ignore layer_name
        return self.forward(x)

    def get_weights(self, layer_name):
        return self.W

    def set_weights(self, weights):
        self.W = weights

    # minimal interface
    @property
    def layers(self):
        return ['linear']


def run_demo():
    """Run enhanced demo showcasing advanced continual learning features."""
    # Create model with enhanced configuration
    model = SimpleNumpyModel(8, 4)
    plugin = SynapsePlugin(
        learning_rate=0.01,
        adaptation_rate=0.05,
        stabilization_factor=0.5,
        enable_real_time=True
    )
    plugin.attach(model)

    # Generate more challenging task sequence
    tasks = []
    
    # Task A: Standard classification
    print("Generating Task A: Standard classification...")
    for _ in range(200):
        x = np.concatenate([np.random.randn(4)*0.5 + 2.0, np.random.randn(4)*0.1])
        y = np.ones(4)
        tasks.append((x, y))
        
    # Task B: Inverted classification
    print("Generating Task B: Inverted classification...")
    for _ in range(200):
        x = np.concatenate([np.random.randn(4)*0.1, np.random.randn(4)*0.5 + 2.0])
        y = -np.ones(4)
        tasks.append((x, y))
        
    # Task C: Mixed features (tests generalization)
    print("Generating Task C: Mixed features...")
    for _ in range(200):
        mask = np.random.rand(8) > 0.5
        x = np.where(mask, np.random.randn(8)*0.5 + 1.0, np.random.randn(8)*0.1)
        y = np.where(np.mean(x[:4]) > np.mean(x[4:]), np.ones(4), -np.ones(4))
        tasks.append((x, y))

    # Performance tracking
    losses = []
    task_losses = {
        'A': [], 'B': [], 'C': []
    }
    consolidations = 0
    
    print("\nStarting training loop...")
    print("=" * 50)
    
    # Enhanced training loop
    for i, (x, y) in enumerate(tasks):
        # Forward pass with plasticity
        out = plugin.forward(x)
        loss = np.mean((out - y)**2)
        losses.append(loss)
        
        # Track task-specific performance
        task_idx = i // 200
        task_name = ['A', 'B', 'C'][task_idx]
        task_losses[task_name].append(loss)
        
        # Priority replay buffer update
        error = abs(float(loss))
        plugin.injector.replay.add(x, y)
        
        # Adaptive learning with replay
        if i % 25 == 0:
            batch, indices, weights = plugin.injector.replay.sample(batch_size=16)
            replay_losses = []
            for (x_replay, y_replay), weight in zip(batch, weights):
                out_replay = plugin.forward(x_replay)
                replay_loss = np.mean((out_replay - y_replay)**2) * weight
                replay_losses.append(replay_loss)
            
            # Update replay priorities
            plugin.injector.replay.update_priorities(indices, replay_losses)
            
        # Dynamic consolidation based on performance
        if i > 0 and i % plugin.injector.consolidation_schedule[consolidations % len(plugin.injector.consolidation_schedule)] == 0:
            plugin.injector.consolidate()
            consolidations += 1
            print(f"\nConsolidation {consolidations} at step {i}")
            print(f"Recent performance: {np.mean(losses[-20:]):.4f}")
            print(f"Task losses: A={np.mean(task_losses['A'][-20:] or [0]):.4f}, "
                  f"B={np.mean(task_losses['B'][-20:] or [0]):.4f}, "
                  f"C={np.mean(task_losses['C'][-20:] or [0]):.4f}")
            
        # Progress updates
        if i % 100 == 0:
            print(f"\nStep {i}: Loss={np.mean(losses[-20:]):.4f}")
            print(f"Current task: {task_name}")
            plugin.injector._adjust_hyperparameters(np.mean(losses[-20:]))
            plugin.injector._update_consolidation_schedule()
            
        # Small sleep to simulate real-time processing
        sleep(0.005)

    print("Demo finished. Final stats:")
    print(plugin.get_stats())

if __name__ == '__main__':
    run_demo()

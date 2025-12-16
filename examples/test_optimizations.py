"""
Test script to verify optimized components of the Synapse system.
Tests ReplayBuffer, AdaptiveLearningSystem, and PerformanceMetrics.
"""
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.continual import ReplayBuffer
from synapse.adaptive_learning import AdaptiveLearningSystem, PerformanceMetrics
from synapse.plasticity import PlasticityConfig
from synapse.plasticity_rules import PlasticityRule

def test_replay_buffer():
    """Test ReplayBuffer with dynamic batch sizing."""
    print("\nTesting ReplayBuffer...")
    
    # Initialize buffer
    buffer = ReplayBuffer(
        capacity=1000,
        min_batch_size=16,
        max_batch_size=128
    )
    
    # Add some synthetic data
    for i in range(500):
        x = np.random.randn(10)  # 10-dimensional input
        y = np.sum(x) + np.random.randn() * 0.1  # Noisy target
        buffer.add(x, y)
        
    # Test sampling with different gradient norms
    print("Testing dynamic batch sizing:")
    for grad_norm in [0.1, 1.0, 5.0]:
        batch, indices, weights = buffer.sample(grad_norm=grad_norm)
        print(f"Grad norm: {grad_norm:.1f}, Batch size: {len(indices)}")
    
    # Test memory statistics
    stats = buffer.get_stats()
    print("\nBuffer Statistics:")
    print(f"Size: {stats['size']}/{stats['capacity']}")
    print(f"Optimal batch size: {stats['optimal_batch_size']}")
    print(f"Memory pressure: {stats['memory_pressure']:.2%}")

def test_adaptive_learning():
    """Test AdaptiveLearningSystem with sparse updates."""
    print("\nTesting AdaptiveLearningSystem...")
    
    # Initialize system
    config = PlasticityConfig(
        plasticity_rules=[PlasticityRule.HEBBIAN],
        base_learning_rate=0.1
    )
    system = AdaptiveLearningSystem(
        config,
        adaptation_interval=timedelta(seconds=0.1),
        update_threshold=1e-4,
        sparsity_target=0.1
    )
    
    # Test weight updates with synthetic data
    weights = np.random.randn(100, 100) * 0.1
    for i in range(10):
        pre_act = np.random.randn(100)
        post_act = np.random.randn(100)
        
        new_weights = system.update_weights(pre_act, post_act, weights)
        
        # Check sparsity
        changes = np.abs(new_weights - weights) > 0
        sparsity = np.mean(changes)
        print(f"Update {i+1}: Sparsity = {sparsity:.2%}")
        
        weights = new_weights
        
        # Simulate error for adaptation
        error = np.random.rand() * (0.9 ** i)  # Decreasing error
        system.adapt_parameters(error)
    
    # Check final status
    status = system.get_status()
    print("\nSystem Status:")
    print(f"Learning rate: {status['learning_rate']:.3f}")
    print(f"Sparsity target: {status['sparsity_target']:.2%}")

def test_performance_metrics():
    """Test PerformanceMetrics with circular buffers."""
    print("\nTesting PerformanceMetrics...")
    
    # Initialize metrics
    metrics = PerformanceMetrics(buffer_capacity=100)
    
    # Add synthetic metrics
    for i in range(200):
        error = 1.0 / (i + 1)  # Decreasing error
        activity = np.random.rand()
        weight_change = 0.1 * (0.95 ** i)  # Decreasing weight changes
        
        metrics.add_metrics(
            error=error,
            activity=activity,
            weight_change=weight_change
        )
    
    # Get recent metrics
    window = timedelta(milliseconds=100)
    recent = metrics.get_recent_metrics(window)
    
    print("Recent Metrics:")
    for metric, stats in recent.items():
        if isinstance(stats, dict):
            print(f"{metric}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value:.3f}")
        else:
            print(f"{metric}: {stats:.3f}")

if __name__ == "__main__":
    print("Running Synapse optimization tests...")
    
    test_replay_buffer()
    test_adaptive_learning()
    test_performance_metrics()
    
    print("\nAll tests completed!")
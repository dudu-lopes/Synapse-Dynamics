# Synapse Dynamics

Synapse Dynamics is a weight modulation system inspired by human neuroplasticity, ready to be integrated into any AI model without altering its architecture.

## Objetivo
Allow neural models to learn and adapt dynamically, with safe plastic rules (Hebbian, Oja) and automatic rollback for maximum robustness. Focus on reducing training time and facilitating experimentation.

## Easy Installation
Requirements: Python 3.8+

1. Install directly via pip (in the project directory):

```bash
pip install .
```

Or just copy the `synapse/` folder to your project and install numpy:

```bash
pip install numpy
```

To run the tests (optional):
```bash
pip install pytest
pytest tests/
```

## Example of Use
```python
import numpy as np
from synapse import Injector, PlasticityConfig

# Initialize weights and activities
weights = np.zeros((2,2), dtype=np.float32)
pre = np.array([1, 0], dtype=np.float32)
post = np.array([0, 1], dtype=np.float32)

# Create an injector with a Hebbian rule
inj = Injector(weights, PlasticityConfig(rule='hebbian', lr=0.5))
updated = inj.apply_updates(pre, post)
print(updated)
```

## Recommendations for Efficient Use
- For maximum GPU performance, keep tensors on the correct device (use `.to(device)` in PyTorch).
- Use batch updates whenever possible to take advantage of parallelism.
- The Injector automatically converts between numpy, torch, and tensorflow.
- For large models, adjust the `lr` parameter to avoid numerical instability.
- Benchmarks show that weight modulation is efficient and does not negatively impact training time.

## Resources
- PlasticityEngine: weight update with validation and rollback
- Injector: pluggable interface for numpy, torch, and tensorflow arrays
- Plugin Shim: easy integration with extension systems
- No dependencies other than numpy

## License
MIT

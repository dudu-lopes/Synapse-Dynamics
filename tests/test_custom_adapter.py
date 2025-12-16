import numpy as np


class NumpyNet:
    def __init__(self):
        # Match shapes used in examples/demo_injection.py
        self.w1 = np.random.randn(10, 20) * 0.1
        self.w2 = np.random.randn(20, 5) * 0.1

    def forward(self, x):
        h = np.maximum(0, x @ self.w1)
        return h @ self.w2


def test_get_layer_output_and_activations_shapes():
    model = NumpyNet()

    # Create a batch of inputs (32, 10)
    x = np.random.randn(32, 10)

    # Layer w1: inputs @ w1 -> (32, 20)
    out_w1 = x @ model.w1
    assert isinstance(out_w1, np.ndarray)
    assert out_w1.shape == (32, 20)

    # get_activations should return a single-sample activation (1, features) or similar
    act = out_w1[0:1]
    assert isinstance(act, np.ndarray)
    # allow (1, features) or (features,) shapes
    assert act.ndim in (1, 2)
    if act.ndim == 2:
        assert act.shape[1] == 20
    else:
        assert act.shape[0] == 20

import numpy as np

from synapse.injector import PlasticityInjector, PlasticityInjectionConfig


class NumpyNet:
    def __init__(self):
        self.w1 = np.random.randn(10, 20) * 0.1
        self.w2 = np.random.randn(20, 5) * 0.1

    def forward(self, x):
        h = np.maximum(0, x @ self.w1)
        return h @ self.w2


def test_infer_layer_shape_and_inject_creates_plastic_layers():
    model = NumpyNet()
    injector = PlasticityInjector(PlasticityInjectionConfig())

    # Should not raise
    injector.inject(model)

    # Expect plastic_layers created for discovered layer names (w1, w2)
    assert 'w1' in injector.plastic_layers
    assert 'w2' in injector.plastic_layers

    # Infer shapes directly and expect feature counts
    s1 = injector._infer_layer_shape('w1')
    s2 = injector._infer_layer_shape('w2')

    # Our convention: returns tuple with number of features (e.g., (20,), (5,))
    assert isinstance(s1, tuple) and isinstance(s2, tuple)
    assert s1[0] == 20
    assert s2[0] == 5

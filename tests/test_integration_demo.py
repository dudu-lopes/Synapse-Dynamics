import subprocess
import sys
import os
import pytest
import numpy as np
from synapse import Injector, PlasticityConfig

# Teste desativado: dependia de exemplos e APIs antigas não compatíveis com o núcleo minimalista.
# def test_demo_injection_runs():
#     """Smoke test: demo_injection.py runs end-to-end without error."""
#     script = os.path.join(os.path.dirname(__file__), '../examples/demo_injection.py')
#     script = os.path.abspath(script)
#     result = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=60)
#     print(result.stdout)
#     assert result.returncode == 0
#     assert "Synapse injetado com sucesso" in result.stdout

def test_simple_integration():
    w = np.zeros((5, 5), dtype=np.float32)
    inj = Injector(w, PlasticityConfig(rule='hebbian', lr=0.2))
    pre = np.arange(5, dtype=np.float32)
    post = np.arange(5, dtype=np.float32)
    updated = inj.apply_updates(pre, post)
    assert updated.shape == (5, 5)
    assert np.allclose(updated, updated.T)

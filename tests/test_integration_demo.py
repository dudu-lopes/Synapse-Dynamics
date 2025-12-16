import subprocess
import sys
import os
import pytest

def test_demo_injection_runs():
    """Smoke test: demo_injection.py runs end-to-end without error."""
    script = os.path.join(os.path.dirname(__file__), '../examples/demo_injection.py')
    script = os.path.abspath(script)
    result = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=60)
    print(result.stdout)
    assert result.returncode == 0
    assert "Synapse injetado com sucesso" in result.stdout

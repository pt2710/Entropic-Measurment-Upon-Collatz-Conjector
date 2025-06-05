import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.your_module import core_function

@pytest.mark.parametrize("x", [0, 1, -5])
def test_core_function_returns_expected(x):
    assert core_function(x) == 42

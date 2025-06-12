import pytest
import torch
import ast

from softassertion.engine.input_selector import get_input_generator
from softassertion.engine.oracle_selector import get_oracle_for_function
from softassertion.analysis.function_finder import find_unstable_calls
from softassertion.engine import fuzz_runner

# ---------- input_selector ----------
def test_input_generator_softmax():
    generator = get_input_generator("softmax")
    assert generator is not None
    sample = generator()
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, 3)

# ---------- oracle_selector ----------
def test_oracle_for_log():
    oracle = get_oracle_for_function("log")
    assert oracle is not None
    x = torch.tensor([[0.5, -1.0], [1.0, 2.0]])
    result = oracle(torch.log(torch.abs(x)))
    assert isinstance(result, bool)

# ---------- function_finder ----------
def test_function_finder_detects_unstable():
    code = "x = torch.log(torch.rand(3,3) - 0.5)"
    found = find_unstable_calls(code)
    assert "log" in found

# ---------- fuzz_runner smoke test ----------
def test_fuzz_runner_smoke():
    try:
        fuzz_runner.run_fuzzing_on_calls(["softmax"])
    except Exception as e:
        pytest.fail(f"Fuzz runner crashed: {e}")


# test_perlin_utils.py
import pytest
import numpy as np
import torch
from transforms_class.utils.perlin_utils import random_perlin_threshold, generate_perlin_map


# ---------------------------------------------------------------------------
# random_perlin_threshold
# ---------------------------------------------------------------------------

class TestRandomPerlinThreshold:

    def test_returns_float(self):
        v = random_perlin_threshold()
        assert isinstance(v, float)

    def test_within_default_range(self):
        for _ in range(50):
            v = random_perlin_threshold()
            assert 0.0 <= v <= 0.4 + 1e-9, f"Out of range: {v}"

    def test_custom_range(self):
        for _ in range(30):
            v = random_perlin_threshold(start=0.1, end=0.3, step=0.05)
            assert 0.1 <= v <= 0.3 + 1e-9

    def test_small_values_more_frequent(self):
        """Bias toward 0: mean of many samples should be < midpoint."""
        samples = [random_perlin_threshold() for _ in range(500)]
        midpoint = 0.2
        assert np.mean(samples) < midpoint, "Expected bias toward smaller values"

    def test_values_are_multiples_of_step(self):
        step = 0.05
        for _ in range(30):
            v = random_perlin_threshold(step=step)
            remainder = round(v / step, 6) % 1
            assert remainder < 1e-4 or remainder > 1 - 1e-4, f"Not a multiple of step: {v}"


# ---------------------------------------------------------------------------
# generate_perlin_map
# ---------------------------------------------------------------------------

class TestGeneratePerlinMap:

    def test_output_shape_matches_input(self):
        shape = (8, 10, 12)
        result = generate_perlin_map(shape)
        assert result.shape == shape

    def test_output_is_numpy_or_torch(self):
        result = generate_perlin_map((4, 4, 4))
        assert isinstance(result, (np.ndarray, torch.Tensor))

    def test_values_in_reasonable_range(self):
        """Perlin noise values are typically in [-1, 1]."""
        result = generate_perlin_map((8, 8, 8))
        arr = result if isinstance(result, np.ndarray) else result.numpy()
        assert arr.min() >= -1.5 and arr.max() <= 1.5

    def test_same_base_produces_same_output(self):
        shape = (6, 6, 6)
        r1 = generate_perlin_map(shape, base=42)
        r2 = generate_perlin_map(shape, base=42)
        arr1 = r1 if isinstance(r1, np.ndarray) else r1.numpy()
        arr2 = r2 if isinstance(r2, np.ndarray) else r2.numpy()
        np.testing.assert_array_equal(arr1, arr2)

    def test_different_base_produces_different_output(self):
        shape = (6, 6, 6)
        r1 = generate_perlin_map(shape, base=0)
        r2 = generate_perlin_map(shape, base=7)
        arr1 = r1 if isinstance(r1, np.ndarray) else r1.numpy()
        arr2 = r2 if isinstance(r2, np.ndarray) else r2.numpy()
        assert not np.array_equal(arr1, arr2)

    def test_scale_affects_output(self):
        shape = (8, 8, 8)
        r1 = generate_perlin_map(shape, scale=5.0)
        r2 = generate_perlin_map(shape, scale=50.0)
        arr1 = r1 if isinstance(r1, np.ndarray) else r1.numpy()
        arr2 = r2 if isinstance(r2, np.ndarray) else r2.numpy()
        assert not np.array_equal(arr1, arr2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

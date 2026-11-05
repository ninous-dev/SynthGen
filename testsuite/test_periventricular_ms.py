# test_periventricular_ms.py
import pytest
from transforms_class.PeriventricularMS import PeriventricularMS


def _make(**kwargs):
    defaults = dict(
        ventricular_label=4,
        grid_space=10,
        labels_to_target=[2, 3],
        lesion_label=100,
    )
    defaults.update(kwargs)
    return PeriventricularMS(**defaults)


class TestPeriventricularMSInit:
    def test_suffix_set(self):
        assert _make().anomaly_suffix == "PeriVent"

    def test_ventricular_label_stored(self):
        assert _make(ventricular_label=7).ventricular_label == 7

    def test_labels_to_target_stored(self):
        assert _make(labels_to_target=[1, 2]).labels_to_target == [1, 2]

    def test_invalid_perlin_threshold_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            _make(perlin_threshold=1.5)

    def test_valid_perlin_threshold_does_not_raise(self):
        _make(perlin_threshold=0.5)  # no exception


class TestPeriventricularMSValidation:
    def test_threshold_tuple_valid(self):
        t = _make(perlin_threshold=(0.3, 0.7))
        # Stored as-is, will be resolved at call time
        assert t.perlin_threshold == (0.3, 0.7)

    def test_grid_space_list_stored(self):
        t = _make(grid_space=[5, 10, 20])
        assert isinstance(t.grid_space, list)

    def test_displacement_list_stored(self):
        t = _make(displacement=[1.0, 2.0])
        assert isinstance(t.displacement, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

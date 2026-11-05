# test_ms_transform.py
"""
Tests for MSTransform.

nb_sclerosis normalisation is handled by AnomalyLabelTransform.int_to_tuple
at __init__ time, so we test the observable init behaviour.
"""
import pytest
from transforms_class.MSTransform import MSTransform
from transforms_class.AnomalyLabelTransform import AnomalyLabelTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(**kwargs) -> MSTransform:
    defaults = dict(
        labels_to_target=1,
        nb_sclerosis=5,
        lesion_label=100,
        background_value=0,
    )
    defaults.update(kwargs)
    return MSTransform(**defaults)


# ---------------------------------------------------------------------------
# __init__ — nb_sclerosis storage
# ---------------------------------------------------------------------------

class TestNbSclerosisInit:

    def test_int_converted_to_symmetric_tuple(self):
        t = _make(nb_sclerosis=7)
        assert t.nb_sclerosis == (7, 7)

    def test_valid_tuple_stored_as_is(self):
        t = _make(nb_sclerosis=(3, 10))
        assert t.nb_sclerosis == (3, 10)

    def test_valid_list_stored_as_is(self):
        t = _make(nb_sclerosis=[2, 8])
        assert t.nb_sclerosis == [2, 8]

    def test_reversed_bounds_raise_at_init(self):
        """int_to_tuple enforces min <= max."""
        with pytest.raises(ValueError):
            _make(nb_sclerosis=[10, 3])


# ---------------------------------------------------------------------------
# __init__ — labels_to_target storage
# ---------------------------------------------------------------------------

class TestLabelsToTargetInit:

    def test_int_converted_to_list(self):
        t = _make(labels_to_target=2)
        assert t.labels_to_target == [2]

    def test_list_stored_as_is(self):
        t = _make(labels_to_target=[1, 2, 3])
        assert t.labels_to_target == [1, 2, 3]


# ---------------------------------------------------------------------------
# __init__ — other attributes
# ---------------------------------------------------------------------------

class TestMSTransformInit:

    def test_anomaly_suffix_is_ms(self):
        assert _make().anomaly_suffix == "MS"

    def test_lesion_label_stored(self):
        assert _make(lesion_label=42).lesion_label == 42

    def test_sclerosis_size_none_by_default(self):
        t = _make()
        assert t.sclerosis_min_size is None
        assert t.sclerosis_max_size is None

    def test_sclerosis_size_tuple_unpacked(self):
        t = _make(sclerosis_size=(3, 9))
        assert t.sclerosis_min_size == 3
        assert t.sclerosis_max_size == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


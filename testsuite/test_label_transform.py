# test_label_transform.py
import pytest
from transforms_class.LabelTransform import LabelTransform


# ---------------------------------------------------------------------------
# resolve_value
# ---------------------------------------------------------------------------

class TestResolveValue:
    def test_string_resolved_to_int(self):
        assert LabelTransform.resolve_value("WM", {"WM": 1}) == 1

    def test_int_key_in_map_resolved(self):
        assert LabelTransform.resolve_value(1, {1: 5}) == 5

    def test_int_key_not_in_map_returned_as_is(self):
        assert LabelTransform.resolve_value(3, {"WM": 1}) == 3

    def test_float_returned_as_is(self):
        assert LabelTransform.resolve_value(1.5, {"WM": 1}) == 1.5

    def test_list_resolved_to_list_of_ints(self):
        result = LabelTransform.resolve_value(["WM", "GM"], {"WM": 1, "GM": 2})
        assert set(result) == {1, 2}

    def test_list_deduplicates(self):
        result = LabelTransform.resolve_value(["WM", "WM"], {"WM": 1})
        assert result == [1]

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            LabelTransform.resolve_value({"a": 1}, {"WM": 1})

    def test_nan_filtered_from_list(self):
        import math
        result = LabelTransform.resolve_value("WM", {"WM": [1, float("nan")]})
        assert all(not (isinstance(x, float) and math.isnan(x)) for x in result)


# ---------------------------------------------------------------------------
# resolve_anatomy
# ---------------------------------------------------------------------------

class TestResolveAnatomy:

    def _make_transform(self, **anatomy_kwargs):
        """Create a minimal LabelTransform with given anatomy args."""
        class _Concrete(LabelTransform):
            def apply_transform(self, subject):
                return subject

        t = _Concrete()
        for k, v in anatomy_kwargs.items():
            setattr(t, k, v)
        # Only register provided keys so resolve_anatomy touches them
        t.anatomy_args = list(anatomy_kwargs.keys())
        return t

    def test_string_labels_to_target_resolved(self):
        t = self._make_transform(labels_to_target="WM")
        t.resolve_anatomy({"WM": 1})
        assert t.labels_to_target == 1

    def test_list_labels_to_keep_resolved(self):
        t = self._make_transform(labels_to_keep=["WM", "GM"])
        t.resolve_anatomy({"WM": 1, "GM": 2})
        assert set(t.labels_to_keep) == {1, 2}

    def test_none_attribute_not_touched(self):
        t = self._make_transform(labels_to_target=None)
        t.resolve_anatomy({"WM": 1})
        assert t.labels_to_target is None


# ---------------------------------------------------------------------------
# check_anatomy_resolved
# ---------------------------------------------------------------------------

class TestCheckAnatomyResolved:

    def _make_transform(self, labels_to_target):
        class _Concrete(LabelTransform):
            def apply_transform(self, subject):
                return subject

        t = _Concrete()
        t.labels_to_target = labels_to_target
        t.anatomy_args = ["labels_to_target"]
        return t

    def test_int_passes(self):
        t = self._make_transform(3)
        t.check_anatomy_resolved()  # no raise

    def test_list_of_ints_passes(self):
        t = self._make_transform([1, 2, 3])
        t.check_anatomy_resolved()  # no raise

    def test_string_raises(self):
        t = self._make_transform("WM")
        with pytest.raises(ValueError):
            t.check_anatomy_resolved()

    def test_list_with_string_raises(self):
        t = self._make_transform([1, "WM"])
        with pytest.raises(ValueError):
            t.check_anatomy_resolved()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

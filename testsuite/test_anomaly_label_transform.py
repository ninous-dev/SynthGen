# test_anomaly_label_transform.py
import pytest
import torch
import torchio as tio
from transforms_class.AnomalyLabelTransform import AnomalyLabelTransform


# ---------------------------------------------------------------------------
# Minimal concrete subclass (AnomalyLabelTransform is abstract)
# ---------------------------------------------------------------------------

class _DummyAnomaly(AnomalyLabelTransform):
    """Concrete stub that does nothing to the subject."""
    def __init__(self, **kwargs):
        super().__init__(anomaly_suffix="Dummy", **kwargs)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        return subject


def _make_subject(label_value: int = 1) -> tio.Subject:
    tensor = torch.full((1, 8, 8, 8), label_value, dtype=torch.long)
    s = tio.Subject(label=tio.LabelMap(tensor=tensor, affine=torch.eye(4)))
    s["suffix"] = ""
    return s


# ---------------------------------------------------------------------------
# int_to_list
# ---------------------------------------------------------------------------

class TestIntToList:
    def test_int_becomes_list(self):
        assert AnomalyLabelTransform.int_to_list(3) == [3]

    def test_list_unchanged(self):
        assert AnomalyLabelTransform.int_to_list([1, 2]) == [1, 2]

    def test_none_unchanged(self):
        assert AnomalyLabelTransform.int_to_list(None) is None


# ---------------------------------------------------------------------------
# int_to_tuple
# ---------------------------------------------------------------------------

class TestIntToTuple:
    def test_int_becomes_symmetric_tuple(self):
        assert AnomalyLabelTransform.int_to_tuple(5) == (5, 5)

    def test_valid_list_unchanged(self):
        assert AnomalyLabelTransform.int_to_tuple([2, 8]) == [2, 8]

    def test_reversed_list_raises(self):
        with pytest.raises(ValueError):
            AnomalyLabelTransform.int_to_tuple([10, 3])

    def test_tuple_unchanged(self):
        assert AnomalyLabelTransform.int_to_tuple((1, 5)) == (1, 5)


# ---------------------------------------------------------------------------
# choose_random_param
# ---------------------------------------------------------------------------

class TestChooseRandomParam:
    def test_scalar_int_returns_as_is(self):
        assert AnomalyLabelTransform.choose_random_param(7) == 7

    def test_scalar_float_returns_as_is(self):
        assert AnomalyLabelTransform.choose_random_param(0.5) == 0.5

    def test_list_choice_returns_element_from_list(self):
        opts = [10, 20, 30]
        result = AnomalyLabelTransform.choose_random_param(opts, list_choice=True)
        assert result in opts

    def test_int_tuple_range(self):
        for _ in range(20):
            v = AnomalyLabelTransform.choose_random_param((2, 8))
            assert 2 <= v <= 8

    def test_float_tuple_range(self):
        for _ in range(20):
            v = AnomalyLabelTransform.choose_random_param((0.1, 0.9))
            assert 0.1 <= v <= 0.9


# ---------------------------------------------------------------------------
# update_subject_anomaly
# ---------------------------------------------------------------------------

class TestUpdateSubjectAnomaly:
    def test_anomaly_type_set_on_new_subject(self):
        t = _DummyAnomaly()
        s = _make_subject()
        t.update_subject_anomaly(s)
        assert s["anomaly_type"] == "Dummy"

    def test_anomaly_type_appended_if_different(self):
        t = _DummyAnomaly()
        s = _make_subject()
        s["anomaly_type"] = "PrevAnomaly"
        t.update_subject_anomaly(s)
        assert "PrevAnomaly" in s["anomaly_type"]
        assert "Dummy" in s["anomaly_type"]

    def test_suffix_appended(self):
        t = _DummyAnomaly()
        s = _make_subject()
        s["suffix"] = "Ano"
        t.update_subject_anomaly(s)
        assert "Dummy" in s["suffix"]

    def test_suffix_not_duplicated(self):
        t = _DummyAnomaly()
        s = _make_subject()
        s["suffix"] = "AnoDummy"
        t.update_subject_anomaly(s)
        assert s["suffix"].count("Dummy") == 1


# ---------------------------------------------------------------------------
# get_contour3d
# ---------------------------------------------------------------------------

class TestGetContour3d:
    def _sphere_mask(self, r=3, size=12) -> torch.Tensor:
        """Build a filled sphere boolean mask (1, D, H, W)."""
        c = size // 2
        z, y, x = torch.meshgrid(
            torch.arange(size), torch.arange(size), torch.arange(size), indexing="ij"
        )
        return ((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= r ** 2).unsqueeze(0)

    def test_external_contour_is_outside_mask(self):
        mask = self._sphere_mask()
        contour = AnomalyLabelTransform.get_contour3d(mask, iterations=1, external=True)
        # Contour must not overlap original mask
        assert not (contour & mask).any()

    def test_internal_contour_is_inside_mask(self):
        mask = self._sphere_mask()
        contour = AnomalyLabelTransform.get_contour3d(mask, iterations=1, external=False)
        assert (contour & mask).any()
        assert not (contour & ~mask).any()

    def test_raises_on_non_bool_tensor(self):
        mask = torch.ones(1, 8, 8, 8, dtype=torch.float32)
        with pytest.raises(TypeError):
            AnomalyLabelTransform.get_contour3d(mask)

    def test_raises_on_zero_iterations(self):
        mask = self._sphere_mask()
        with pytest.raises(ValueError):
            AnomalyLabelTransform.get_contour3d(mask, iterations=0)

    def test_thicker_contour_with_more_iterations(self):
        mask = self._sphere_mask(r=4, size=16)
        c1 = AnomalyLabelTransform.get_contour3d(mask, iterations=1, external=True)
        c2 = AnomalyLabelTransform.get_contour3d(mask, iterations=2, external=True)
        assert c2.sum() >= c1.sum()


# ---------------------------------------------------------------------------
# anomaly_suffix / nickname
# ---------------------------------------------------------------------------

class TestAnomalySuffix:
    def test_suffix_set_from_init(self):
        t = _DummyAnomaly()
        assert t.anomaly_suffix == "Dummy"

    def test_nickname_overrides_suffix(self):
        t = _DummyAnomaly(nickname="MyNick")
        assert t.anomaly_suffix == "MyNick"

    def test_missing_suffix_raises(self):
        class _BadAnomaly(AnomalyLabelTransform):
            def __init__(self):
                super().__init__(anomaly_suffix=None)
            def apply_transform(self, subject):
                return subject

        with pytest.raises(NotImplementedError):
            _BadAnomaly()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

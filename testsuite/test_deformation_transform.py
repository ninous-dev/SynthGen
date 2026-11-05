# test_deformation_transform.py
import pytest
import torch
import torchio as tio
from transforms_class.utils.DeformationTransform import DeformationTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(**kwargs) -> DeformationTransform:
    defaults = dict(
        mode="push",
        labels_to_target=[1],
        displacement=0.5,
        grid_space=8,
    )
    defaults.update(kwargs)
    return DeformationTransform(**defaults)


def _make_subject(target_label=1, size=16) -> tio.Subject:
    tensor = torch.zeros(1, size, size, size, dtype=torch.long)
    mid = size // 2
    tensor[0, mid - 3:mid + 3, mid - 3:mid + 3, mid - 3:mid + 3] = target_label
    return tio.Subject(label=tio.LabelMap(tensor=tensor, affine=torch.eye(4)))


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

class TestDeformationTransformInit:

    def test_valid_push_mode(self):
        t = _make(mode="push")
        assert t.mode == "push"

    def test_valid_pull_mode(self):
        t = _make(mode="pull")
        assert t.mode == "pull"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            _make(mode="stretch")

    def test_invalid_random_field_mask_raises(self):
        with pytest.raises(ValueError, match="random_field_mask"):
            _make(random_field_mask="invalid_mask_type")

    def test_valid_random_field_mask_values(self):
        for mask in [None, "bidirectional", "unidirectional",
                     "perlin_unidirectional", "perlin_unidirectional_thresholded"]:
            _make(random_field_mask=mask)  # must not raise

    def test_overlapping_keep_and_target_raises(self):
        with pytest.raises(ValueError, match="overlapping"):
            _make(labels_to_target=[1, 2], labels_to_keep=[2, 3])

    def test_non_overlapping_keep_and_target_ok(self):
        _make(labels_to_target=[1], labels_to_keep=[2])  # must not raise

    def test_labels_to_target_stored(self):
        t = _make(labels_to_target=[1, 2])
        assert t.labels_to_target == [1, 2]

    def test_locked_borders_stored(self):
        t = _make(locked_borders=5)
        assert t.locked_borders == 5


# ---------------------------------------------------------------------------
# to_tensor static method
# ---------------------------------------------------------------------------

class TestToTensor:

    def test_tensor_unchanged(self):
        t = torch.ones(3, 3, 3)
        result = DeformationTransform.to_tensor(t)
        assert result is t

    def test_numpy_converted(self):
        import numpy as np
        arr = np.ones((3, 3, 3), dtype=np.float32)
        result = DeformationTransform.to_tensor(arr)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3, 3)


# ---------------------------------------------------------------------------
# is_invertible
# ---------------------------------------------------------------------------

class TestIsInvertible:
    def test_not_invertible(self):
        assert _make().is_invertible() is False


# ---------------------------------------------------------------------------
# check_target_keep_labels_consistency
# ---------------------------------------------------------------------------

class TestCheckConsistency:

    def test_overlap_raises(self):
        t = _make.__wrapped__ if hasattr(_make, "__wrapped__") else None
        dt = DeformationTransform.__new__(DeformationTransform)
        dt.labels_to_keep = [1, 2]
        dt.labels_to_target = [2, 3]
        with pytest.raises(ValueError, match="overlapping"):
            dt.check_target_keep_labels_consistency()

    def test_no_overlap_ok(self):
        dt = DeformationTransform.__new__(DeformationTransform)
        dt.labels_to_keep = [1]
        dt.labels_to_target = [2]
        dt.check_target_keep_labels_consistency()  # must not raise

    def test_none_keep_ok(self):
        dt = DeformationTransform.__new__(DeformationTransform)
        dt.labels_to_keep = None
        dt.labels_to_target = [1, 2]
        dt.check_target_keep_labels_consistency()  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

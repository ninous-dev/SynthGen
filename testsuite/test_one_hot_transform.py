# test_one_hot_transform.py
import pytest
import torch
import torchio as tio
import numpy as np
from transforms_class.OneHotTransform import OneHotTransform, fill_mask_with_nearest_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_label_subject(values, size=8) -> tio.Subject:
    """Subject with a label map containing the given integer values."""
    tensor = torch.zeros(1, size, size, size, dtype=torch.long)
    # Fill quadrants with different label values
    mid = size // 2
    for i, v in enumerate(values):
        if i == 0:
            tensor[0, :mid, :mid, :mid] = v
        elif i == 1:
            tensor[0, mid:, :mid, :mid] = v
        elif i == 2:
            tensor[0, :mid, mid:, :mid] = v
        elif i == 3:
            tensor[0, mid:, mid:, mid:] = v
    return tio.Subject(label=tio.LabelMap(tensor=tensor, affine=torch.eye(4)))


# ---------------------------------------------------------------------------
# fill_mask_with_nearest_value
# ---------------------------------------------------------------------------

class TestFillMaskWithNearestValue:
    def test_no_hole_unchanged(self):
        vol = np.ones((1, 4, 4, 4), dtype=np.int64)
        result = fill_mask_with_nearest_value(vol)
        np.testing.assert_array_equal(result, vol)

    def test_hole_filled(self):
        vol = np.ones((1, 6, 6, 6), dtype=np.int64)
        # Carve an internal hole
        vol[0, 2:4, 2:4, 2:4] = 0
        result = fill_mask_with_nearest_value(vol)
        assert result[0, 2, 2, 2] != 0, "Internal hole should be filled"

    def test_border_zero_not_filled(self):
        vol = np.ones((1, 6, 6, 6), dtype=np.int64)
        vol[0, 0, :, :] = 0  # border, should stay 0
        result = fill_mask_with_nearest_value(vol)
        assert result[0, 0, 0, 0] == 0

    def test_raises_on_wrong_ndim(self):
        with pytest.raises(ValueError):
            fill_mask_with_nearest_value(np.ones((4, 4, 4), dtype=np.int64))

    def test_raises_on_non_array(self):
        with pytest.raises(TypeError):
            fill_mask_with_nearest_value([[1, 2], [3, 4]])


# ---------------------------------------------------------------------------
# OneHotTransform — forward (encode)
# ---------------------------------------------------------------------------

class TestOneHotForward:
    def test_output_channels_equals_num_classes(self):
        subject = _make_label_subject([1, 2])
        t = OneHotTransform(num_classes=3, invert_transform=False)
        out = t(subject)
        assert out.label.data.shape[0] == 3

    def test_each_channel_is_binary(self):
        subject = _make_label_subject([1, 2])
        t = OneHotTransform(num_classes=3, invert_transform=False)
        out = t(subject)
        unique = out.label.data.unique().tolist()
        assert set(unique).issubset({0.0, 1.0})

    def test_background_dropped_when_not_segmented(self):
        subject = _make_label_subject([1, 2])
        t = OneHotTransform(num_classes=3, invert_transform=False, background_segmented=False)
        out = t(subject)
        # Channel 0 (background) removed → 2 channels
        assert out.label.data.shape[0] == 2


# ---------------------------------------------------------------------------
# OneHotTransform — inverse (decode)
# ---------------------------------------------------------------------------

class TestOneHotInverse:
    def test_invert_roundtrip(self):
        subject = _make_label_subject([0, 1, 2, 3])
        original_data = subject.label.data.clone()

        fwd = OneHotTransform(num_classes=4, invert_transform=False)
        inv = OneHotTransform(invert_transform=True)

        encoded = fwd(subject)
        decoded = inv(encoded)

        assert decoded.label.data.shape[0] == 1
        # Voxel values should match original after roundtrip
        assert (decoded.label.data == original_data).all()

    def test_fill_holes_runs_without_error(self):
        subject = _make_label_subject([0, 1, 2, 3])
        fwd = OneHotTransform(num_classes=4, invert_transform=False)
        inv = OneHotTransform(invert_transform=True, fill_holes=True)
        decoded = inv(fwd(subject))
        assert decoded.label.data.shape[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

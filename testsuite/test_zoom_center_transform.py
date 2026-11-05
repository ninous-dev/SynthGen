# test_zoom_center_transform.py
import pytest
import torch
import torchio as tio
from transforms_class.ZoomCenterTransform import ZoomCenterTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere_subject(r: int = 4, size: int = 20) -> tio.Subject:
    """Subject with a filled sphere label."""
    c = size // 2
    z, y, x = torch.meshgrid(
        torch.arange(size), torch.arange(size), torch.arange(size), indexing="ij"
    )
    sphere = ((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= r ** 2).long()
    return tio.Subject(label=tio.LabelMap(tensor=sphere.unsqueeze(0), affine=torch.eye(4)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestZoomCenterTransform:

    def test_output_shape_unchanged(self):
        subject = _sphere_subject()
        original_shape = subject.label.data.shape
        out = ZoomCenterTransform(ratio=1.5)(subject)
        assert out.label.data.shape == original_shape

    def test_zoom_in_increases_foreground(self):
        subject = _sphere_subject(r=3, size=20)
        n_before = (subject.label.data > 0).sum().item()
        out = ZoomCenterTransform(ratio=2.0)(subject)
        n_after = (out.label.data > 0).sum().item()
        assert n_after >= n_before

    def test_zoom_out_decreases_foreground(self):
        subject = _sphere_subject(r=5, size=20)
        n_before = (subject.label.data > 0).sum().item()
        out = ZoomCenterTransform(ratio=0.5)(subject)
        n_after = (out.label.data > 0).sum().item()
        assert n_after <= n_before

    def test_ratio_one_is_near_identity(self):
        subject = _sphere_subject()
        original = subject.label.data.clone()
        out = ZoomCenterTransform(ratio=1.0)(subject)
        # With ratio=1 the transform is identity; values should be identical
        assert (out.label.data == original).all()

    def test_empty_label_raises(self):
        tensor = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        subject = tio.Subject(label=tio.LabelMap(tensor=tensor, affine=torch.eye(4)))
        with pytest.raises(ValueError, match="empty"):
            ZoomCenterTransform(ratio=1.5)(subject)

    def test_apply_to_tensor_classmethod(self):
        sphere = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        sphere[0, 6:10, 6:10, 6:10] = 1
        result = ZoomCenterTransform.apply_to_tensor(sphere, ratio=1.5)
        assert result.shape == sphere.shape

    def test_apply_to_tensor_3d_input(self):
        mask = torch.zeros(16, 16, 16, dtype=torch.long)
        mask[6:10, 6:10, 6:10] = 1
        result = ZoomCenterTransform.apply_to_tensor(mask, ratio=1.5)
        assert result.shape == mask.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

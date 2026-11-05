# test_random_morphology_transform.py
import pytest
import torch
import torchio as tio
from transforms_class.RandomMorphologyTransform import RandomMorphologyTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subject(wm_label: int = 1, gm_label: int = 2, size: int = 12) -> tio.Subject:
    """Subject with a WM cube surrounded by GM."""
    tensor = torch.zeros(1, size, size, size, dtype=torch.long)
    mid = size // 2
    # Inner cube = WM
    tensor[0, mid - 2:mid + 2, mid - 2:mid + 2, mid - 2:mid + 2] = wm_label
    # Surrounding shell = GM
    tensor[0, mid - 3:mid + 3, mid - 3:mid + 3, mid - 3:mid + 3] = torch.where(
        tensor[0, mid - 3:mid + 3, mid - 3:mid + 3, mid - 3:mid + 3] == 0,
        torch.tensor(gm_label),
        tensor[0, mid - 3:mid + 3, mid - 3:mid + 3, mid - 3:mid + 3],
    )
    s = tio.Subject(label=tio.LabelMap(tensor=tensor, affine=torch.eye(4)))
    s["suffix"] = ""
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRandomMorphologyTransform:

    def test_output_shape_unchanged(self):
        subject = _make_subject()
        t = RandomMorphologyTransform(
            label_to_dilate=[1], label_within=[2], nb_iter_min=1, nb_iter_max=2
        )
        out = t(subject)
        assert out.label.data.shape == subject.label.data.shape

    def test_dilation_increases_target_label_count(self):
        subject = _make_subject(wm_label=1, gm_label=2)
        n_before = (subject.label.data == 1).sum().item()
        t = RandomMorphologyTransform(
            label_to_dilate=[1], label_within=[2], nb_iter_min=2, nb_iter_max=3
        )
        out = t(subject)
        n_after = (out.label.data == 1).sum().item()
        assert n_after >= n_before

    def test_suffix_updated(self):
        subject = _make_subject()
        t = RandomMorphologyTransform(
            label_to_dilate=[1], label_within=[2],
            nb_iter_min=1, nb_iter_max=2, transform_suffix="dil"
        )
        out = t(subject)
        assert "dil" in out["suffix"]

    def test_raises_on_multiple_labelmaps(self):
        tensor = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        subject = tio.Subject(
            label1=tio.LabelMap(tensor=tensor, affine=torch.eye(4)),
            label2=tio.LabelMap(tensor=tensor, affine=torch.eye(4)),
        )
        t = RandomMorphologyTransform(
            label_to_dilate=[1], label_within=[2], nb_iter_min=1, nb_iter_max=2
        )
        with pytest.raises(NotImplementedError):
            t(subject)

    def test_morphology_static_method(self):
        """_morphology must dilate target label into within label."""
        import numpy as np
        data = np.zeros((1, 10, 10, 10), dtype=np.int64)
        data[0, 4:6, 4:6, 4:6] = 1  # target
        data[0, 3:7, 3:7, 3:7] = np.where(data[0, 3:7, 3:7, 3:7] == 0, 2, data[0, 3:7, 3:7, 3:7])
        n_before = (data == 1).sum()
        result = RandomMorphologyTransform._morphology(
            data, target_labels=[1], iterations=1, within_labels=[2]
        )
        assert (result == 1).sum() >= n_before


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

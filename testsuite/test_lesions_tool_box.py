# test_lesions_tool_box.py
import pytest
import torch
from transforms_class.utils.lesions_tool_box import (
    compute_coverage,
    get_random_position_in_label,
    get_biggest_component,
    compute_anatomical_resistance,
)


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------

class TestComputeCoverage:

    def _make(self, size=8):
        label = torch.zeros(size, size, size, dtype=torch.long)
        label[2:6, 2:6, 2:6] = 1           # target region
        lesion = torch.zeros_like(label)
        return label, lesion

    def test_fully_inside_returns_one(self):
        label, lesion = self._make()
        lesion[3:5, 3:5, 3:5] = 1          # entirely inside label==1
        score = compute_coverage(label, lesion)
        assert score == 1.0

    def test_fully_outside_returns_zero(self):
        label, lesion = self._make()
        lesion[0, 0, 0] = 1                 # outside label==1
        score = compute_coverage(label, lesion)
        assert score == 0.0

    def test_empty_lesion_returns_zero(self):
        label, lesion = self._make()
        score = compute_coverage(label, lesion)
        assert score == (0.0, 1.0)

    def test_partial_inside(self):
        label, lesion = self._make()
        lesion[3:5, 3:5, 3:5] = 1          # inside
        lesion[0, 0, 0] = 1                 # outside
        score = compute_coverage(label, lesion)
        assert 0.0 < score < 1.0

    def test_shape_mismatch_raises(self):
        label = torch.zeros(4, 4, 4)
        lesion = torch.zeros(4, 4, 5)
        with pytest.raises(RuntimeError, match="Shape mismatch"):
            compute_coverage(label, lesion)

    def test_score_in_range(self):
        label = torch.randint(0, 3, (10, 10, 10))
        lesion = (torch.rand(10, 10, 10) > 0.7).long()
        score = compute_coverage(label, lesion)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# get_random_position_in_label
# ---------------------------------------------------------------------------

class TestGetRandomPositionInLabel:

    def _volume(self, size=10, target=1):
        t = torch.zeros(size, size, size, dtype=torch.long)
        t[3:7, 3:7, 3:7] = target
        return t

    def test_returns_list_of_three_ints(self):
        pos = get_random_position_in_label(self._volume(), target_labels=[1])
        assert isinstance(pos, list)
        assert len(pos) == 3
        assert all(isinstance(c, int) for c in pos)

    def test_position_has_correct_label(self):
        vol = self._volume()
        pos = get_random_position_in_label(vol, target_labels=[1])
        assert vol[pos[0], pos[1], pos[2]].item() == 1

    def test_int_target_label_accepted(self):
        vol = self._volume()
        pos = get_random_position_in_label(vol, target_labels=1)
        assert vol[pos[0], pos[1], pos[2]].item() == 1

    def test_multiple_target_labels(self):
        vol = self._volume()
        vol[0, 0, 0] = 2
        pos = get_random_position_in_label(vol, target_labels=[1, 2])
        assert vol[pos[0], pos[1], pos[2]].item() in {1, 2}

    def test_missing_label_raises(self):
        vol = self._volume()
        with pytest.raises(ValueError, match="No voxel"):
            get_random_position_in_label(vol, target_labels=[99])


# ---------------------------------------------------------------------------
# get_biggest_component
# ---------------------------------------------------------------------------

class TestGetBiggestComponent:

    def test_returns_only_largest(self):
        t = torch.zeros(10, 10, 10, dtype=torch.bool)
        t[1:4, 1:4, 1:4] = True    # small component (27 voxels)
        t[6:10, 6:10, 6:10] = True # large component (64 voxels)
        result = get_biggest_component(t)
        # Large component voxels must all be True
        assert result[7, 7, 7].item() is True
        # Small component must not be in result
        assert result[2, 2, 2].item() is False

    def test_output_same_shape(self):
        t = torch.zeros(8, 8, 8, dtype=torch.bool)
        t[2:5, 2:5, 2:5] = True
        result = get_biggest_component(t)
        assert result.shape == t.shape

    def test_single_component_returned_fully(self):
        t = torch.zeros(8, 8, 8, dtype=torch.bool)
        t[2:6, 2:6, 2:6] = True
        result = get_biggest_component(t)
        assert result.sum() == t.sum()


# ---------------------------------------------------------------------------
# compute_anatomical_resistance
# ---------------------------------------------------------------------------

class TestComputeAnatomicalResistance:

    def _labels(self, size=10):
        t = torch.zeros(size, size, size, dtype=torch.long)
        t[0:3, :, :] = 1    # wm_label
        t[3:6, :, :] = 10   # gm_label
        t[6:8, :, :] = 32   # csf_label
        return t

    def test_wm_has_low_resistance(self):
        t = self._labels()
        crop = compute_anatomical_resistance(t, wm_label=1, start=(0, 0, 0), end=(3, 3, 3))
        assert (crop == 0.1).all()

    def test_csf_has_lowest_resistance(self):
        t = self._labels()
        crop = compute_anatomical_resistance(t, wm_label=1, start=(6, 0, 0), end=(8, 3, 3))
        assert (crop == 0.01).all()

    def test_gm_resistance(self):
        t = self._labels()
        crop = compute_anatomical_resistance(t, wm_label=1, start=(3, 0, 0), end=(6, 3, 3))
        assert (crop == 0.3).all()

    def test_other_labels_have_resistance_one(self):
        t = torch.zeros(8, 8, 8, dtype=torch.long)
        t[:] = 99  # unknown label
        crop = compute_anatomical_resistance(t, wm_label=1, start=(0, 0, 0), end=(4, 4, 4))
        assert (crop == 1.0).all()

    def test_crop_shape_correct(self):
        t = self._labels()
        crop = compute_anatomical_resistance(t, wm_label=1, start=(0, 0, 0), end=(3, 4, 5))
        assert crop.shape == (3, 4, 5)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])

# test_deform_csf_transform.py
import pytest
from transforms_class.DeformCSFTransform import DeformCSFTransform


def _make(**kwargs):
    defaults = dict(
        displacement=3.0,
        grid_space=10,
        mode="add",
        background_value=0,
        labels_to_target=[1],
        labels_to_keep=[2],
    )
    defaults.update(kwargs)
    return DeformCSFTransform(**defaults)


class TestDeformCSFTransformInit:
    def test_mode_stored(self):
        assert _make(mode="add").mode == "add"

    def test_suffix_contains_mode(self):
        assert "Add" in _make(mode="add").anomaly_suffix

    def test_invalid_retry_raises(self):
        with pytest.raises(ValueError):
            _make(retry_while_deform_same=0)

    def test_list_displacement_stored(self):
        t = _make(displacement=[1.0, 3.0])
        assert isinstance(t.displacement, list)

    def test_labels_to_keep_is_none(self):
        # TODO test the behavior when labels_to_keep is None
        t = _make(labels_to_keep=None)


class TestResolveRandomParameters:
    def _make_list(self):
        return DeformCSFTransform(
            displacement=[1.0, 3.0, 5.0],
            grid_space=[5, 10, 20],
            mode="add",
            background_value=0,
            random_field_mask=["bidirectional", "unidirectional"],
            perlin_unidirectional_threshold=[0.2, 0.3, 0.4],
        )

    def test_displacement_scalar_after_resolve(self):
        t = self._make_list()
        t.resolve_random_parameters()
        assert isinstance(t.displacement, (int, float))

    def test_grid_space_scalar_after_resolve(self):
        t = self._make_list()
        t.resolve_random_parameters()
        assert isinstance(t.grid_space, (int, float))

    def test_random_field_mask_string_after_resolve(self):
        t = self._make_list()
        t.resolve_random_parameters()
        assert isinstance(t.random_field_mask, str)

    def test_perlin_threshold_scalar_after_resolve(self):
        t = self._make_list()
        t.resolve_random_parameters()
        assert isinstance(t.perlin_unidirectional_threshold, (int, float))




if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

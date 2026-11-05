# test_deform_label.py
import pytest
from transforms_class.DeformLabel import DeformLabel


class TestDeformLabelInit:
    def _make(self, **kwargs):
        defaults = dict(
            random_field_mask="bidirectional",
            displacement=3.0,
            grid_space=10,
            mode="add",
            labels_to_target=[1],
            labels_to_keep=[2],
        )
        defaults.update(kwargs)
        return DeformLabel(**defaults)

    def test_init_sets_mode(self):
        t = self._make(mode="add")
        assert t.mode == "add"

    def test_suffix_contains_mode(self):
        t = self._make(mode="add")
        assert "Add" in t.anomaly_suffix

    def test_list_displacement_stored(self):
        t = self._make(displacement=[1.0, 2.0, 3.0])
        assert isinstance(t.displacement, list)

    def test_list_grid_space_stored(self):
        t = self._make(grid_space=[5, 10, 20])
        assert isinstance(t.grid_space, list)


class TestResolveRandomParameters:
    def _make(self):
        return DeformLabel(
            random_field_mask=["bidirectional", "unidirectional"],
            displacement=[1.0, 3.0, 5.0],
            grid_space=[5, 10, 20],
            mode="add",
            nb_deforms=[1, 2, 3],
        )

    def test_resolved_displacement_is_scalar(self):
        t = self._make()
        t.resolve_random_parameters()
        assert isinstance(t.displacement, (int, float))

    def test_resolved_grid_space_is_scalar(self):
        t = self._make()
        t.resolve_random_parameters()
        assert isinstance(t.grid_space, (int, float))

    def test_resolved_random_field_mask_is_string(self):
        t = self._make()
        t.resolve_random_parameters()
        assert isinstance(t.random_field_mask, str)

    def test_resolved_nb_deforms_is_scalar(self):
        t = self._make()
        t.resolve_random_parameters()
        assert isinstance(t.nb_deforms, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

import torch
import torchio as tio
import pytest

from transforms_class.transformPipeline import TransformPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _AppendSuffix(tio.Transform):
    """Appends a token to subject["suffix"] — useful to track execution order."""
    def __init__(self, token: str):
        super().__init__()
        self.token = token

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject["suffix"] = subject.get("suffix", "") + self.token
        return subject


class _CountCalls(tio.Transform):
    """Records how many times it has been called."""
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        self.call_count += 1
        return subject


def _make_subject() -> tio.Subject:
    tensor = torch.zeros(1, 4, 4, 4)
    return tio.Subject(label=tio.LabelMap(tensor=tensor, affine=torch.eye(4)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTransformPipeline:

    # --- Execution -----------------------------------------------------------

    def test_applies_transforms_in_order(self):
        """Transforms must be applied left-to-right."""
        pipeline = TransformPipeline([_AppendSuffix("A"), _AppendSuffix("B"), _AppendSuffix("C")])
        out = pipeline(_make_subject())
        assert out["suffix"] == "ABC"

    def test_single_transform(self):
        pipeline = TransformPipeline([_AppendSuffix("X")])
        out = pipeline(_make_subject())
        assert out["suffix"] == "X"

    def test_empty_pipeline_leaves_subject_unchanged(self):
        subject = _make_subject()
        pipeline = TransformPipeline([])
        out = pipeline(subject)
        assert "suffix" not in out  # no transform touched it

    def test_each_transform_called_exactly_once(self):
        t1, t2 = _CountCalls(), _CountCalls()
        TransformPipeline([t1, t2])(_make_subject())
        assert t1.call_count == 1
        assert t2.call_count == 1

    def test_subject_is_threaded_through_all_transforms(self):
        """Output of transform N must be the input of transform N+1."""
        pipeline = TransformPipeline([
            _AppendSuffix("1"),
            _AppendSuffix("2"),
            _AppendSuffix("3"),
        ])
        out = pipeline(_make_subject())
        # Each step appends in order → "123"
        assert out["suffix"] == "123"

    # --- Representation ------------------------------------------------------

    def test_repr_contains_pipeline_class_name(self):
        pipeline = TransformPipeline([_AppendSuffix("A")])
        assert "TransformPipeline" in repr(pipeline)

    def test_repr_lists_all_transform_class_names(self):
        pipeline = TransformPipeline([_AppendSuffix("A"), _CountCalls()])
        r = repr(pipeline)
        assert "_AppendSuffix" in r
        assert "_CountCalls" in r

    def test_str_equals_repr(self):
        pipeline = TransformPipeline([_AppendSuffix("A"), _AppendSuffix("B")])
        assert str(pipeline) == repr(pipeline)

    def test_print_shows_all_transforms(self, capsys):
        pipeline = TransformPipeline([_AppendSuffix("A"), _AppendSuffix("B")])
        print(pipeline)
        captured = capsys.readouterr()
        assert "_AppendSuffix" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

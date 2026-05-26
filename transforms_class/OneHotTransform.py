# -----------------------------------------------------------------------------
# One-Hot Transform
# -----------------------------------------------------------------------------
from typing import Optional, Sequence
import numpy as np
import torch
import torchio as tio

from scipy.ndimage import (
    label as ndi_label,
    distance_transform_edt,
)


def fill_mask_with_nearest_value(volume: np.ndarray) -> np.ndarray:
    if not isinstance(volume, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray")
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError("Expected shape (1, D, H, W)")

    out = volume.copy()
    labels = volume[0]

    background = labels == 0
    if not np.any(background):
        return out

    # Connected components of background
    labeled_bg, _ = ndi_label(background)

    # Background connected to the border
    border_labels = np.unique(np.concatenate([
        labeled_bg[0].ravel(),
        labeled_bg[-1].ravel(),
        labeled_bg[:, 0].ravel(),
        labeled_bg[:, -1].ravel(),
        labeled_bg[:, :, 0].ravel(),
        labeled_bg[:, :, -1].ravel(),
    ]))
    border_labels = border_labels[border_labels != 0]

    # Holes = background NOT connected to border
    holes = background & (~np.isin(labeled_bg, border_labels))
    if not np.any(holes):
        return out

    # distance to nearest NON-ZERO
    indices = distance_transform_edt(
        labels == 0,
        return_indices=True,
        return_distances=False
    )

    z, y, x = np.where(holes)
    out[0, z, y, x] = labels[
        indices[0][z, y, x],
        indices[1][z, y, x],
        indices[2][z, y, x],
    ]

    return out

class OneHotTransform(tio.OneHot):
    """
    Extend TorchIO's OneHot to allow inversion and hole-filling.
    """
    def __init__(
        self,
        num_classes: int = -1,
        invert_transform: bool = False,
        background_segmented: bool = True,
        exclude: Optional[Sequence[str]] = None,
        fill_holes: bool = False,
        **kwargs
    ):
        """
        Initialize the transform.

        Parameters
        ----------
        num_classes : int
            Number of classes (-1 to infer).
        invert_transform : bool
            If True, invert one-hot back to labels.
        background_segmented : bool
            If False, drop the background channel after one-hot.
        exclude : Optional[Sequence[str]]
            Image names to exclude.
        fill_holes : bool
            If True, fill holes after inversion.
        **kwargs
            Passed to `tio.OneHot`.
        """
        super().__init__(num_classes, exclude=exclude, **kwargs)
        self.background_segmented = background_segmented
        self.invert_transform = invert_transform
        self.fill_holes = fill_holes

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """
        Apply on each image in the subject.

        - If `invert_transform` is False: one-hot encode; optionally drop background.
        - If True: argmax back to labels; optionally fill holes.

        Returns
        -------
        tio.Subject
            The (modified) subject.
        """
        for image in self.get_images(subject):
            if not self.invert_transform:
                # Forward one-hot
                self.one_hot(image)
                if not self.background_segmented:
                    image.set_data(image.data[1:])
            else:
                # Invert back to labels
                self.argmax(image)
                if self.fill_holes:
                    filled = fill_mask_with_nearest_value(image.data.numpy())
                    image.set_data(torch.from_numpy(filled))
        return subject
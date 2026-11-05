import logging
import torch
import torchio as tio
import numpy as np
from scipy.ndimage import center_of_mass
from transforms_class.LabelTransform import LabelTransform
import SimpleITK as sitk

from utils.logger import BLUE, RESET, configure_logging

logger = logging.getLogger(__name__)

class ZoomCenterTransform(LabelTransform):
    """
    Applies a zoom (scaling) transformation centered on the center of mass of a label mask.

    This transform rescales the input label mask by a given ratio, anchoring the zoom at the center of mass
    of the nonzero region. The scaling is performed in physical space using an affine transformation,
    and the result is resampled back into the original image space.

    Parameters
    ----------
    ratio : float
        The scaling factor to apply. Values >1 zoom in (enlarge), values <1 zoom out (shrink).

    verbose_dev : bool, optional
        If True, enables verbose logging for development purposes. Default is True.

    Attributes
    ----------
    ratio : float
        The scaling factor used for the zoom transformation.
    """

    def __init__(self, ratio: float, verbose_dev: bool = True, **kwargs):
        super().__init__(**kwargs)
        configure_logging(dev_mode=verbose_dev, module_name=__name__)

        self.ratio = ratio
        
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        logger.info("%s[ZoomCenterTransform]%s Started", BLUE, RESET)
        for name, image in subject.items():
            if not isinstance(image, tio.LabelMap):
                continue

            data = image.data.numpy()
            assert data.shape[0] == 1, f"Expected single channel label map, got {data.shape[0]} channels"

            if data.sum() == 0:
                raise ValueError(f"Label map '{name}' is empty, cannot zoom.")

            self._zoom_centered(image)

        return subject
    
    @classmethod
    def apply_to_tensor(cls, mask: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Apply a zoom (scaling) transformation to a tensor mask, centered on the center of mass.
    
        This method rescales the input mask by the specified ratio, anchoring the zoom at the center of mass
        of the nonzero region. The scaling is performed in physical space using an affine transformation,
        and the result is returned as a tensor in the original image space.
    
        Parameters
        ----------
        mask : torch.Tensor
            The input binary or label mask tensor. Shape (C, D, H, W) or (D, H, W).
        ratio : float
            The scaling factor to apply. Values >1 zoom in (enlarge), values <1 zoom out (shrink).
    
        Returns
        -------
        torch.Tensor
            The zoomed mask tensor, with the same shape as the input.
        """
        is_3d = mask.dim() == 3
        if is_3d:
            mask = mask.unsqueeze(0)

        subject = tio.Subject(label=tio.LabelMap(tensor=mask))
        result = cls(ratio=ratio)(subject)["label"].data
        
        if is_3d:
            result = result.squeeze(0)
        return result

    def _zoom_centered(self, image: tio.LabelMap) -> None:
        """
        Apply a zoom (scaling) transformation to a label map, centered on the center of mass of the mask.
    
        The function rescales the input label mask by the specified ratio, anchoring the zoom at the center of mass
        of the nonzero region in physical space. The scaling is performed using an affine transformation,
        and the result is resampled back into the original image space.
    
        Parameters
        ----------
        image : tio.LabelMap
            The label map to be zoomed. Must contain a single channel and nonzero voxels.
    
        Returns
        -------
        None
            The input image is modified in place with the zoomed label data.
        """
        logger.info(f"{BLUE}[ZoomCenterTransform]{RESET} ratio = {BLUE}{self.ratio}{RESET}")
        mask = image.data.numpy()[0]
        sitk_image = image.as_sitk()

        center_index = np.array(center_of_mass(mask))
        # Center of mass in physical space
        center_lps = sitk_image.TransformContinuousIndexToPhysicalPoint(center_index.tolist())

        scale_transform = sitk.AffineTransform(3)
        # SetCenter anchors the scaling to the lesion center of mass instead of the image origin
        scale_transform.SetCenter(center_lps)
        # SetMatrix sets the 3x3 linear part of the affine transform
        # We want to scale by self.ratio in physical space, which means we need to scale the coordinates by 1/ratio
        scale_transform.SetMatrix((np.eye(3) / self.ratio).flatten().tolist())

        # Resample into the same image space (same size/spacing/origin)
        zoomed_sitk = sitk.Resample(sitk_image, sitk_image, scale_transform, sitk.sitkNearestNeighbor, 0)
        image.set_data(tio.LabelMap.from_sitk(zoomed_sitk).data)

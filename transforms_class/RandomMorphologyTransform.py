import logging
import torch
import numpy as np
from torchio import Subject
from scipy.ndimage import binary_dilation, generate_binary_structure

from transforms_class.LabelTransform import LabelTransform
from utils.logger import BLUE, RESET, configure_logging

# Configure logging
logger = logging.getLogger(__name__)

class RandomMorphologyTransform(LabelTransform):
    """
    TorchIO transform that applies random dilations/erosions

    Parameters
    ----------
    """

    def __init__(
        self,
        p: float = 1,
        label_to_dilate: str = "",
        label_within: str = "",
        nb_iter_min: int = 1,
        nb_iter_max: int = 2,
        transform_suffix: str = "dil",
        label_within_delete: str = None,
        verbose_dev: bool = True,
        **kwargs
    ):
        super().__init__(p=p, **kwargs)
        configure_logging(dev_mode=verbose_dev, module_name=__name__)


        self.nb_iter_min = nb_iter_min
        self.nb_iter_max = nb_iter_max
        self.transform_suffix = transform_suffix
        self.label_within_delete = label_within_delete
        self.label_to_dilate=label_to_dilate
        self.label_within = label_within
        

    def apply_transform(self, subject: Subject) -> Subject:
        logger.info("%s[RandomMorphologyTransform]%s Applying morphology transform %s%s%s", BLUE, RESET, BLUE, self.transform_suffix, RESET)
        if len(self.get_images(subject)) > 1:
            raise NotImplementedError(f"Morphology transform with more than one labelmap image.")

        if "suffix" in subject:
            transform_suffix = subject["suffix"]
        else:
            transform_suffix = ""
        for label_map in self.get_images(subject):
            data = label_map.data.numpy()

            nb_iter = torch.randint(self.nb_iter_min, self.nb_iter_max, (1,)).item()
            if self.label_within_delete is not None:
                data = self._wm_carve_gm(data,self.label_within_delete, nb_iter)
            else:
                data = self._morphology(
                    data,
                    target_labels=self.label_to_dilate,
                    iterations=nb_iter,
                    within_labels=self.label_within,
                )

            transform_suffix += self.transform_suffix +  f"_It{nb_iter}"

            label_map.set_data(torch.from_numpy(data))
            subject["label"] = label_map
            subject["suffix"] = transform_suffix
        return subject

    @staticmethod
    def _morphology(
        data,
        target_labels,
        iterations=1,
        within_labels=None,
    ):
        for label in target_labels:
            # Prepare boolean mask for dilation
            mask_total = np.isin(data, label)
            new_mask = binary_dilation(
                mask_total, iterations=iterations
            )

            if within_labels is not None:
                within_mask = np.isin(data, within_labels)
                new_mask &= within_mask
                
            # For label in target_labels:
            data[(new_mask) & (~mask_total)] = label
        return data

    def _wm_carve_gm(self, data, replace_label_by, iters):
        if len(self.label_within) != 1:
            raise NotImplementedError(f"Currently only one label_within is supported for wm_carve_gm. got {self.label_within}")
        if len(self.label_to_dilate) != 1:
            raise NotImplementedError(f"Currently only one label_to_dilate is supported for wm_carve_gm. got {self.label_to_dilate}")
        
        gm_mask = np.isin(data, self.label_within)
        wm_mask = np.isin(data, self.label_to_dilate)

        dilated_wm = binary_dilation(
            wm_mask[0], generate_binary_structure(3, 2), iters
        ).astype(int)
        dilated_wm = np.expand_dims(dilated_wm, 0)
        dilated_wm[wm_mask] = 0

        # Temporarily convert GM voxels to CSF to avoid overlap
        data[gm_mask] = replace_label_by

        # Create GM boundary
        gm_boundary = binary_dilation(gm_mask, iterations=2).astype(int)
        new_gm = dilated_wm * gm_boundary
        data[new_gm > 0] = self.label_within[0] # Possible because len(self.label_within) == 1

        return data
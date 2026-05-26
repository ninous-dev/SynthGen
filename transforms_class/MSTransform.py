import logging
import random
import numpy as np
import torch
import torchio as tio
from typing import Tuple, Union, List

from transforms_class.AnomalyLabelTransform import AnomalyLabelTransform
from transforms_class.utils.lesions_tool_box import (
        get_random_position_in_label
    )
from transforms_class.utils.perlin_utils import random_perlin_threshold
from utils.logger import BLUE, GREEN, ORANGE, RESET

logger = logging.getLogger(__name__)

class MSTransform(AnomalyLabelTransform):
    """
    MS-like lesions synthesis in a target label.

    Parameters
    ----------
    labels_to_target : Union[int, List[int]]
        Labels in which lesions will be inserted.

    nb_sclerosis : Union[int, Tuple[int]]
        The number of sclerosis to add. If an int, it will be the same for both min and max.
        If a tuple, it should be (min_nb, max_nb).

    lesion_label : int
        Output label value for lesions.

    background_value : int
        The value of the background in the label tensor, used for generating the lesion shape.

    sclerosis_size : Tuple[int], optional, default=None
        The size of the sclerosis to add. If None, sclerosis will not be constrained by size.
        If a tuple, it should be (min_size, max_size).

    gradient : bool, optional, default=False
        Whether to use a gradient for the sclerosis shape.

    random_field_mask : str, optional, default=None
        The type of random field mask to use for the sclerosis shape. Can be "bidirectional", "unidirectional", or None.

    perlin_contour_proba : float, optional, default=0.9
        The probability of using Perlin noise for the contour of the sclerosis shape.

    random_sizes : bool, optional, default=True

    **kwargs
        Additional keyword arguments for the parent class.
    """

    def __init__(
        self,
        labels_to_target: Union[int, List[int]],
        nb_sclerosis: Union[int, Tuple[int]],
        lesion_label: int,
        background_value: int,
        sclerosis_size: Tuple[int] = None,
        gradient: bool = False,
        random_field_mask: str = None,
        perlin_contour_proba: float = 0.9,
        random_sizes: bool = True,
        **kwargs
    ):
        suffix = "MS"
        super().__init__(anomaly_suffix=suffix, **kwargs)

        self.labels_to_target = AnomalyLabelTransform.int_to_list(labels_to_target)
        self.nb_sclerosis = AnomalyLabelTransform.int_to_tuple(nb_sclerosis)

        self.sclerosis_min_size, self.sclerosis_max_size = sclerosis_size if sclerosis_size is not None else (None, None)

        self.lesion_label = lesion_label
        self.gradient = gradient
        self.background_value = background_value
        self.random_field_mask = random_field_mask
        self.perlin_contour_proba = perlin_contour_proba
        self.random_sizes = random_sizes

        self.args_names = [
            "labels_to_target",
            "nb_sclerosis",
            "lesion_label",
            "background_value"
        ]

   
    def create_random_elliptical(
            self,
            cropped_tensor: Union[torch.Tensor, np.ndarray]
        ) -> np.ndarray:
        """
        Create a random elliptical array with the same shape as `cropped_tensor`.

        Parameters
        ----------
        cropped_tensor : torch.Tensor | np.ndarray
            Input volume; only its shape is used.

        Returns
        -------
        np.ndarray
            Array with the same shape as `cropped_tensor`.
        """
        # Generate elliptical shape
        center, radii = AnomalyLabelTransform.generate_radii_from_shape(cropped_tensor.shape)
        elliptical = AnomalyLabelTransform.generate_elliptical(cropped_tensor.shape, center, radii, self.gradient)
        mask = (elliptical.data > 0)
        try :
            mask = AnomalyLabelTransform.deform_mask(
                random_field_mask=self.random_field_mask,
                mask=mask.unsqueeze(0),
                displacement=3,
            ).squeeze(0)
        except ValueError as e:
            if "Deformation resulted in identical image data" in str(e):
                logger.warning("%s%s%s", ORANGE, e, RESET)
            else:
                raise e

        # Add Perlin noise-based contours
        if self.perlin_contour_proba > 0 and random.random() < self.perlin_contour_proba:
            mask = self.apply_perlin_contours(mask.bool())

        # Add gradient-based shape
        if self.gradient:
            return self.add_weighted_noise(mask, noise_level=0.15) + mask.float() * self.lesion_label

        return torch.where(
            mask.bool() & torch.isin(cropped_tensor, torch.tensor(self.labels_to_target)),
            self.lesion_label,
            cropped_tensor
        )

    def apply_perlin_contours(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply Perlin noise-based contours to the given mask.

        Parameters
        ----------
        mask : torch.Tensor
            The binary mask to which Perlin noise-based contours will be applied.

        Returns
        -------
        torch.Tensor
            The updated mask with Perlin noise-based contours applied.
        """
        scale = 20
        max_growth = 2
        dilation = 5
        lacunarity = 3

        # Get binary base mask from elliptical shape
        mask_np = mask.cpu().numpy()

        # Generate irregular Perlin-based contours
        perlin_threshold_contour = random_perlin_threshold()
        contours_mask = self.create_irregular_contours_perlin(
            lesion_mask=mask_np,
            max_growth=max_growth,
            scale=scale,
            perlin_threshold=perlin_threshold_contour,
            dilation=random.randint(1, dilation),
            lacunarity=lacunarity
        )

        # Combine Perlin contour + lesion
        return torch.from_numpy((mask_np | contours_mask).astype(int))

    def add_weighted_noise(self, elliptical_tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
        """
        Add random noise weighted by `elliptical_tensor`.

        Parameters
        ----------
        elliptical_tensor : torch.Tensor
            Weight map (same shape as output); higher values → more noise.
        noise_level : float
            Global noise amplitude.

        Returns
        -------
        torch.Tensor
            Tensor of same shape as `elliptical_tensor`, with weighted noise added.
        """
        noise = torch.randn_like(elliptical_tensor) * noise_level
        weight = (1 - elliptical_tensor).clamp(min=0.0, max=1.0)

        elipse_noisy = (noise * weight) + elliptical_tensor

        return elipse_noisy.clamp(0.0, 1.0)
    
    @AnomalyLabelTransform.update_subject_anomaly_and_log_success
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """
        Apply the transform to a TorchIO subject.

        Parameters
        ----------
        subject : tio.Subject
            Input subject.

        Returns
        -------
        tio.Subject
            The transformed subject (input modified in place and returned).
        """
        if isinstance(self.lesion_label, list) and len(self.lesion_label) == 1:
            self.lesion_label = self.lesion_label[0]
        if not isinstance(self.lesion_label, int):
            raise ValueError(f"lesion_label must be an integer. Got {type(self.lesion_label)} with value {self.lesion_label}.")

        logger.info(f"{BLUE}[MS]{RESET} Lesion label = {GREEN}{self.lesion_label}{RESET}")

        nb_sclerosis = random.randint(*self.nb_sclerosis) # nosec B311
        min_size, max_size = float('inf'), 0
        
        for image_name, tio_image in self.get_images_dict(subject).items():
            if not isinstance(tio_image, tio.LabelMap):
                continue
            if image_name == "__label_without_lesion":
                continue

            if "__label_without_lesion" not in subject:
                without_lesion_label_image = tio.LabelMap(
                    tensor=tio_image.data,
                    affine=tio_image.affine)  
                subject.add_image(without_lesion_label_image, "__label_without_lesion")
                logger.info(f"{BLUE}[MS]{RESET} Adding __label_without_lession in subject ")

            logger.info(f"{BLUE}[MS]{RESET} Adding {BLUE}{nb_sclerosis}{RESET} sclerosis")
            data_tensor = tio_image.data


            nb_cur_sclerosis = 0
            while nb_cur_sclerosis != nb_sclerosis:
                # Select a random position in the target label and crop the corresponding box
                point_on_skeleton = get_random_position_in_label(data_tensor.squeeze(), self.labels_to_target)
                cropped_label_tensor, start_indices, end_indices = AnomalyLabelTransform.crop_pure_target_box(
                    data_tensor,
                    point_on_skeleton,
                    self.sclerosis_min_size,
                    self.sclerosis_max_size,
                    self.labels_to_target,
                    random_max_sizes=self.random_sizes
                )

                # Skip if the cropped region is too small
                if self.sclerosis_min_size and (min(cropped_label_tensor.shape) < self.sclerosis_min_size):
                    continue

                # Log size of sclerosis (cropped_label_tensor)
                min_size = min(min_size, min(cropped_label_tensor.shape))
                max_size = max(max_size, max(cropped_label_tensor.shape))
                logger.info(f"{BLUE}[MS]{RESET} Sclerosis size: {GREEN}{cropped_label_tensor.shape}{RESET}")

                # Generate and insert the lesion
                elliptical_noisy_deform = self.create_random_elliptical(cropped_label_tensor).squeeze()
                data_tensor = AnomalyLabelTransform.put_cropped_patch_back(
                    data_tensor.squeeze(),
                    cropped_patch=elliptical_noisy_deform,
                    start_indices=start_indices,
                    end_indices=end_indices
                )

                nb_cur_sclerosis += 1

            tio_image.set_data(data_tensor)

        subject["min_size"] = min_size
        subject["max_size"] = max_size
        subject["nb_sclerosis"] = nb_sclerosis
        
        self.add_labels_names(subject, {"MS_lesion": self.lesion_label})
        
        
        return subject


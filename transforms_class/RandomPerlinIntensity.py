import logging
import random
from typing import List, Tuple, Union
import numpy as np

import torch
import torchio as tio
from transforms_class.LabelTransform import LabelTransform
from transforms_class.utils.perlin_utils import generate_perlin_map
from utils.logger import BLUE, RED, RESET, configure_logging
import SimpleITK as sitk
from utils.random import random_probability

logger = logging.getLogger(__name__)

class RandomPerlinIntensity(tio.IntensityTransform):
    def __init__(
        self,
        label_image_name: str,
        labels_to_target: List[int],
        scale: float,
        alpha: Union[float, List[float]] = 0.8,
        octaves: int = 4,
        lacunarity: float = 2.0,
        threshold: Union[float, List[float]] = None,
        p_gradient: float = 0.0,
        verbose_dev: bool = False,
        seed: int = None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        
        configure_logging(dev_mode=verbose_dev, module_name=__name__)

        self.labels_to_target = labels_to_target
        self.scale = scale
        self.alpha = alpha
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.threshold = threshold
        self.label_image_name = label_image_name
        self.p_gradient = p_gradient
        self.seed = seed
        
        self.args_names = [
            "labels_to_target",
            "label_image_name",
            "scale",
            "alpha",
            "octaves",
            "lacunarity",
            "threshold",
            "p_gradient"
        ]
        
        self.anatomy_args = [
            "labels_to_target",
        ]

    def create_gradient_mask_from_tensor(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        mask = mask_tensor.squeeze(0)
        mask_array = mask.detach().cpu().numpy().astype(np.uint8)

        sitk_mask = sitk.GetImageFromArray(mask_array)
        sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)
        
        # Compute the signed Maurer distance map
        distance_map = sitk.SignedMaurerDistanceMap(
            sitk_mask,
            insideIsPositive=True,
            squaredDistance=False,
            useImageSpacing=True,
        )

        distance_array = sitk.GetArrayFromImage(distance_map)
        
        # Set the distance to 0 for all voxels outside the mask
        distance_array[mask_array == 0] = 0

        # Normalize to [0, 1] using the maximum distance
        max_distance = distance_array.max()
        if max_distance > 0:
            normalized_distance_map = distance_array / max_distance
        else:
            normalized_distance_map = distance_array
        
        # Convert back to Torch tensor
        gradient_mask_tensor = torch.from_numpy(normalized_distance_map).float()
        
        gradient_mask_tensor = tio.RandomBlur(std=0.5)(tio.ScalarImage(tensor=gradient_mask_tensor.unsqueeze(0))).data.squeeze(0)
        # min max normalization of the gradient mask
        gradient_mask_tensor = (gradient_mask_tensor - gradient_mask_tensor.min()) / (gradient_mask_tensor.max() - gradient_mask_tensor.min())

        return gradient_mask_tensor

    def resolve_anatomy(self, name_map: dict[str, int]) -> None:
        """
        Resolve the anatomy labels to their corresponding integer IDs.

        Parameters
        ----------
        name_map : dict[str, int]
            A mapping from string labels to integer IDs.
        """
        for anatomy_arg in self.anatomy_args:
            if hasattr(self, anatomy_arg) and getattr(self, anatomy_arg) is not None:
                current_value = getattr(self, anatomy_arg)

                # Keep track of labels_for_edema mapping ONLY if instance is TumorTransform
                if anatomy_arg == "labels_for_edema" and type(self).__name__ == "TumorTransform":
                    if not hasattr(self, "original_edema_labels"):
                        setattr(self, "original_edema_labels", {})

                    original_dict = getattr(self, "original_edema_labels")
                    
                    # First time only: create keys with original items and resolve them
                    if not original_dict and isinstance(current_value, list):
                        for item in current_value:
                            resolved = LabelTransform.resolve_value(item, name_map)
                            original_dict[str(item)] = resolved
                    
                    # Not first time: resolve the existing values again
                    elif original_dict:
                        for key in original_dict.keys():
                            current_resolved = original_dict[key]
                            new_resolved = LabelTransform.resolve_value(current_resolved, name_map)
                            original_dict[key] = new_resolved
                
                # Resolve the value
                setattr(self, anatomy_arg, LabelTransform.resolve_value(
                    getattr(self, anatomy_arg), name_map
                ))

    def resolve_random_value(self, value: Union[float, Tuple[float, float]]) -> float:
        if value is None:
            return None
        if isinstance(value, (tuple, list)):
            value = round(random.uniform(value[0], value[1]), 2)
            return value

        if isinstance(value, (float, int)):
            return value

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if self.labels_to_target is None or not any(label in torch.unique(subject[self.label_image_name].data) for label in self.labels_to_target):
            logger.info(f"Skipping RandomPerlinIntensity as none of the labels {self.labels_to_target} are present in the label image {self.label_image_name}.")
            return subject
        
        alpha = self.resolve_random_value(self.alpha)
        threshold = self.resolve_random_value(self.threshold)
        p_gradient = random_probability(self.p_gradient)
        logger.info(f"{BLUE}[RandomPerlinIntensity]{RESET} alpha = {BLUE}{alpha}{RESET}")
        logger.info(f"{BLUE}[RandomPerlinIntensity]{RESET} threshold = {BLUE}{threshold}{RESET}")
        logger.info(f"{BLUE}[RandomPerlinIntensity]{RESET} p_gradient = {BLUE}{p_gradient}{RESET}")

        for image_name, image in self.get_images_dict(subject).items():
            if not isinstance(image, tio.ScalarImage):
                continue
            logger.info(f"{BLUE}[RandomPerlinIntensity]{RESET} Apply on RandomPerlinIntensity to image: {image_name} with labels: {self.labels_to_target}")
            spatial_shape = tuple(image.data.shape[1:])

            device = image.data.device
            label_data = subject[self.label_image_name].data
            for label in self.labels_to_target:
                mask = label_data == label
                if not mask.any():
                    continue

                base = random.randint(0, 100000) if self.seed is None else self.seed
                perlin = torch.from_numpy(generate_perlin_map(
                    spatial_shape,
                    scale=self.scale,
                    octaves=self.octaves,
                    lacunarity=self.lacunarity,
                    base=base,
                    threshold=threshold,
                    normalize=True
                )).float().to(device, non_blocking=True)

                if p_gradient:
                    gradient_mask = self.create_gradient_mask_from_tensor(mask.float())
                    perlin *= gradient_mask

                mask_3d = mask.squeeze(0)
                perlin_values = perlin[mask_3d]
                perlin_values -= perlin_values.mean()
                perlin_values *= alpha

                updated = (image.data[mask] + perlin_values).clamp_(0, 1)
                image.data.masked_scatter_(mask, updated)

        return subject
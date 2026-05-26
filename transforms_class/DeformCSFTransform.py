import logging
from typing import List, Sequence, Union
import torchio as tio
import torch
import random
from scipy.ndimage import center_of_mass

from transforms_class.AnomalyLabelTransform import AnomalyLabelTransform
from utils.logger import BLUE, GREEN, RED, RESET, YELLOW

logger = logging.getLogger(__name__)

class DeformCSFTransform(AnomalyLabelTransform):
    def __init__(
        self,
        displacement: Union[float, List[float]],
        grid_space: Union[int, List[int]],
        mode: str,
        background_value: int,
        locked_borders: int = 3,
        labels_to_target: List[Union[int, str]] = None,
        labels_to_keep: List[Union[int, str]] = None,
        perlin_unidirectional_threshold: Union[float, List[float]] = 0.25,
        retry_while_deform_same: int = 1,
        random_field_mask: Union[str, List[str]] = "perlin_unidirectional_thresholded",
        **kwargs
    ):
        suffix = f"DeformCSF{mode.capitalize()}"
        super().__init__(anomaly_suffix=suffix, **kwargs)

        # Initialize attributes
        self.mode = mode

        self.random_field_mask = random_field_mask
        self.labels_to_target = labels_to_target
        self.labels_to_keep = labels_to_keep
        self.displacement = displacement
        self.grid_space = grid_space
        
        self.locked_borders = locked_borders
        self.background_value = background_value
        self.perlin_unidirectional_threshold = perlin_unidirectional_threshold
        self.try_deform = retry_while_deform_same

        if self.try_deform < 1:
            raise ValueError(f"retry_while_deform_same must be a positive integer got {self.try_deform}.")

        self.args_names = [
            "random_field_mask",
            "displacement",
            "grid_space",
            "mode",
            "background_value",
            "labels_to_keep",
            "labels_to_target"
        ]
    
    def resolve_random_parameters(self):
        """
        Resolve and assign random parameters for grid space, Perlin noise scale, octaves, threshold, and lacunarity.
    
        Returns
        -------
        dict
            A dictionary containing resolved random values for the parameters.
        """
        random_params = {
            'displacement': AnomalyLabelTransform.choose_random_param(self.displacement, list_choice=True),
            'grid_space': AnomalyLabelTransform.choose_random_param(self.grid_space, list_choice=True),
            'random_field_mask': AnomalyLabelTransform.choose_random_param(self.random_field_mask, list_choice=True),
            'perlin_unidirectional_threshold': AnomalyLabelTransform.choose_random_param(self.perlin_unidirectional_threshold, list_choice=True)
        }
    
        # Assign directly to self
        self.displacement = random_params['displacement']
        self.grid_space = random_params['grid_space']

        self.random_field_mask = random_params['random_field_mask']
        self.perlin_unidirectional_threshold = random_params['perlin_unidirectional_threshold']
    
        # log each parameter value
        for param_name, param_value in random_params.items():
            logger.info(f"{BLUE}[DEFORMLABEL]{RESET} {param_name} = {BLUE}{param_value}{RESET}")
    
    @AnomalyLabelTransform.update_subject_anomaly_and_log_success
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        self.check_anatomy_resolved()
        self.check_unique_background_value()
        self.resolve_random_parameters()
        
        for _, label_image in self.get_images_dict(subject).items():
            
            keep_mask = torch.isin(
                    label_image.data.squeeze(0),
                    torch.tensor(self.labels_to_keep)
            ).unsqueeze(0)
            label_image_data_without_keep = torch.where(
                keep_mask,
            self.background_value, 
            label_image.data)

            logger.info(f"{BLUE}[DeformCSFTransform]{RESET} displacement = {BLUE}{self.displacement}{RESET}")
            logger.info(f"{BLUE}[DeformCSFTransform]{RESET} grid_space = {BLUE}{self.grid_space}{RESET}")
            logger.info(f"{BLUE}[DeformCSFTransform]{RESET} perlin_unidirectional_threshold = {BLUE}{self.perlin_unidirectional_threshold}{RESET}")
            logger.info(f"{BLUE}[DeformCSFTransform]{RESET} random_field_mask = {BLUE}{self.random_field_mask}{RESET}")
            logger.info(f"{BLUE}[DeformCSFTransform]{RESET} background_value = {BLUE}{self.background_value}{RESET}")
            logger.info(f"{BLUE}[DeformCSFTransform]{RESET} label_to_target = {BLUE}{self.labels_to_target}{RESET}")
            logger.info(f"{BLUE}[DeformCSFTransform]{RESET} label_to_keep = {BLUE}{self.labels_to_keep}{RESET}")

            target_point = center_of_mass(torch.isin(label_image_data_without_keep.squeeze(0), torch.tensor(self.labels_to_target)))
            target_point = [int(round(coord)) for coord in target_point]

            for i in range(self.try_deform):
                try:
                    if self.try_deform > 1:
                        logger.info(f"{BLUE}[DeformCSFTransform]{RESET} Attempt {GREEN}{i+1}/{self.try_deform}{RESET} to deform...")

                    label_image_data = AnomalyLabelTransform.deform_mask(
                        random_field_mask=self.random_field_mask,
                        mask=label_image_data_without_keep.clone(),
                        mode=self.mode,
                        displacement=self.displacement,
                        max_grid_space=self.grid_space,
                        labels_to_keep=None,
                        labels_to_target=self.labels_to_target,
                        target_point=target_point,
                        keep_dilation=0,
                        background_value=self.background_value,
                        locked_borders=3,
                        perlin_unidirectional_threshold=self.perlin_unidirectional_threshold
                    )
                    break
                except Exception as e:
                    if "Deformation resulted in identical image data" in str(e) and i < self.try_deform - 1:
                        logger.warning(f"{YELLOW}Deformation resulted in identical image data. Retrying... (Attempt {i+1}/{self.try_deform}){RESET}")
                        continue
                    else:
                        logger.error(f"{RED}Unexpected error during deformation: {e}{RESET}")
                        raise e
            
            label_image_keep_only = torch.where(torch.isin(label_image.data.squeeze(0), torch.tensor(self.labels_to_keep)).unsqueeze(0), label_image.data, -1)
            
            # Put back the labels to keep in the deformed image
            label_image_data = torch.where(label_image_keep_only != -1, label_image_keep_only, label_image_data)
            
            label_image.set_data(label_image_data)

        return subject

        
    def get_mask(
        self,
        tensor: torch.Tensor,
        indices: Sequence[int]
    ) -> torch.Tensor:
        """
        Return a boolean mask where `tensor` equals any value in `indices`.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor (any shape).

        indices : Sequence[int]
            Label values to include.

        Returns
        -------
        torch.Tensor
            Bool mask with the same shape as `tensor`.
        """
        return torch.isin(tensor, torch.tensor(indices, device=tensor.device))
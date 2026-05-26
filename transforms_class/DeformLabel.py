import logging
from typing import List, Union
import torchio as tio

from transforms_class.AnomalyLabelTransform import AnomalyLabelTransform
from utils.logger import BLUE, GREEN, RESET

logger = logging.getLogger(__name__)

class DeformLabel(AnomalyLabelTransform):
    def __init__(
        self,
        random_field_mask: Union[str, List[str]],
        displacement: Union[float, List[float]],
        grid_space: Union[int, List[int]],
        mode: str,
        labels_to_target: List[Union[int, str]] = None,
        labels_to_keep: List[Union[int, str]] = None,
        keep_dilation: int = 0,
        perlin_unidirectional_threshold: Union[float, List[float]] = 0.25,
        locked_borders: int = 3,
        background_value: int = 0,
        nb_deforms: Union[int, List[int]] = 1,
        **kwargs
    ):
        suffix = f"DefLab{mode.capitalize()}"
        super().__init__(anomaly_suffix=suffix, **kwargs)

        # Initialize attributes
        self.mode = mode

        self.random_field_mask = random_field_mask
        self.labels_to_target = labels_to_target
        self.labels_to_keep = labels_to_keep
        self.displacement = displacement
        self.grid_space = grid_space
        self.keep_dilation = keep_dilation
        self.perlin_unidirectional_threshold = perlin_unidirectional_threshold
        self.locked_borders = locked_borders
        self.background_value = background_value
        self.nb_deforms = nb_deforms
        
        self.args_names = [
            "random_field_mask",
            "displacement",
            "grid_space",
            "mode"
        ]

    def resolve_random_parameters(self):
        """
        Resolve random parameters for grid space, Perlin noise scale, octaves, threshold, and lacunarity.

        Returns
        -------
        dict
            A dictionary containing resolved random values for the parameters.
        """
        random_params = {
            'nb_deforms': AnomalyLabelTransform.choose_random_param(self.nb_deforms, list_choice=True),
            'displacement': AnomalyLabelTransform.choose_random_param(self.displacement, list_choice=True),
            'grid_space': AnomalyLabelTransform.choose_random_param(self.grid_space, list_choice=True),
            'perlin_unidirectional_threshold': AnomalyLabelTransform.choose_random_param(self.perlin_unidirectional_threshold, list_choice=True),
            'random_field_mask': AnomalyLabelTransform.choose_random_param(self.random_field_mask, list_choice=True),
        }

        # Assign directly to self
        self.nb_deforms = random_params['nb_deforms']
        self.displacement = random_params['displacement']
        self.grid_space = random_params['grid_space']
        self.perlin_unidirectional_threshold = random_params['perlin_unidirectional_threshold']
        self.random_field_mask = random_params['random_field_mask']

        # log each parameter value
        for param_name, param_value in random_params.items():
            logger.info(f"{BLUE}[DEFORMLABEL]{RESET} {param_name} = {BLUE}{param_value}{RESET}")

        return random_params

    @AnomalyLabelTransform.update_subject_anomaly_and_log_success
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        self.check_anatomy_resolved()
        
        # Resolve random parameters
        random_params = self.resolve_random_parameters()
        self.check_unique_background_value()
        
        for _, tio_image in self.get_images_dict(subject).items():
            label_image_data = tio_image.data
            for deform_i in range(random_params['nb_deforms']):
                logger.info(f"{BLUE}[DEFORMLABEL]{RESET} Applying deformation {GREEN}{deform_i + 1}/{random_params['nb_deforms']}{RESET}")
                label_image_data = AnomalyLabelTransform.deform_mask(
                    random_field_mask=random_params['random_field_mask'],
                    mask=label_image_data,
                    mode=self.mode,
                    displacement=random_params['displacement'],
                    max_grid_space=random_params['grid_space'],
                    labels_to_keep=self.labels_to_keep,
                    labels_to_target=self.labels_to_target,
                    keep_dilation=self.keep_dilation,
                    background_value=self.background_value,
                    locked_borders=self.locked_borders,
                    perlin_unidirectional_threshold=random_params['perlin_unidirectional_threshold']
                )
            
            tio_image.set_data(label_image_data)

        return subject
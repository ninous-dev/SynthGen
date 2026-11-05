import logging
from typing import Tuple, Union
import torch
import torchio as tio

# Internal imports
from transforms_class.AnomalyLabelTransform import AnomalyLabelTransform

from utils.logger import BLUE, GREEN, RESET, configure_logging

logger = logging.getLogger(__name__)


class PeriventricularMS(AnomalyLabelTransform):
    """
    PeriventricularMS generates synthetic periventricular MS

    Parameters
    ----------
    ventricular_label : int
        Label value for ventricles.

    grid_space : Union[int, List[int]]
        Grid space for deformation transform. If a list, a random value is chosen in the list.
        
    displacement : Union[float, Tuple[float, float]]
        Displacement for deformation transform. If a tuple, a random value is chosen in the list.

    labels_to_target : Tuple[int]
        Label values for target lesions.

    lesion_label : int, optional, default=1
        Value to assign to lesions.

    perlin_scale : Union[int, Tuple[int, int]], optional, default=20
        Scale for Perlin noise, controls the size of the pattern. If a tuple, a random value is chosen in the range.

    perlin_octaves : Union[int, Tuple[int, int]], optional, default=4
        Number of Perlin noise octaves, controls the level of detail. If a tuple, a random value is chosen in the range.

    perlin_threshold : Union[float, Tuple[float, float]], optional, default=0.5
        Threshold for Perlin noise [0, 1], controls lesion discreteness. If a tuple, a random value is chosen in the range.

    perlin_lacunarity : Union[float, Tuple[float, float]], optional, default=2.0
        Lacunarity for Perlin noise, controls frequency multiplier between octaves. If a tuple, a random value is chosen in the range.

    random_field_mask : str, optional, default="bidirectional"
        Type of random field mask to use for deformation.
    
    perlin_unidirectional_threshold_deform : float, optional, default=0.25
        Threshold for unidirectional Perlin noise used in deformation, controls directionality of deformation.
    
    perlin_threshold_deform : float, optional, default=0.25
        Threshold for applying deformation based on Perlin noise, controls how much of the mask is deformed.

    verbose_dev : bool, optional
        If True, enables verbose logging for development purposes. Default is True.
    """    
    def __init__(
        self,
        ventricular_label: int,
        grid_space: Union[int, Tuple[int, int]],
        labels_to_target: Tuple[int],
        lesion_label: int = 1,
        displacement: Union[float, Tuple[float, float]] = 1,
        perlin_scale: Union[int, Tuple[int, int]] = 20,
        perlin_octaves: Union[int, Tuple[int, int]] = 4,
        perlin_threshold: Union[float, Tuple[float, float]] = 0.5,
        perlin_lacunarity: Union[float, Tuple[float, float]] = 2.0,
        random_field_mask="bidirectional",
        perlin_threshold_deform: float = 0.5,
        verbose_dev: bool = True,
        **kwargs
    ):
        suffix = "PeriVent"
        super().__init__(anomaly_suffix=suffix, **kwargs)
        configure_logging(dev_mode=verbose_dev, module_name=__name__)

        
        self.ventricular_label = ventricular_label
        self.labels_to_target = labels_to_target
        self.lesion_label = lesion_label
        self.validate_perlin_threshold(perlin_threshold)
        
        self.grid_space = grid_space
        self.perlin_scale = perlin_scale
        self.perlin_octaves = perlin_octaves
        self.perlin_threshold = perlin_threshold
        self.perlin_lacunarity = perlin_lacunarity
        self.displacement = displacement
        self.random_field_mask = random_field_mask
        self.perlin_threshold_deform = perlin_threshold_deform


        self.args_names = [
            'ventricular_label',
            'grid_space', 
            'labels_to_target',
            'perlin_scale',
            'perlin_octaves',
            'perlin_threshold',
            'perlin_lacunarity',
            'displacement',
            'random_field_mask',
            'perlin_threshold_deform',
        ]
    
    @staticmethod
    def validate_perlin_threshold(perlin_threshold):
        """
        Validate that the Perlin threshold is between 0 and 1.

        Parameters
        ----------
        perlin_threshold : Union[float, Tuple[float, float]]
            The Perlin threshold value or range to validate.

        Raises
        ------
        ValueError
            If the Perlin threshold is not within the range [0, 1].
        """
        if isinstance(perlin_threshold, tuple):
            if perlin_threshold[0] < 0 or perlin_threshold[1] > 1:
                raise ValueError("Perlin threshold values must be between 0 and 1.")
        elif isinstance(perlin_threshold, float):
            if perlin_threshold < 0 or perlin_threshold > 1:
                raise ValueError("Perlin threshold value must be between 0 and 1.")

    def __resolve_random_parameters(self):
        """
        Resolve random parameters for grid space, Perlin noise scale, octaves, threshold, and lacunarity.

        Returns
        -------
        dict
            A dictionary containing resolved random values for the parameters.
        """
        random_params = {
            'grid_space': AnomalyLabelTransform.choose_random_param(self.grid_space, list_choice=True),
            'displacement': AnomalyLabelTransform.choose_random_param(self.displacement, list_choice=True),
            'perlin_scale': AnomalyLabelTransform.choose_random_param(self.perlin_scale),
            'perlin_octaves': AnomalyLabelTransform.choose_random_param(self.perlin_octaves),
            'perlin_threshold': AnomalyLabelTransform.choose_random_param(self.perlin_threshold),
            'perlin_lacunarity': AnomalyLabelTransform.choose_random_param(self.perlin_lacunarity),
        }

        self.grid_space = random_params['grid_space']
        self.displacement = random_params['displacement']
        self.perlin_scale = random_params['perlin_scale']
        self.perlin_octaves = random_params['perlin_octaves']
        self.perlin_threshold = random_params['perlin_threshold']
        self.perlin_lacunarity = random_params['perlin_lacunarity']
        
        # log each parameter value
        for param_name, param_value in random_params.items():
            logger.info(f"{BLUE}[PERIVENTRICULAR_MS]{RESET} {param_name} = {BLUE}{param_value}{RESET}")

        return random_params

    def __validate_labels(self):
        """
        Validate lesion_label and ventricular_label attributes.
    
        Ensures lesion_label is an integer (or a list of one integer), and ventricular_label is set.
        Raises ValueError if the checks fail.
        """
        if isinstance(self.lesion_label, list) and len(self.lesion_label) == 1:
            self.lesion_label = self.lesion_label[0]
        if not isinstance(self.lesion_label, int):
            raise ValueError(f"lesion_label must be an integer. Got {type(self.lesion_label)} with value {self.lesion_label}.")
        if self.ventricular_label is None:
            raise ValueError("Ventricular label must be specified.")
        
        logger.info(f"{BLUE}[MS]{RESET} Lesion label = {GREEN}{self.lesion_label}{RESET}")
        
        
    @AnomalyLabelTransform.update_subject_anomaly_and_log_success
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """
        Apply PeriventricularMS lesion synthesis.
        
        Parameters
        ----------
        subject : tio.Subject
            The input subject containing the images and labels to transform.
            
        Returns
        -------
        tio.Subject
            The transformed subject with synthetic periventricular MS lesions.
        """
        logger.info("%s[PeriventricularMS]%s Started", BLUE, RESET)

        # Validate labels and resolve random parameters before processing
        self.__validate_labels()
        self.__resolve_random_parameters()

        logger.info(f"{GREEN}=== Starting PeriventricularMS Transform ==={RESET}")
        for img_name, img in self.get_images_dict(subject).items():
            if not isinstance(img, tio.LabelMap):
                continue
            if img_name == "__label_without_lesion":
                continue

            if "__label_without_lesion" not in subject:
                without_lesion_label_image = tio.LabelMap(
                    tensor=img.data,
                    affine=img.affine)
                subject.add_image(without_lesion_label_image, "__label_without_lesion")
                logger.info(f"{BLUE}[PERIVENTRICULAR_MS]{RESET} Adding __label_without_lession in subject")

            logger.info(f"{BLUE}[PERIVENTRICULAR_MS]{RESET} Processing image: {BLUE}{img_name}{RESET}")

            # Check if ventricular label exists
            if not torch.isin(
                img.data,
                torch.tensor(self.ventricular_label, device=img.data.device)
            ).any():
                logger.warning(
                    f"Ventricular label {self.ventricular_label} not found in {img_name}. Skipping."
                )
                continue

            # Extract ventricular mask
            ventricular_mask = torch.isin(
                img.data,
                torch.tensor(self.ventricular_label, device=img.data.device)
            ).float()  # (1, D, H, W)

            # Get minimal bounding box
            bbox = PeriventricularMS.get_bounding_box(ventricular_mask, min(img.spacing))

            # Crop region
            ventricules_crop = PeriventricularMS.crop_with_bbox(ventricular_mask, bbox)

            # Create LabelMap from cropped region
            ventricules_labelmap = tio.LabelMap(tensor=ventricules_crop, affine=img.affine)

            # Apply deformation transforms
            ventricules_labelmap_data = AnomalyLabelTransform.deform_mask(
                random_field_mask=self.random_field_mask,
                mask=ventricules_labelmap.data,
                mode="push",
                displacement=self.displacement,
                max_grid_space=self.grid_space,
                labels_to_keep=self.ventricular_label,
                labels_to_target=[1],
                perlin_unidirectional_threshold=self.perlin_threshold_deform
            )

            # Apply Perlin noise
            perlin_lesion = PeriventricularMS.apply_perlin_on_mask(
                mask=ventricules_labelmap_data,
                octaves=self.perlin_octaves,
                perlin_threshold=self.perlin_threshold,
                perlin_scale=self.perlin_scale,
                lacunarity=self.perlin_lacunarity
            )
            perlin_lesion = torch.from_numpy(perlin_lesion).bool()

            logger.info(f"{BLUE}[PERIVENTRICULAR_MS]{RESET} insert mask of size  {perlin_lesion.sum()}")

            # Put the result back
            result = PeriventricularMS.apply_mask_with_value(
                img_data=img.data,
                bbox=bbox,
                new_label_mask=perlin_lesion,
                labels_to_target=self.labels_to_target,
                value=self.lesion_label
            )

            # Apply lesion
            img.set_data(result)
            
        self.add_labels_names(subject, [self.lesion_label], contrasted_labels=[self.lesion_label])
            
        return subject

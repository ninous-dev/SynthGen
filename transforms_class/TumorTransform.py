import logging
import os
import random
import uuid
import numpy as np
import torch
import torchio as tio
from typing import Optional, Tuple, Union, List
from scipy.ndimage import convolve
from transforms_class.AnomalyLabelTransform import AnomalyLabelTransform
from transforms_class.ZoomCenterTransform import ZoomCenterTransform
from transforms_class.utils.lesions_tool_box import compute_coverage, get_biggest_component
from transforms_class.utils.perlin_utils import generate_perlin_map, random_perlin_threshold
from utils.logger import BLUE, GREEN, RED, RESET, YELLOW
from utils.random import random_probability
import os.path as op

logger = logging.getLogger(__name__)

class TumorTransform(AnomalyLabelTransform):
    """
    Generate a synthetic tumor lesion in a label image, with optional core, necrosis, enhanced ring, and edema components.

    This transform allows for the creation of realistic tumor shapes by combining geometric primitives, random deformations,
    and Perlin noise. It can target specific labels in the input image and optionally preserve or include other labels in the
    generated lesion or edema.

    Parameters
    ----------
    labels_to_target : Union[int, List[int]]
        Label(s) in the input image to target for tumor generation. The tumor will be generated in the vicinity of these labels.
    lesion_label : int
        The label value to assign to the generated tumor lesion in the output image.
    tumor_size : Union[int, Tuple[int, int]], optional
        Desired size of the tumor core. If an int is provided, it is used as both the minimum and maximum size.
        If a tuple is provided, it should be (min_size, max_size). If None, no size constraint is applied.
    edema_ratio : Union[float, Tuple[float, float]], optional
        Ratio to determine the size of the edema relative to the core. If a float is provided, it is used as both
        the minimum and maximum ratio. If a tuple is provided, it should be (min_ratio, max_ratio). Default is (1.2, 2.0).
    core_ratio : Union[float, Tuple[float, float]], optional
        Ratio to determine the size of the core relative to the initial crop. If a float is provided, it is used as both
        the minimum and maximum ratio. If a tuple is provided, it should be (min_ratio, max_ratio). Default is (1.2, 1.5).
    edema_proba : float, optional
        Probability of generating edema around the tumor core. Default is 0.5.
    deformed_core : bool, optional
        Whether to apply deformation to the tumor core to create irregular shapes. Default is True.
    ring_around_core_proba : float, optional
        Probability of generating an enhanced ring around the tumor core. Default is 0.5.
    necrosis_proba : float, optional
        Probability of generating necrosis within the tumor core. Default is 0.75.
    labels_to_keep : Union[List[Union[int, str]], None], optional
        List of label values in the input image that should be preserved during tumor generation. Default is None.
    labels_for_edema : Union[List[Union[int, str]], None], optional
        List of label values in the input image that can be included in the edema region. If None, edema can be generated around any label.
    edema_pnoise_proba : float, optional
        Probability of applying Perlin noise to the edema region to create more irregular shapes. Default is 0.5.

    Attributes
    ----------
    tumor_size : Union[int, Tuple[int, int]]
        The minimum and maximum size of the tumor core.
    edema_ratio : Union[float, Tuple[float, float]]
        The ratio used for edema size relative to the core.
    core_ratio : Union[float, Tuple[float, float]]
        The ratio used for core size relative to the initial crop.
    transform_name : str
        Suffix string describing the tumor components generated (e.g., "D" for deformed, "N" for necrosis, etc.).
    labels_to_target : list of int
        List of labels to target for tumor generation.
    labels_to_keep : Union[List[Union[int, str]], None]
        List of labels to preserve during tumor generation.
    labels_for_edema : Union[List[Union[int, str]], None]
        List of labels that can be included in the edema region.
    """
    def __init__(
        self,
        labels_to_target: Union[int, List[int]],
        lesion_label: int,
        tumor_size: Tuple[int] = None,
        edema_ratio: Union[float, Tuple[float, float]] = (1.2, 2.0),
        core_ratio: Union[float, Tuple[float, float]] = (1.2, 1.5),
        edema_proba: float = 0.5,
        deformed_core: bool = True,
        ring_around_core_proba: float = 0.5,
        necrosis_proba: float = 0.75,
        labels_to_keep: List[Union[int, str]] = None,
        labels_for_edema: List[Union[int, str]] = None,
        edema_pnoise_proba: float = 0.5,
        **kwargs
    ):
        self.lesion_label = lesion_label
        suffix = "Tumor"
        super().__init__(anomaly_suffix=suffix, **kwargs)
        
        self.labels_to_target = AnomalyLabelTransform.int_to_list(labels_to_target)
        self.tumor_size = tumor_size
        self.edema_proba = edema_proba
        self.deformed_core = deformed_core
        self.ring_around_core_proba = ring_around_core_proba
        self.necrosis_proba = necrosis_proba
        self.labels_to_keep = labels_to_keep
        self.labels_for_edema = labels_for_edema
        self.core_ratio = core_ratio
        self.edema_ratio = edema_ratio
        self.edema_pnoise_proba = edema_pnoise_proba
        
        # Perlin parameters
        self.perlin_args = {
                "scale": 20,
                "octaves": 10,
                "lacunarity": 2,
                "max_growth": 2,
                "dilation": 2
            }

        self.args_names = [
            "labels_to_target",
            "lesion_label",
            "edema_proba",
            "necrosis_proba",
            "ring_around_core_proba",
            "edema_pnoise_proba",
            "edema_ratio",
            "core_ratio"
        ]

    def check_parameters(self):
        if self.tumor_size is not None:
            if isinstance(self.tumor_size, int):
                self.tumor_size = (self.tumor_size, self.tumor_size)
            elif isinstance(self.tumor_size, tuple) and len(self.tumor_size) == 2:
                if self.tumor_size[1] < self.tumor_size[0]:
                    raise ValueError(f"Invalid tumor_size range: {self.tumor_size}. The second value must be greater than or equal to the first.")
            else:
                raise ValueError(f"Invalid tumor_size '{self.tumor_size}'. Must be an int or a tuple of two ints.")
    
    def resolve_random_ratio(self, ratio: Union[float, Tuple[float, float]]) -> float:
        if isinstance(ratio, (tuple, list)):
            ratio = round(random.uniform(ratio[0], ratio[1]), 1)
            return ratio

        if isinstance(ratio, (float, int)):
            return ratio

        raise ValueError(f"Invalid ratio '{ratio}'. Must be a float or a tuple of two floats got {type(ratio)}.")

    def get_contours(self, lesion_mask: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask of contour voxels for `lesion_label`.

        Parameters
        ----------
        lesion_mask : np.ndarray
            3D label/segmentation array.

        Returns
        -------
        np.ndarray
            Boolean array (same shape) where True marks contour voxels.
        """
        # Get contours
        kernel = np.ones((3, 3, 3), dtype=np.uint8)
        neighbor_count = convolve(lesion_mask.astype(np.uint8), weights=kernel, mode="constant")
        contours = (neighbor_count < 27) & (neighbor_count > 0)
        return contours
    
    def get_core(self, label_image: torch.Tensor):

        max_iterations = 1000
        device = label_image.device
        target_tensor = torch.tensor(self.labels_to_target, device=device)

        for _ in range(max_iterations):
            # Get random point in self labels_to_target
            point_on_target = torch.nonzero(torch.isin(label_image, target_tensor).squeeze(0), as_tuple=False)
            if len(point_on_target) == 0:
                raise ValueError(f"No voxels found for target labels {self.labels_to_target} in the label image.")
            random_idx = random.randint(0, len(point_on_target) - 1)
            position = point_on_target[random_idx].tolist()

            (
                cropped_pure,
                cropped_margin,
                min_idx,
                max_idx,
                min_idx_margin,
                max_idx_margin,
            ) = AnomalyLabelTransform.crop_pure_target_box(
                label_image,
                position,
                self.tumor_size[0],
                self.tumor_size[1],
                self.labels_to_target,
                random_max_sizes=False,
                margin=0.6
            )

            if self.tumor_size[0] and (min(cropped_pure.shape) < self.tumor_size[0]):
                logger.warning("%s box too small (%s < %s), retrying... %s" % (YELLOW, cropped_pure.shape, self.tumor_size[0], RESET))
                continue
            logger.info(f"{BLUE}[TumorTransform]{RESET} Box size: {BLUE}{cropped_pure.shape}{RESET}")

            # Ellipse on PURE box
            center, radii = AnomalyLabelTransform.generate_radii_from_shape(cropped_pure.shape)
            elliptical = AnomalyLabelTransform.generate_elliptical(cropped_pure.shape, center, radii)

            # Insert into margin box
            offset = [
                min_idx[d] - min_idx_margin[d]
                for d in range(3)
            ]

            full = torch.zeros_like(cropped_margin, dtype=torch.bool)

            full[
                offset[0]:offset[0] + elliptical.shape[0],
                offset[1]:offset[1] + elliptical.shape[1],
                offset[2]:offset[2] + elliptical.shape[2],
            ] = elliptical.bool()

            targets = torch.isin(cropped_margin, target_tensor)

            return (
                full & targets,
                min_idx_margin,
                max_idx_margin,
                position,
                min(cropped_pure.shape),
                max(cropped_pure.shape)
            )
        raise RuntimeError("Max iterations reached while trying to find a valid tumor core location.")

    def add_lesion_labels(
        self,
        label_image_tensor: torch.Tensor,
        lesion_label_full: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add lesion labels to the label image tensor based on the generated lesion mask.

        Parameters
        ----------
        label_image_tensor : torch.Tensor
            The original label image tensor to which lesion labels will be added.
        lesion_label_full : torch.Tensor
            A tensor of the same shape as `label_image_tensor` containing the generated lesion mask

        Returns
        -------
        torch.Tensor
            A new tensor with lesion labels added to the original label image tensor.
        """
        # Define the mapping of lesion components to their respective label values
        lesion_label_names = {
            "tumor_core": self.lesion_label,
            "tumor_necrosis": self.lesion_label + 1,
            "tumor_enhanced_ring": self.lesion_label + 2,
        }
        
        # Add the main tumor core label to the target locations
        target_mask = torch.isin(label_image_tensor, torch.tensor(self.labels_to_target))

        tumor_without_edema_label = torch.where(lesion_label_full <= 3, lesion_label_full, 0)
        tumor_without_edema_label = torch.where(get_biggest_component((tumor_without_edema_label > 0).squeeze()).unsqueeze(0) > 0, tumor_without_edema_label, 0)

        result = torch.where(target_mask & (tumor_without_edema_label > 0), tumor_without_edema_label + self.lesion_label - 1, label_image_tensor)

        # Add edema label if edema is generated
        if self.edema_proba:
            lesion_label_names["tumor_edema"] = self.lesion_label + 3
            if self.edema_pnoise_proba:
                lesion_label_names["tumor_edema_pnoise"] = self.lesion_label + 4

            edema_mask = torch.isin(label_image_tensor, torch.tensor(self.labels_for_edema))
            edema_label = torch.where(lesion_label_full > 3, lesion_label_full, 0)
            result = torch.where(edema_mask & (edema_label > 0), edema_label + self.lesion_label - 1, result)
        return result, lesion_label_names

    def prepare_tumor_generation_parameters(self) -> None:
        if isinstance(self.lesion_label, list) and len(self.lesion_label) == 1:
            self.lesion_label = self.lesion_label[0]

        if not isinstance(self.lesion_label, int):
            raise ValueError(
                f"lesion_label must be an integer. Got {type(self.lesion_label)} with value {self.lesion_label}."
            )

        self.edema_ratio = self.resolve_random_ratio(self.edema_ratio)
        self.core_ratio = self.resolve_random_ratio(self.core_ratio)

        logger.info(f"{BLUE}[TumorTransform]{RESET} Starting tumor generation with parameters:")
        logger.info(f"{BLUE}[TumorTransform]{RESET} Edema ratio: {BLUE}{self.edema_ratio}{RESET}")
        logger.info(f"{BLUE}[TumorTransform]{RESET} Core ratio: {BLUE}{self.core_ratio}{RESET}")

    def deform_and_scale_core(self, core: torch.Tensor, spacing: Union[int, float]) -> torch.Tensor:
        """
        Deform and scale the tumor core mask.

        Parameters
        ----------
        core : torch.Tensor
            The binary mask of the tumor core. Shape (C, D, H, W) or (D, H, W).
        spacing : Union[int, float]
            The voxel spacing used to determine the deformation grid space.

        Returns
        -------
        torch.Tensor
            The deformed and scaled tumor core mask, same shape as input.
        """
        if core.ndim == 3:
            core = core.unsqueeze(0)

        core = (core > 0)

        if self.deformed_core:
            self.transform_name += "D"
            random_field_mask = "bidirectional"
            logger.info(
                f"{BLUE}[TumorTransform]{RESET} {GREEN}Deforming core{RESET} with random {BLUE}{random_field_mask}{RESET} field"
            )

            grid_space = int(3 * 1/min(spacing)) # in mm
            logger.info(f"{BLUE}[TumorTransform]{RESET} Deforming core with grid space of {BLUE}{grid_space}{RESET} (in voxel)")
            core = AnomalyLabelTransform.deform_mask(
                random_field_mask=random_field_mask,
                mask=core,
                max_grid_space=grid_space,
                displacement=1
            )
        core = ZoomCenterTransform.apply_to_tensor(core, ratio=self.core_ratio)
        return core

    def generate_necrosis_mask(
        self,
        core: torch.Tensor,
    ) -> Optional[np.ndarray]:
        """
        Generate a necrosis mask within the tumor core using Perlin noise.
    
        This function probabilistically generates a necrosis region inside the tumor core.
        If necrosis is not selected (based on `self.necrosis_proba`), returns None.
        Otherwise, a Perlin noise map is generated and thresholded to create an irregular necrosis mask
        within the core region.
    
        Parameters
        ----------
        core : torch.Tensor
            Binary mask of the tumor core. Shape (C, D, H, W) or (D, H, W).
    
        Returns
        -------
        Union[np.ndarray, None]
            Boolean numpy array mask of necrosis (same shape as core, squeezed), or None if necrosis is not generated.
        """
        self.necrosis_proba = random_probability(self.necrosis_proba)
        if not self.necrosis_proba:
            return None

        logger.info(f"{BLUE}[TumorTransform]{RESET} {GREEN}Necrosis generation{RESET}")
        self.transform_name += "N"

        perlin_map = generate_perlin_map(
            core.squeeze().shape,
            scale=self.perlin_args["scale"],
            octaves=self.perlin_args["octaves"],
            lacunarity=self.perlin_args["lacunarity"],
            base=random.randint(0, 100),
        )
        perlin_map = (perlin_map - perlin_map.min()) / (perlin_map.max() - perlin_map.min())
        threshold = random.uniform(0.25, 0.5)

        return (perlin_map > threshold) & core.bool().cpu().numpy()

    def generate_enhanced_ring_mask(
        self,
        core: torch.Tensor,
    ) -> Optional[np.ndarray]:
        """
        Generate an enhanced ring mask around the tumor core using Perlin noise and morphological operations.

        This function probabilistically generates an enhanced ring region surrounding the tumor core.
        If ring generation is not selected (based on `self.ring_around_core_proba`), returns None.
        Otherwise, an irregular ring contour is created using Perlin noise and morphological dilation.

        Parameters
        ----------
        core : torch.Tensor
            Binary mask of the tumor core. Shape (C, D, H, W) or (D, H, W).
        max_growth : int
            Maximum number of voxels the ring can grow outward from the core.
        scale : int
            Scale parameter for the Perlin noise generation.
        octaves : int
            Number of octaves for the Perlin noise.
        dilation : int
            Number of dilation iterations to apply to the ring.

        Returns
        -------
        Union[np.ndarray, None]
            Boolean numpy array mask of the enhanced ring (same shape as core, squeezed), or None if ring is not generated.
        """
        self.ring_around_core_proba = random_probability(self.ring_around_core_proba)
        if not self.ring_around_core_proba:
            return None

        logger.info(
            f"{BLUE}[TumorTransform]{RESET} {GREEN}Enhanced ring generation around core{RESET} with irregular contours"
        )
        self.transform_name += "R"

        return self.create_irregular_contours_perlin(
            lesion_mask=core.bool().numpy(),
            max_growth=self.perlin_args["max_growth"],
            scale=self.perlin_args["scale"],
            octaves=self.perlin_args["octaves"],
            dilation=self.perlin_args["dilation"],
            perlin_threshold=random_perlin_threshold(),
        )

    def generate_edema_mask(
        self,
        core: torch.Tensor,
        spacing: Union[int, float],
    ) -> torch.Tensor:
        self.edema_proba = random_probability(self.edema_proba)
        if not self.edema_proba:
            return torch.zeros_like(core).int()

        logger.info(f"{BLUE}[TumorTransform]{RESET} {GREEN}Edema generation{RESET}")
        self.transform_name += "E"

        center_of_core = [dim // 2 for dim in core.shape[1:]]
        logger.info(
            f"{BLUE}[TumorTransform]{RESET} Initial center of core on crop: {BLUE}{','.join(map(str, center_of_core))}{RESET}"
        )

        # Zoom edema
        grid_space = int(3 * 1/min(spacing)) # in mm
        random_field_mask="bidirectional"

        edema_mask = AnomalyLabelTransform.deform_mask(
            random_field_mask=random_field_mask,
            mask=(core > 0).int(),
            max_grid_space=grid_space,
            displacement=1
        )
        edema_mask = ZoomCenterTransform.apply_to_tensor(edema_mask, ratio=self.core_ratio)
        grid_space = int(3 * 1/min(spacing)) # in mm
        random_field_mask="bidirectional"

        edema_mask = AnomalyLabelTransform.deform_mask(
            random_field_mask=random_field_mask,
            mask=edema_mask,
            max_grid_space=grid_space,
            displacement=1
        )
        edema_mask = ZoomCenterTransform.apply_to_tensor(edema_mask, ratio=self.edema_ratio)

        self.edema_pnoise_proba = random_probability(self.edema_pnoise_proba)
        if self.edema_pnoise_proba:
            perlin_map = generate_perlin_map(
                edema_mask.squeeze(0).shape,
                scale=self.perlin_args["scale"],
                octaves=self.perlin_args["octaves"],
                lacunarity=self.perlin_args["lacunarity"],
                base=random.randint(0, 100),
            )
            perlin_map = (perlin_map - perlin_map.min()) / (perlin_map.max() - perlin_map.min())
            threshold = random.uniform(0.25, 0.5)
            perlin_mask = torch.from_numpy(perlin_map > threshold).int()
            edema_mask = edema_mask.int()
            edema_mask = edema_mask + (perlin_mask * edema_mask)


        # if no overlap between edema and core, we consider that edema generation failed and we return an empty mask
        if ((edema_mask > 0) & (core > 0)).sum() == 0:
            raise ValueError(f"{RED}Edema mask does not overlap with core mask. Edema generation failed.{RESET}")
        return edema_mask.int()
    
    def build_lesion_label_full(
        self,
        tio_image_data: torch.Tensor,
        core: torch.Tensor,
        necrosis_mask: Optional[np.ndarray],
        enhanced_ring_mask: Optional[np.ndarray],
        edema_mask: torch.Tensor,
        start_indices: Union[tuple, list],
        end_indices: Union[tuple, list],
    ) -> torch.Tensor:
        """
        Build the full lesion label tensor by aggregating core, necrosis, enhanced ring, and edema components.

        This function combines the tumor core, necrosis, enhanced ring, and edema masks into a single lesion label tensor,
        and inserts it into the correct location within the full image tensor.

        Parameters
        ----------
        tio_image_data : torch.Tensor
            The full label image tensor to fill (e.g., shape (C, D, H, W)).
        core : torch.Tensor
            Binary mask of the tumor core (same shape as the cropped region).
        necrosis_mask : Union[np.ndarray, None]
            Boolean mask of necrosis region (same shape as core), or None if not generated.
        enhanced_ring_mask : Union[np.ndarray, None]
            Boolean mask of enhanced ring region (same shape as core), or None if not generated.
        edema_mask : torch.Tensor
            Mask of the edema region (same shape as core).
        start_indices : Union[tuple, list]
            Start indices (z, y, x) for inserting the cropped region into the full image.
        end_indices : Union[tuple, list]
            End indices (z, y, x) for inserting the cropped region into the full image.

        Returns
        -------
        torch.Tensor
            The full lesion label tensor with all tumor components inserted at the correct location.
        """
        tumor_lesion = core.int().clone()

        if necrosis_mask is not None:
            tumor_lesion[necrosis_mask] = 2

        if enhanced_ring_mask is not None:
            tumor_lesion[enhanced_ring_mask] = 3

        tumor_lesion = torch.where(
            (tumor_lesion == 0) & (edema_mask > 0),
            3 + edema_mask,
            tumor_lesion,
        )

        lesion_label_full = torch.zeros_like(tio_image_data)
        lesion_label_full[
            :,
            start_indices[0]:end_indices[0],
            start_indices[1]:end_indices[1],
            start_indices[2]:end_indices[2],
        ] = tumor_lesion

        return lesion_label_full

    @AnomalyLabelTransform.update_subject_anomaly_and_log_success
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """
        Apply the transform to the given subject.

        Parameters
        ----------
        subject : tio.Subject
            The subject to which the transform will be applied.

        Returns
        -------
        tio.Subject
            The transformed subject.
        """
        self.prepare_tumor_generation_parameters()
        nb_images_processed = 0
        # Apply the transform to each image in the subject
        for image_name, tio_image in self.get_images_dict(subject).items():
            if not isinstance(tio_image, tio.LabelMap):
                continue
            if image_name == "__label_without_lesion":
                continue
            
            if nb_images_processed > 0:
                raise NotImplementedError(f"Multiple label images found in subject. Currently only supports one. Found {nb_images_processed+1}.")
            nb_images_processed += 1
            
            spacing = tio_image.spacing

            if "__label_without_lesion" not in subject:
                without_lesion_scalar_image = tio.LabelMap(
                    tensor=tio_image.data,
                    affine=tio_image.affine)  
                subject.add_image(without_lesion_scalar_image, "__label_without_lesion")
                logger.info(f"{BLUE}[TumorTransform]{RESET} Adding __label_without_lession in subject ")

            logger.info(f"{BLUE}[TumorTransform]{RESET} Apply on label name : {BLUE}{image_name}{RESET}")

            self.transform_name = ""
            
            # Get core and its position in the label image referential
            core_initial, start_indices, end_indices, position, tumor_min_size, tumor_max_size = self.get_core(tio_image.data)
            logger.info(f"{BLUE}[TumorTransform]{RESET} Core generated with size: {BLUE}{core_initial.shape}{RESET}")
            logger.info(f"{BLUE}[TumorTransform]{RESET} Core located at: {BLUE}{','.join(map(str, position))}{RESET}")
            logger.info(f"{BLUE}[TumorTransform]{RESET} Core tumor ratio (min/max size) (information about spherical aspect): {BLUE}{tumor_min_size/tumor_max_size}{RESET}")

            #generate directory 
            core = self.deform_and_scale_core(core_initial, spacing)
            
            # Process necrosis            
            necrosis_mask = self.generate_necrosis_mask(
                core=core,
            )
            
            # Process ring around core
            enhanced_ring_mask = self.generate_enhanced_ring_mask(
                core=core,
            )
                            
            # Process edema
            edema_mask = self.generate_edema_mask(
                core=core_initial,
                spacing=spacing
            )

            # Aggregate all the different part of the tumor and add it to the label image
            lesion_label_full = self.build_lesion_label_full(
                tio_image_data=tio_image.data,
                core=core,
                necrosis_mask=necrosis_mask,
                enhanced_ring_mask=enhanced_ring_mask,
                edema_mask=edema_mask,
                start_indices=start_indices,
                end_indices=end_indices,
            )
            logger.info(f"{BLUE}[TumorTransform]{RESET} Core, necrosis, enhanced ring and edema generated. {GREEN}Aggregating tumor components.{RESET}")

            tumor_without_edema = torch.where(lesion_label_full <= 3, lesion_label_full , 0)
            tumor_without_edema = torch.where(torch.isin(tio_image.data, torch.tensor(self.labels_to_keep)), 0, tumor_without_edema)
            
            # Get lesion label in label image referential
            new_label_image = tio_image.data.clone()
            
            deformation_iteration_limit = 5
            deformation_iteration = 1
            displacement = 0.8

            grid_space = int(5 * 1/min(spacing)) # in mm
            minimal_coverage = 0.95

            logger.info(f"{BLUE}[TumorTransform]{RESET} deformation_iteration_limit={BLUE}{deformation_iteration_limit}{RESET}")

            while deformation_iteration <= deformation_iteration_limit:
                logger.info("%s[TumorTransform]%s deformation iteration %s%d%s", BLUE, RESET, BLUE, deformation_iteration, RESET)

                target_mask = torch.isin(new_label_image, torch.tensor(self.labels_to_target))
                target_lesion_mask = target_mask & (lesion_label_full > 0)
                target_lesion_mask = get_biggest_component(target_lesion_mask.squeeze()).unsqueeze(0) > 0
                target_lesion_mask_labelmap = tio.LabelMap(tensor=target_lesion_mask.int(), affine=tio_image.affine)

                previous_coverage = compute_coverage(
                    torch.isin(new_label_image.squeeze(0), torch.tensor(self.labels_to_target)), 
                    lesion_label_full.squeeze(0) > 0)
                logger.info(f"{BLUE}[TumorTransform]{RESET} Current coverage: {YELLOW}{previous_coverage}{RESET}")

                # Deform the label image
                deformed_label_image = AnomalyLabelTransform.deform_mask(
                    random_field_mask=None,
                    mask=new_label_image.clone(),
                    mode="push",
                    displacement=displacement,
                    max_grid_space=grid_space,
                    labels_to_keep=self.labels_to_keep,
                    labels_to_target=self.labels_to_target,
                    attention_mask=target_lesion_mask_labelmap,
                    target_point=position,
                    keep_dilation=5, #TODO definir sans magic number
                )

                current_coverage = compute_coverage(
                    torch.isin(deformed_label_image.squeeze(0), torch.tensor(self.labels_to_target)),
                    lesion_label_full.squeeze() > 0)
                logger.info(f"{BLUE}[TumorTransform]{RESET} Coverage after deformation: {YELLOW}{current_coverage}{RESET}")

                if current_coverage > minimal_coverage:
                    print(f"{GREEN}Inside is good after deformation got {current_coverage} > {minimal_coverage}. Stopping deformation process.{RESET}")
                    new_label_image = deformed_label_image
                    break

                diff = current_coverage - previous_coverage
                if (current_coverage <= previous_coverage):
                    print(f"{RED}Inside did not improve after deformation got {current_coverage} <= {previous_coverage}. Stopping deformation process.{RESET}")
                    deformed_label_image = new_label_image
                    break
                if current_coverage <= previous_coverage:
                    print(f"{RED}Inside did not improve after deformation got {current_coverage} <= {previous_coverage}. Stopping deformation process.{RESET}")
                    break
                if diff < 0.002:
                    print(f"{YELLOW}Inside improvement is low after deformation got {current_coverage} <= {previous_coverage}. Stopping deformation process.{RESET}")
                    break
                elif deformation_iteration >= deformation_iteration_limit:
                    print(f"{YELLOW}Maximum number of deformations reached: {deformation_iteration}. Stopping deformation process.{RESET}")
                    break
                deformation_iteration += 1
                new_label_image = deformed_label_image

            # inside with tumor_full and new_label_image
            current_coverage = compute_coverage(torch.isin(new_label_image.squeeze(0), torch.tensor(self.labels_to_target)), lesion_label_full.squeeze() > 0)
            if current_coverage < 0.95:
                print(RED, f"Inside is still low after deformation process: {current_coverage}. Consider increasing the number of deformations or adjusting deformation parameters.", RESET)

            tio_image.set_data(new_label_image)
            tio_image_no_lession = tio.LabelMap(tensor=new_label_image, affine=tio_image.affine)
            # TODO Save label deformed without lesion TODO here we still have lesion labels in it ...
            subject.add_image(tio_image_no_lession, "__label_without_lesion")

            result, lesion_label_names = self.add_lesion_labels(tio_image.data, lesion_label_full)
            tio_image.set_data(result)

        # Add tumor informations to subject for logging and post processing
        subject["tumor_min_size"] = tumor_min_size
        subject["tumor_max_size"] = tumor_max_size
        
        self.add_labels_names(subject, lesion_label_names)

        return subject

import logging
import random
from transforms_class.LabelTransform import LabelTransform
from transforms_class.utils import DeformationTransform
from utils.logger import BLUE, GREEN, RED, RESET
from typing import List, Optional, Sequence, Tuple
import numpy as np
import torch
import torchio as tio
import torch.nn.functional as F
from scipy.ndimage import binary_dilation

from transforms_class.utils.perlin_utils import generate_perlin_map

logger = logging.getLogger(__name__)

class AnomalyLabelTransform(LabelTransform):
    """
    AnomalyLabelTransform applies transformations to the labels of an input image.

    This class is designed to modify the label map of an image by introducing anomalies
    based on a specified suffix. It supports resolving and validating anatomy labels
    and updating subject metadata.

    Attributes
    ----------
    anomaly_suffix : str
        The suffix used to define the anomaly type.
    labels_to_target : list[int] or None
        The list of target labels to modify.
    labels_to_keep : list[int] or None
        The list of labels to preserve during transformation.
    """
    def __init__(self, anomaly_suffix: str, nickname: str = None, **kwargs):
        super().__init__(**kwargs)

        if anomaly_suffix is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define class attribute "
                "`ANOMALY_SUFFIX`"
            )

        self.anomaly_suffix = anomaly_suffix
        if nickname is not None:
            self.anomaly_suffix = nickname #erase suffix to stay short
    
    def update_subject_anomaly_and_log_success(func):
        """
        Decorator to update the subject's anomaly metadata and log a success message after a transformation.

        After the decorated function is called, updates the subject's anomaly fields and logs a formatted success message.

        Parameters
        ----------
        func : callable
            The function to decorate.
    
        Returns
        -------
        callable
            The wrapped function.
        """
        def wrapper(self, subject: tio.Subject, *args, **kwargs):
            result = func(self, subject, *args, **kwargs)
            self.update_subject_anomaly(result)
            subject.update_attributes()
            
            # Log success message with clear formatting
            logger.info("%s-----------------------------------------%s", GREEN, RESET)
            logger.info("%sSuccess %s.%s", GREEN, self.anomaly_suffix, RESET)
            logger.info("%s-----------------------------------------%s", GREEN, RESET)
            return result
        return wrapper
    
    @staticmethod
    def int_to_list(value):
        """Convertit un int en liste, sinon retourne la valeur inchangée."""
        if isinstance(value, int):
            return [value]
        return value

    @staticmethod
    def int_to_tuple(value):
        """Convertit un int en tuple (value, value), sinon retourne la valeur inchangée."""
        if isinstance(value, int):
            return (value, value)
        return value
    
    @staticmethod
    def get_contour3d(
            mask: torch.Tensor,
            iterations: int = 1,
            external: bool = True
        ) -> torch.Tensor:
            """
            Extract a 3D contour from a binary mask.

            Parameters
            ----------
            mask : torch.Tensor
                Binary/boolean tensor of shape (N, C, D, H, W) or (D, H, W) after unsqueeze.

            iterations : int, optional
                Thickness (>=1). default=1.

            external : bool, optional
                If True, outer contour; else inner contour. default=True.

            Returns
            -------
            torch.Tensor
                Contour mask with the same shape and dtype as `mask`.
            """
            # Check that mask is of boolean type
            if isinstance(mask, torch.Tensor) and mask.dtype != torch.bool:
                raise TypeError("Input mask must be of dtype torch.bool")

            if isinstance(mask, np.ndarray) and mask.dtype != np.bool_:
                raise TypeError("Input mask must be of dtype np.bool_")

            if iterations < 1:
                raise ValueError("`iterations` must be >= 1")

            # Kernel size for contour thickness
            k = 2 * iterations + 1

            # Outer contour: dilate mask and subtract original mask
            if external:
                # Dilate the mask using max_pool3d, then find the difference
                dilated = F.max_pool3d(mask.float(), kernel_size=k, stride=1, padding=iterations).bool()
                contour = dilated & ~mask
            else:
                # Inner contour: erode the inverse mask, then find the difference
                inv = ~mask
                eroded = ~F.max_pool3d(inv.float(), kernel_size=3, stride=1, padding=1).bool()
                contour = mask & ~eroded

            # Ensure the output shape matches the input shape using tio.CropOrPad
            contour = tio.CropOrPad(mask.shape[-3:])(contour).bool()

            return contour
        
    @staticmethod
    def choose_random_param(param, list_choice=False):
        """
        If param is a tuple, return a random value in the range (inclusive for int, uniform for float).
        Otherwise, return param as is.
        """
        if isinstance(param, (int, float, str)):
            return param
        if list_choice:
            if isinstance(param, list):
                return random.choice(param)
            if isinstance(param, (int, float, str)):
                return param
            raise ValueError(f"Expected int or list for list_choice=True, got {type(param)}")
        if isinstance(param, (tuple, list)):
            if all(isinstance(x, int) for x in param):
                return random.randint(param[0], param[1])
            elif all(isinstance(x, float) for x in param):
                return round(random.uniform(param[0], param[1]), 2)
        else:
            raise ValueError("Parameter must be either a list, a tuple, or a single value. Got: {}, which is of type {}".format(param, type(param)))
        return param

    def update_subject_anomaly(self, subject: tio.Subject) -> None:
        """
        Update the subject's anomaly metadata fields with the current anomaly suffix.
    
        Updates or appends the "anomaly_type" and "suffix" fields in the subject's metadata.
    
        Parameters
        ----------
        subject : tio.Subject
            The subject whose metadata will be updated.
        """
        # TODO anomaly_type used only in history maybe should be same
        if "anomaly_type" in subject and self.anomaly_suffix not in subject["anomaly_type"]:
            subject["anomaly_type"] += self.anomaly_suffix
        elif "anomaly_type" not in subject:
            subject["anomaly_type"] = self.anomaly_suffix

        if "suffix" not in subject:
            subject["suffix"] = "Ano"
        else:
            if self.anomaly_suffix not in subject["suffix"]:
                subject["suffix"] += self.anomaly_suffix

    @staticmethod
    def get_bounding_box(mask: torch.Tensor, spacing: float) -> dict:
        """
        Get the minimal bounding box of non-zero elements in the mask.

        Parameters
        ----------
        mask : torch.Tensor
            Tensor of shape (1, D, H, W) or (D, H, W).
        spacing : float
            Spacing value used to calculate padding.

        Returns
        -------
        dict
            Dictionary with keys: 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'.
        """
        if mask.dim() == 4:
            mask = mask.squeeze(0)  # Remove channel dimension

        # Find non-zero indices
        indices = torch.nonzero(mask, as_tuple=True)

        if len(indices[0]) == 0:
            raise ValueError("No non-zero elements in mask")
        padding = int(5 * (1 / spacing))

        # Ensure consistent naming and usage of bounding box keys
        bbox = {
            'x_min': indices[0].min().item(),
            'x_max': indices[0].max().item() + 1,
            'y_min': indices[1].min().item(),
            'y_max': indices[1].max().item() + 1,
            'z_min': indices[2].min().item(),
            'z_max': indices[2].max().item() + 1,
        }

        # Apply padding to the bounding box dimensions
        for axis in ['x', 'y', 'z']:
            bbox[f'{axis}_min'] = max(0, bbox[f'{axis}_min'] - padding)
            bbox[f'{axis}_max'] = min(mask.shape['xyz'.index(axis)], bbox[f'{axis}_max'] + padding)

        return bbox
    
    @staticmethod
    def crop_with_bbox(tensor: torch.Tensor, bbox: dict) -> torch.Tensor:
        """
        Crop a tensor using a bounding box dictionary with keys: z_min, z_max, y_min, y_max, x_min, x_max.
        Args:
            tensor: Tensor of shape (1, D, H, W) or (D, H, W)
            bbox: dict with keys as above
        Returns:
            Cropped tensor
        """
        # Crop the tensor using the bounding box for 4D tensors
        return tensor[:,
            bbox['x_min']:bbox['x_max'],
            bbox['y_min']:bbox['y_max'],
            bbox['z_min']:bbox['z_max']
        ]
        
    @staticmethod
    def apply_mask_with_value(
        img_data: torch.Tensor,
        bbox: dict,
        new_label_mask: np.ndarray,
        labels_to_target: Tuple[int],
        value: int
        ) -> torch.Tensor:
        """
        Apply the mask with the given lesion value to the image data.

        Parameters
        ----------
        img_data : torch.Tensor
            The image data tensor.
            
        bbox : dict
            The bounding box dictionary with keys "x_min", "x_max", "y_min", "y_max", "z_min", "z_max".
        
        new_label_mask : np.ndarray
            The new label mask to apply within the bounding box.
        
        labels_to_target : Tuple[int]
            The target lesion labels.
        
        value : int
            The value to assign to the new mask.

        Returns
        -------
        torch.Tensor
            The updated image data tensor with the lesion applied.
        """
        # Insert the lesion mask into the correct region
        target_mask = torch.zeros_like(img_data, dtype=torch.bool)
        target_mask[
            :,
            bbox['x_min']:bbox['x_max'],
            bbox['y_min']:bbox['y_max'],
            bbox['z_min']:bbox['z_max']
        ] = new_label_mask

        # Remove ventricles from mask
        target_mask = target_mask & torch.isin(img_data, torch.tensor(labels_to_target))

        # Apply lesion
        return torch.where(target_mask, value, img_data)

    @staticmethod
    def apply_perlin_on_mask(
        mask: np.ndarray,
        octaves: int = 4,
        perlin_threshold: int = 0,
        lacunarity: float = 2.0,
        perlin_scale: float = 20.0,
        base: int = None
    ) -> np.ndarray:
        """
        Apply Perlin noise to the input mask.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask of the target.
        octaves : int, optional
            Number of Perlin octaves (default=4).
        perlin_threshold : int, optional
            Threshold for Perlin noise mask (default=0).
        lacunarity : float, optional
            Lacunarity parameter for Perlin noise (default=2.0).

        Returns
        -------
        np.ndarray
            Binary mask with irregular lesion contours.
        """
        base = base if base is not None else np.random.randint(0, 10)
        mask_np = mask.squeeze(0).squeeze(0).numpy()
        perlin_map = generate_perlin_map(
                mask_np.shape,
                scale=perlin_scale,
                octaves=octaves,
                lacunarity=lacunarity,
                base=base
            )

        # Normalize Perlin map to [0, 1]
        perlin_map = (perlin_map - perlin_map.min()) / (perlin_map.max() - perlin_map.min() + 1e-8)
        perlin_thresholded = perlin_map < perlin_threshold
        
        # Add to lesion mask
        lesion_mask_with_lesion = mask_np.astype(bool) & perlin_thresholded & mask_np.astype(bool)

        return lesion_mask_with_lesion
    
    @staticmethod
    def put_cropped_patch_back(label_tensor: torch.Tensor, cropped_patch: torch.Tensor, start_indices: List[int], end_indices: List[int]) -> torch.Tensor:
        """
        Puts the cropped patch back into the original label_tensor at the correct position.

        Parameters
        ----------
        label_tensor: torch.Tensor
            The original label tensor to put the cropped patch into.

        cropped_patch: torch.Tensor
            The cropped patch to insert back into the label tensor.

        start_indices: List[int]
            The starting indices of the cropped patch in the original label tensor.

        end_indices: List[int]
            The ending indices of the cropped patch in the original label tensor.
            
        Returns
        -------
        torch.Tensor
            Updated label_tensor with the cropped patch placed back.
        """
        expected_shape = [
            end_indices[dim] - start_indices[dim] for dim in range(len(start_indices))
        ]
        
        if cropped_patch.shape != tuple(expected_shape):
            raise ValueError(f"Shape mismatch: Cropped patch shape {cropped_patch.shape} does not match "
                            f"expected shape {tuple(expected_shape)} for the slice.")

        updated_label_tensor = label_tensor.squeeze(0).clone()

        updated_label_tensor[
            start_indices[0]:end_indices[0],
            start_indices[1]:end_indices[1],
            start_indices[2]:end_indices[2]
        ] = cropped_patch

        return updated_label_tensor.unsqueeze(0)

    @staticmethod
    def deform_mask(
        random_field_mask: str,
        mask: torch.Tensor,
        inverse_axes: List[int]=[0, 1],
        flip_axes: List[int]=[0, 1],
        mode: str='pull',
        displacement: float=1.0,
        max_grid_space: Optional[int] = None,
        labels_to_target: List[int] = [1],
        labels_to_keep: Optional[List[int]] = None,
        background_value: int = 0,
        locked_borders: int = 0,
        spline_order: int = 3,
        target_point: Optional[Tuple[int, int, int]] = None,
        keep_dilation: int = 2,
        perlin_unidirectional_threshold: float = 0.25,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create a deformation mask based on the cropped tensor and elliptical shape.
    
        The deformation is only applied if the mask is large enough for the specified grid spacing.
    
        Parameters
        ----------
        random_field_mask : str
            The type of random field mask to apply to the deformation. Can be "bidirectional", "unidirectional", or None.
        mask : torch.Tensor
            The tensor to be deformed. Can be 3D (D, H, W) or 4D (C, D, H, W).
        inverse_axes : list of int, optional
            Axes to invert for deformation, by default=[0, 1]).
        flip_axes : list of int, optional
            Axes to flip for deformation, by default=[0, 1]).
        mode : str, optional
            Deformation mode, either 'pull' or 'push', by default='pull'.
        displacement : float, optional
            The magnitude of displacement for deformation, by default=1.0.
        max_grid_space : int, optional
            Maximum grid space for deformation. If None, computed from mask size and spline order.
        labels_to_target : list of int, optional
            List of label values to target for deformation, by default=[1].
        labels_to_keep : list of int, optional
            List of label values to preserve during deformation, by default=None.
        background_value : int, optional
            Value to use for the background, by default=0.
        locked_borders : int, optional
            Number of border voxels to lock (prevent deformation), by default=0.
        spline_order : int, optional
            Spline interpolation order for deformation, by default=3.
        target_point : tuple of int, optional
            A specific point (z, y, x) in the image to target for deformation. If provided, deformation will be focused around this point.
        keep_dilation : int, optional
            Number of dilation iterations to apply to the keep_mask, which defines the area around the target point that should be preserved during deformation, by default=2.
        perlin_unidirectional_threshold : float, optional
            Threshold for Perlin noise in unidirectional deformation, by default=0.25.
        attention_mask : torch.Tensor, optional
            Optional attention mask to guide deformation.
    
        Returns
        -------
        torch.Tensor
            The deformation mask, with the same shape as the input mask.
        """
        if len(mask.shape) == 4:
            min_shape = min(mask.shape[1:])  # Exclude channel dimension
        elif len(mask.shape) == 3:
            min_shape = min(mask.shape)  # No channel dimension
        else:
            raise ValueError(f"Input mask must be 3D or 4D, got shape {mask.shape}")

        grid_space_max = min_shape // (spline_order + 1)
        grid_space = grid_space_max // spline_order if max_grid_space is None else min(max_grid_space, grid_space_max)

        if grid_space > 1:
            # case deformed large enough to be deformed
            mask = DeformationTransform.apply_to_tensor(
                mask,
                grid_space=grid_space,
                displacement=displacement,
                mode=mode,
                locked_borders=locked_borders,
                background_value=background_value,
                labels_to_target=labels_to_target,
                labels_to_keep=labels_to_keep,
                random_field_mask=random_field_mask,
                flip_axes=flip_axes,
                inverse_axes=inverse_axes,
                spline_order=spline_order,
                target_point=target_point,
                keep_dilation=keep_dilation,
                perlin_unidirectional_threshold=perlin_unidirectional_threshold,
                attention_mask=attention_mask,
            )
            
            logger.info(f"{BLUE}[AnomalyLabelTransform]{RESET} Deformation applied.")

        return mask
    
    @staticmethod
    def generate_radii_from_shape(shape: Sequence[int]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Generate a random ellipsoidal center and radii proportional to the input volume shape.

        The center is placed at the mid-volume (half of each dimension).
        Radii are chosen as random fractions.
        corresponding dimension, with the fractions shuffled across axes.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the volume as (z, y, x). Each dimension must be >= 1.

        Returns
        -------
        center : Tuple[int, int, int]
            The center of the ellipsoid in (z, y, x) coordinates.

        radii : Tuple[int, int, int]
            The radii of the ellipsoid along (z, y, x).
        """
        min_base = 0.15
        max_base = 0.3

        center = tuple(s // 2 for s in shape)

        base = [random.uniform(min_base, max_base) for _ in range(3)] # nosec B311
        random.shuffle(base)

        radii = tuple(int(shape[i] * base[i]) for i in range(3))

        return center, radii
    
    @staticmethod
    def create_irregular_contours_perlin(
        lesion_mask: np.ndarray,
        max_growth: int,
        scale: float,
        octaves: int = 4,
        dilation: int = 2,
        perlin_threshold: int = 0,
        lacunarity: float = 2.0
    ) -> np.ndarray:
        """
        Create irregular lesion contours using Perlin noise.

        Parameters
        ----------
        lesion_mask : np.ndarray
            Binary mask of the lesion.
            
        max_growth : int
            Number of growth iterations.
            
        scale : float
            Scale factor for Perlin noise.
            
        octaves : int, optional
            Number of Perlin octaves (default=4).
            
        dilation : int, optional
            Dilation iterations (default=2).
            
        perlin_threshold : int, optional
            Threshold for Perlin noise mask (default=0).

        lacunarity : float, optional
            Lacunarity parameter for Perlin noise (default=2.0).

        Returns
        -------
        np.ndarray
            Binary mask with irregular lesion contours.
        # """
        # Get initial contours from the lesion mask
        # test lesion mask is either bool or contains only 0 and 1
        if lesion_mask.dtype != np.bool_ and not np.array_equal(lesion_mask, lesion_mask.astype(bool)):
            raise ValueError("lesion_mask must be a binary mask of dtype bool or contain only 0 and 1 values.")

        torch_lesion_mask = torch.from_numpy(lesion_mask)
        if torch_lesion_mask.dim() == 3:
            torch_lesion_mask = torch_lesion_mask.unsqueeze(0)  # Add channel
            
        contours = AnomalyLabelTransform.get_contour3d(
                torch_lesion_mask,
                iterations=1,
                external=False
            ).numpy()
        
        # Get contours irregular with perlin noise
        for _ in range(max_growth):
            perlin_map = generate_perlin_map(
                    lesion_mask.squeeze().shape,
                    scale=scale,
                    octaves=octaves,
                    lacunarity=lacunarity,
                    base=random.randint(0, 100)
                )
            # Perlin on dilation
            dilated = binary_dilation(contours, iterations=dilation)
            # Add to contours where Perlin is above threshold and within dilation of current contours
            contours |= (perlin_map > perlin_threshold) & dilated 

        return contours
    
    @staticmethod
    def create_meshgrid(
        shape: Tuple[int, int, int],
        center: Tuple[int, int, int]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Create 3D coordinate grids centered at `center`.

        Parameters
        ----------
        shape : tuple of int
            Volume shape (z, y, x).
        center : tuple of int
            Center coordinates (z, y, x).

        Returns
        -------
        vz, vy, vx : torch.Tensor
            Coordinate grids for z, y, x shifted so that `center` is the origin.
        """
        z, y, x = torch.meshgrid(
            torch.arange(shape[0]),
            torch.arange(shape[1]),
            torch.arange(shape[2]),
            indexing="ij"
        )
        vz = z - center[0]
        vy = y - center[1]
        vx = x - center[2]

        return vz, vy, vx
    
    @staticmethod
    def generate_elliptical(
            shape: Tuple[int],
            center: Tuple[int],
            radii: Tuple[int],
            gradient: bool = False
        ) -> torch.Tensor:
        """
        Generate an elliptical shape based on the provided shape, center, and radii.
        
        Parameters
        ----------
        shape : Tuple[int]
            The shape of the tensor to create the elliptical mask for.

        center : Tuple[int]
            The center coordinates of the elliptical mask.

        radii : Tuple[int] 
            The radii of the elliptical mask in each dimension (z, y, x).
        
        Returns
        -------
        torch.Tensor
            A tensor of shape `shape` containing the elliptical mask.
            The mask will have values close to 1 inside the ellipse and close to 0 outside.
        """
        vz, vy, vx = AnomalyLabelTransform.create_meshgrid(shape, center)

        norm_dist = (vx / radii[2]) ** 2 + \
                    (vy / radii[1]) ** 2 + \
                    (vz / radii[0]) ** 2
        
        if gradient:
            elliptical = torch.exp(-norm_dist)
            elliptical = torch.where(elliptical > 0.1, elliptical, 0)
        else:
            elliptical = (norm_dist <= 1).float()

        return elliptical
    
    @staticmethod
    def sample_random_max_sizes(lesion_min_size: int, lesion_max_size: int) -> list:
        """
        Sample random maximum sizes for a lesion bounding box in 3D.

        The method samples a random maximum size for each dimension, using a Gaussian distribution
        centered to avoid generating too many small or large boxes.

        Parameters
        ----------
        lesion_min_size : int
            Minimum allowed size for the lesion in any dimension.
        lesion_max_size : int
            Maximum allowed size for the lesion in any dimension.

        Returns
        -------
        list of int
            List of three integers, each representing the randomly sampled maximum size for one dimension.
        """
        # std = (max - min) / 6 gives ~99.7% of values between min and max, with mean at the center
        # Goal is to avoid too many small or large boxes
        mean = (lesion_min_size + lesion_max_size) / 2
        std = (lesion_max_size - lesion_min_size) / 6

        random_max = int(random.gauss(mean, std))
        random_max = max(lesion_min_size, min(lesion_max_size, random_max))
        return [random.randint(lesion_min_size, random_max) for _ in range(3)]

    def crop_pure_target_box(
        label_tensor: torch.Tensor,
        point_on_skeleton: List[int],
        lesion_min_size: int,
        lesion_max_size: int,
        labels_to_target: List[int],
        random_max_sizes: bool = False,
        margin: float = 0,
    ):
        # Check if label_tensor is a 3D tensor
        if label_tensor.ndimension() == 4:
            label_tensor = label_tensor.squeeze(0)

        assert label_tensor.ndimension() == 3  # nosec B101

        # Get tensor operations
        device = label_tensor.device
        target_tensor = torch.tensor(labels_to_target, device=device)

        point = torch.tensor(point_on_skeleton, dtype=torch.long, device=device)
        shape = label_tensor.shape

        # Check if point is in the target labels
        if label_tensor[tuple(point)].item() not in labels_to_target:
            raise ValueError(f"Point not in {labels_to_target}")


        min_idx = point.clone()
        max_idx = point.clone() + 1
        def __is_valid_box(min_i, max_i, max_sizes, threshold=0.7):
            # Nested function to check if the box defined by min_i and max_i is valid based 
            # on the proportion of target labels it contains and its size compared to max_sizes.
            if lesion_min_size:
                for d in range(3):
                    if (max_i[d] - min_i[d]) > max_sizes[d]:
                        return False

            slices = tuple(slice(min_i[d], max_i[d]) for d in range(3))
            block = label_tensor[slices]
            return torch.isin(block, target_tensor).float().mean() > threshold
        
        can_grow_min = [True, True, True]
        can_grow_max = [True, True, True]

        if random_max_sizes:
            max_sizes = AnomalyLabelTransform.sample_random_max_sizes(lesion_min_size, lesion_max_size)
        else:
            max_sizes = [lesion_max_size] * 3

        logger.debug("%s[AnomalyLabel]%s Max sizes for box growth: %s%s%s", BLUE, RESET, BLUE, max_sizes, RESET)

        # Grow pure box until it reaches the target labels with a certain purity threshold, or until it reaches max sizes
        while any(can_grow_min) or any(can_grow_max):
            for dim in range(3):
                # Limit anisotropic growth:
                # - dim 1 ≤ 1.25 × size dim 0
                # - dim 2 ≤ 1.25 × min(size dim 0, dim 1)
                size_dim = max_idx[dim] - min_idx[dim]

                if dim in (1, 2):
                    s0 = max_idx[0] - min_idx[0]
                    ref = s0 if dim == 1 else min(s0, max_idx[1] - min_idx[1])

                    if size_dim > 1.25 * ref:
                        can_grow_min[dim] = False
                        can_grow_max[dim] = False
                        continue
                    
                # Expand box on negative side if within bounds and still valid
                if can_grow_min[dim]:
                    if min_idx[dim] > 0:
                        test_min = min_idx.clone()
                        test_min[dim] -= 1
                        if __is_valid_box(test_min, max_idx, max_sizes):
                            min_idx = test_min
                        else:
                            can_grow_min[dim] = False
                    else:
                        can_grow_min[dim] = False

                if can_grow_max[dim]:
                    if max_idx[dim] < shape[dim]:
                        test_max = max_idx.clone()
                        test_max[dim] += 1
                        if __is_valid_box(min_idx, test_max, max_sizes):
                            max_idx = test_max
                        else:
                            can_grow_max[dim] = False
                    else:
                        can_grow_max[dim] = False
                        
        if margin == 0:
            slices = tuple(slice(min_idx[d].item(), max_idx[d].item()) for d in range(3))
            return label_tensor[slices], min_idx.tolist(), max_idx.tolist()
        
        min_patch_size = 200
        # Compute margin
        box_size = [max_idx[d] - min_idx[d] for d in range(3)]

        margin_sizes = []
        for d in range(3):
            # Margin proportional to box size, with a minimum of 1 voxel
            m = int(max(box_size[d], 1) * margin)

            # Size to reach with margin
            size_with_margin = box_size[d] + 2 * m

            # If margin is too small to reach min_patch_size, increase it to ensure the final patch is at least min_patch_size
            if size_with_margin < min_patch_size:
                extra = (min_patch_size - size_with_margin) // 2
                m += extra + 1  # ensures >= 200

            margin_sizes.append(m)

        # Extended box (clamped)
        min_idx_margin = [
            max(0, min_idx[d] - margin_sizes[d])
            for d in range(3)
        ]
        max_idx_margin = [
            min(shape[d], max_idx[d] + margin_sizes[d])
            for d in range(3)
        ]
        # Crops
        slices_pure = tuple(slice(min_idx[d].item(), max_idx[d].item()) for d in range(3))
        slices_margin = tuple(slice(min_idx_margin[d], max_idx_margin[d]) for d in range(3))

        cropped_pure = label_tensor[slices_pure]
        cropped_margin = label_tensor[slices_margin]

        return (
            cropped_pure,
            cropped_margin,
            min_idx.tolist(),
            max_idx.tolist(),
            min_idx_margin,
            max_idx_margin,
        )

    def check_unique_background_value(self):
        """
        Check that the background_value attribute is a unique integer.
    
        If background_value is a list, ensures all values are the same and that the value is an integer.
        Raises a ValueError if multiple unique values are found or if the value is not an integer.
        Updates self.background_value to the unique integer value if checks pass.
    
        Raises
        ------
        ValueError
            If background_value contains multiple unique values or is not an integer.
        """
        if isinstance(self.background_value, list):
            unique_values = list(set(self.background_value))
            if len(unique_values) > 1:
                raise ValueError(f"background_value list contains multiple unique values: {unique_values}. Please provide a single value or ensure all values in the list are the same.")
            if type(unique_values[0]) != int:
                raise ValueError(f"background_value must be an integer or a list of integers. Got {type(unique_values[0])} in the list.")
            self.background_value = unique_values[0]
            
    def add_labels_names(self, subject: tio.Subject, lesion_label_names: dict):
        if "lesion_label_names" in subject:
            # Update lesion_label_names if it already exists
            subject["lesion_label_names"].update(lesion_label_names)
        else:
            subject["lesion_label_names"] = lesion_label_names
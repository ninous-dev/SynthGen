import concurrent.futures
import logging
import os
from typing import List, Optional, Tuple, Union
import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio
from torchio import Subject
import scipy.ndimage as ndi
from transforms_class.utils.perlin_utils import generate_perlin_map
from utils.logger import BLUE, RED, RESET, YELLOW
from scipy.ndimage import distance_transform_edt
from transforms_class.LabelTransform import LabelTransform

# Configure logging
logger = logging.getLogger(__name__)

class DeformationTransform(LabelTransform):
    """
    Elastic deformation (push / pull), TorchIO-compatible.

    Parameters
    ----------
    mode : str
        Mode of deformation, either "push" or "pull".

    interpolation : str, optional
        The interpolation method used during deformation. Default is 'linear'.

    spline_order : int, optional
        The order of the spline interpolation. Default is 3.

    displacement : Union[float, Tuple[float, float]], optional
        The displacement for deformation, either as a fixed value or a range (tuple)
        representing a percentage of grid_space. Default is 0.8.

    grid_space : Union[int, Tuple[int, int]], optional
        Defines the spacing of the deformation grid.
        If an integer is provided, a fixed grid spacing is used.
        If a tuple is provided, a random integer within the specified range (inclusive)
        is selected for grid spacing. Default is 10.

    labels_to_keep : Optional[List[int]], optional
        Specifies a list of labels to retain during processing.
        If None, all labels are considered. Default is None.

    labels_to_target : Optional[List[int]], optional
        Specifies a list of target labels for the deformation.
    
    locked_borders : int, optional
        Number of voxels to lock at the borders of the image. Default is 3.

    background_value : float, optional
        The background value to use during resampling. Default is 0.
        
    flip_axes : Optional[List[int]], optional
        Axes to flip (0 for RL, 1 for AP, 2 for SI). Default is None.
    
    inverse_axes : Optional[List[int]], optional
        Axes to invert direction (0 for RL, 1 for AP, 2 for SI). Default is None.

    random_field_mask : str, optional
        Whether to apply a random mask to the deformation field. Default is False.

    target_point : Tuple[int], optional
        Target point (Z, Y, X) in full image space coordinates toward which the deformation field should point.
        
    attention_mask: Optional[Tensor], optional
        A binary mask (Tensor) in the same space as the input image, where the deformation field should be applied.
        If defined it replace labels_to_target and is used to compute the distance transform for the deformation field direction.
        Default is None.
    
    **kwargs
        Additional keyword arguments passed to the base class constructor

    """
    def __init__(
        self,
        mode: str,
        labels_to_target: List[int],
        interpolation: str = "nearest",
        spline_order: int = 3,
        displacement: Union[float, Tuple[float, float]] = 0.8,
        grid_space: Union[int, Tuple[int, int]] = 10,
        labels_to_keep: Optional[List[int]] = None,
        locked_borders: int = 3,
        background_value: float = 0,
        flip_axes: Optional[List[int]] = None,
        inverse_axes: Optional[List[int]] = None,
        random_field_mask: Optional[str] = None,
        target_point: Tuple[int] = None,
        keep_dilation: int = 2,
        perlin_unidirectional_threshold: float = 0.25,
        attention_mask: Optional[tio.LabelMap] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if mode not in {"push", "pull"}:
            raise ValueError("mode must be 'push' or 'pull'")

        self.mode = mode
        self.labels_to_keep = labels_to_keep
        self.labels_to_target = labels_to_target
        self.locked_borders = locked_borders

        self.grid_space = self._resolve_grid_space(grid_space)
        self.displacement = self._resolve_displacement(displacement)
        self.spline_order = spline_order
        self.interpolator = self.get_sitk_interpolator(interpolation)
        self.background_value = background_value
        self.flip_axes = flip_axes
        self.inverse_axes = inverse_axes
        self.random_field_mask = random_field_mask
        self.attention_mask = attention_mask
        
        self.keep_dilation = keep_dilation
        
        self.perlin_unidirectional_threshold = perlin_unidirectional_threshold

        if self.random_field_mask not in {None, "bidirectional", "unidirectional", "perlin_unidirectional", "perlin_unidirectional_thresholded"}:
            raise ValueError("random_field_mask must be None, 'bidirectional', 'unidirectional', 'perlin_unidirectional', or 'perlin_unidirectional_thresholded'")
        
        self.target_point = target_point
        
        self.check_target_keep_labels_consistency()

    def check_target_keep_labels_consistency(self):
        if self.labels_to_keep is not None and self.labels_to_target is not None:
            overlap = set(self.labels_to_keep) & set(self.labels_to_target)
            if overlap:
                raise ValueError(f"labels_to_keep and labels_to_target cannot have overlapping labels. Overlap: {overlap}")

    def log_transform_info(self):
        logger.debug("%s[DEFORM]%s mode = %s%s%s", BLUE, RESET, YELLOW, self.mode, RESET)
        logger.debug("%s[DEFORM]%s labels_to_target = %s%s%s", BLUE, RESET, YELLOW, self.labels_to_target, RESET)
        logger.debug("%s[DEFORM]%s labels_to_keep = %s%s%s", BLUE, RESET, YELLOW, self.labels_to_keep, RESET)
        logger.debug("%s[DEFORM]%s locked_borders = %s%s%s", BLUE, RESET, YELLOW, self.locked_borders, RESET)
        logger.debug("%s[DEFORM]%s grid_space = %s%s%s", BLUE, RESET, YELLOW, self.grid_space, RESET)
        logger.debug("%s[DEFORM]%s displacement = %s%s%s", BLUE, RESET, YELLOW, self.displacement, RESET)
        logger.debug("%s[DEFORM]%s background_value = %s%s%s", BLUE, RESET, YELLOW, self.background_value, RESET)
        logger.debug("%s[DEFORM]%s flip_axes = %s%s%s", BLUE, RESET, YELLOW, self.flip_axes, RESET)
        logger.debug("%s[DEFORM]%s inverse_axes = %s%s%s", BLUE, RESET, YELLOW, self.inverse_axes, RESET)
        logger.debug("%s[DEFORM]%s random_field_mask = %s%s%s", BLUE, RESET, YELLOW, self.random_field_mask, RESET)
        logger.debug("%s[DEFORM]%s target_point = %s%s%s", BLUE, RESET, YELLOW, self.target_point, RESET)
        logger.debug("%s[DEFORM]%s attention_mask = %s%s%s", BLUE, RESET, YELLOW, self.attention_mask is not None, RESET)
        logger.debug("%s[DEFORM]%s keep_dilation = %s%s%s", BLUE, RESET, YELLOW, self.keep_dilation, RESET)
        logger.debug("%s[DEFORM]%s perlin_unidirectional_threshold = %s%s%s", BLUE, RESET, YELLOW, self.perlin_unidirectional_threshold, RESET)

    def is_invertible(self):
        """
        Indicates whether the transform is invertible.
        """
        return False

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @staticmethod
    def to_tensor(data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Convert input data to a torch.Tensor if it is not already.

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            Input data, either a NumPy array or a torch.Tensor.

        Returns
        -------
        torch.Tensor
            The input data as a torch.Tensor.
        """
        if isinstance(data, torch.Tensor):
            return data

        return torch.from_numpy(data).float()
    
    def get_target_mask(self, sitk_np: np.ndarray) -> np.ndarray:
        """
        Get the target mask based on the input SimpleITK image and the specified labels or target area.

        Parameters
        ----------
        sitk_np : np.ndarray
            The input image as a NumPy array.

        Returns
        -------
        np.ndarray
            A binary mask indicating the target areas for deformation.
        """
        target_mask = np.isin(sitk_np, self.labels_to_target).astype(np.uint8)
        if self.attention_mask is not None:
            sitk_attention_mask = tio.io.nib_to_sitk(
                self.attention_mask.data,
                self.attention_mask.affine,
                force_3d=True,
            )
            sitk_attention_mask_np = sitk.GetArrayFromImage(sitk_attention_mask)
            return target_mask, sitk_attention_mask_np.astype(np.uint8)
        return target_mask, None

    def update_keep_mask(self, sitk_np: np.ndarray, target_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Update the keep mask based on the input image and the specified labels to keep.
    
        If `labels_to_keep` is provided, creates a binary mask for those labels, optionally dilates it,
        and updates the target mask to exclude the keep region. Returns the updated target mask and the keep mask.
        If no labels to keep are specified, returns the original target mask and None.
    
        Parameters
        ----------
        sitk_np : np.ndarray
            The input image as a NumPy array.
        target_mask : np.ndarray
            The current binary target mask.
    
        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            The updated target mask and the keep mask (or None if not used).
        """
        if self.labels_to_keep is not None:
            keep_mask = np.isin(sitk_np, self.labels_to_keep).astype(np.uint8)
            if self.keep_dilation > 0:
                # Dilate the keep_mask to ensure a larger area is preserved
                keep_mask = ndi.binary_dilation(
                    keep_mask,
                    iterations=self.keep_dilation
                ).astype(np.uint8)
            target_mask = target_mask & (~keep_mask.astype(bool))
            
            return target_mask, keep_mask
        
        return target_mask, None

    def apply_random_field_mask(self, field):
        """
        Apply a random field mask to the deformation field according to the selected mode.

        Parameters
        ----------
        field : np.ndarray
            The deformation field to be masked (shape: [..., 3]).

        Returns
        -------
        np.ndarray
            The masked deformation field.
        """
        random_factors = None
        if self.random_field_mask == "bidirectional":
            random_factors = np.random.uniform(-1, 1, size=(*field.shape[:-1], 3)).astype(np.float32)
        elif self.random_field_mask == "unidirectional":
            random_factors = np.random.uniform(0, 1, size=(*field.shape[:-1], 3)).astype(np.float32)
        elif self.random_field_mask in ["perlin_unidirectional", "perlin_unidirectional_thresholded"]:
            perlin_map = generate_perlin_map(
                field.shape[:-1],
                scale=20,
                octaves=4,
                lacunarity=2.0,
                base=np.random.randint(0, 100)
            )
            if self.random_field_mask == "perlin_unidirectional_thresholded":
                perlin_map[perlin_map < self.perlin_unidirectional_threshold] = 0
            random_factors = (perlin_map - perlin_map.min()) / (perlin_map.max() - perlin_map.min() + 1e-8)
            random_factors = random_factors[..., None]
        if random_factors is not None:
            field = field * random_factors
        return field
    
    def get_target_keep_mask(self, sitk_image: sitk.Image) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieve the target mask, attention mask, and keep mask for deformation.
    
        Parameters
        ----------
        sitk_image : sitk.Image
            The input image as a SimpleITK Image.
    
        Returns
        -------
        target_mask : np.ndarray
            Binary mask indicating the target areas for deformation.
        attention_mask_np : Union[np.ndarray, None]
            Numpy array of the attention mask if defined, otherwise None.
        keep_mask : Union[np.ndarray, None]
            Binary mask of the regions to keep (excluded from deformation), or None if not used.
        """
        sitk_np = sitk.GetArrayFromImage(sitk_image)
        target_mask, attention_mask_np = self.get_target_mask(sitk_np)
        target_mask, keep_mask = self.update_keep_mask(sitk_np, target_mask)
        
        return target_mask, attention_mask_np, keep_mask
    
    def get_radial_mask(self, target_mask, outside_max_value=1.0):
        """
        Compute a radial mask based on the signed distance to the target mask boundary.
    
        For each voxel, the value is 1 at the boundary of the target mask, decreases toward the center inside the mask,
        and decreases toward the exterior outside the mask (scaled by 'outside_max_value').
    
        Parameters
        ----------
        target_mask : np.ndarray
            Binary mask indicating the target region.
        outside_max_value : float, optional
            Maximum value for the radial mask outside the target region (default=1.0).
    
        Returns
        -------
        np.ndarray
            Radial mask of the same shape as `target_mask`, with values in [0, 1].
        """
        # Compute amplitude of the deformation
        # Distance weight FROM combined mask
        dist = sitk.GetArrayFromImage(
            sitk.SignedMaurerDistanceMap(
                sitk.GetImageFromArray(target_mask.astype(np.uint8)),
                insideIsPositive=True,
                squaredDistance=False,
            )
        )
        radial = np.ones_like(dist, dtype=np.float32)

        # Change amplitude:
        # inside target mask: max at boundary, decreasing toward center
        # outdside target mask: max at boundary, decreasing toward exterior (can be limited  with outside_max_value)
        outside_mask = dist < 0
        if np.any(outside_mask):
            d_min = dist[outside_mask].min()
            radial[outside_mask] = outside_max_value * (dist[outside_mask] - d_min) / (-d_min)
            
        return radial
    
    @staticmethod
    def pool_to_grid(array: np.ndarray, grid_space: int, reduction: str = "mean") -> np.ndarray:
        """
        Pool a 3D array to the control grid resolution.

        Parameters
        ----------
        array : np.ndarray
            Input 3D array to pool.
        grid_space : int
            Size of the pooling window along each axis.
        reduction : str
            Reduction method: "mean" or "max".

        Returns
        -------
        np.ndarray
            Pooled array of shape (Dg, Hg, Wg).
        """
        Dg, Hg, Wg = [dim // grid_space for dim in array.shape]
        arr_cropped = array[:Dg*grid_space, :Hg*grid_space, :Wg*grid_space]
        arr_reshaped = arr_cropped.reshape(
            Dg, grid_space,
            Hg, grid_space,
            Wg, grid_space,
        )
        if reduction == "mean":
            pooled = arr_reshaped.mean(axis=(1, 3, 5))
        elif reduction == "max":
            pooled = arr_reshaped.max(axis=(1, 3, 5))
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        return pooled.astype(np.float32)
    
    @staticmethod
    def normalize_field(field, positive_only=False):
        """
        Normalize a vector field along the last axis.

        Parameters
        ----------
        field : np.ndarray
            The vector field to normalize (..., 3).
        positive_only : bool
            If True, negative values are set to zero after normalization.

        Returns
        -------
        np.ndarray
            The normalized field.
        """
        norm = np.linalg.norm(field, axis=-1, keepdims=True)
        normalized = field / np.maximum(norm, 1e-8)
        if positive_only:
            normalized = np.clip(normalized, 0, None)
        return normalized

    def get_radial_weight_and_direction(
        self,
        target_mask: np.ndarray,
        keep_mask: Optional[np.ndarray],
        grid_shape: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Dg, Hg, Wg = grid_shape

        signed_dist = sitk.GetArrayFromImage(
            sitk.SignedMaurerDistanceMap(
                sitk.GetImageFromArray(target_mask.astype(np.uint8)),
                insideIsPositive=True,
                squaredDistance=False,
            )
        )

        dist_positive = np.maximum(signed_dist, 0)
        dist_norm = dist_positive / (dist_positive.max() + 1e-8)

        if self.mode == "push":
            weight = np.where(dist_norm != 0, 1 - dist_norm, 0)
        else:
            # pull
            weight = dist_norm

        if keep_mask is not None:
            weight[keep_mask.astype(bool)] = 0

        raw_z, raw_y, raw_x = np.gradient(signed_dist)
        
        def _crop_pool(arr: np.ndarray) -> np.ndarray:
            arr = arr[: Dg * self.grid_space,
                    : Hg * self.grid_space,
                    : Wg * self.grid_space]
            return DeformationTransform.pool_to_grid(arr, self.grid_space, reduction="mean")

        return weight, _crop_pool(raw_z), _crop_pool(raw_y), _crop_pool(raw_x)
    

    @staticmethod
    def _field_to_point_fullres(
        target_mask_shape: Tuple[int, int, int],
        target_point: Tuple[int, int, int]
    )-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the direction field from every voxel in the target mask to a given target point.
    
        For each voxel, returns the vector components (z, y, x) pointing from the voxel to the specified target point.
    
        Parameters
        ----------
        target_mask_shape : tuple of int
            Shape of the target mask (D, H, W).
        target_point : tuple of int
            Coordinates (x, y, z) of the target point.
    
        Returns
        -------
        grad_z : np.ndarray
            Array of z-components of the direction vectors.
        grad_y : np.ndarray
            Array of y-components of the direction vectors.
        grad_x : np.ndarray
            Array of x-components of the direction vectors.
        """
        D, H, W = target_mask_shape
        Z, Y, X = np.indices((D, H, W), dtype=np.float32)
        tx, ty, tz = target_point

        # Vectors: target - voxel (point to target)
        grad_z = tz - Z
        grad_y = ty - Y
        grad_x = tx - X

        return grad_z, grad_y, grad_x
    

    def _radial_field_fullres(self, target_mask: np.ndarray):
        """
        Compute the normalized gradient field (direction vectors) of the signed distance map to the target mask boundary.
    
        For each voxel, computes the gradient of the signed distance map, resulting in a vector field pointing outward from the mask boundary.
        The gradients are normalized to unit vectors.
    
        Parameters
        ----------
        target_mask : np.ndarray
            Binary mask indicating the target region.
    
        Returns
        -------
        grad_z : np.ndarray
            Normalized z-component of the gradient field.
        grad_y : np.ndarray
            Normalized y-component of the gradient field.
        grad_x : np.ndarray
            Normalized x-component of the gradient field.
        """
        dist_img = sitk.SignedMaurerDistanceMap(
            sitk.GetImageFromArray(target_mask.astype(np.uint8)),
            insideIsPositive=True,
            squaredDistance=False,
        )
        gradient_img = sitk.Gradient(dist_img) 
        gradient = sitk.GetArrayFromImage(gradient_img)

        grad_x, grad_y, grad_z = np.split(gradient, 3, axis=-1)
        grad_x = grad_x[..., 0]
        grad_y = grad_y[..., 0]
        grad_z = grad_z[..., 0]

        # Normalization
        norm = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        norm[norm == 0] = 1.0
        grad_x /= norm
        grad_y /= norm
        grad_z /= norm

        return grad_z, grad_y, grad_x
    
    def get_direction_field(self, target_mask: np.ndarray):
        """
        Compute the direction field for the target mask at the control grid resolution.
    
        If a target point is specified, computes the direction vectors from each voxel to the target point.
        Otherwise, computes the normalized gradient field of the signed distance map to the target mask boundary.
        The resulting field is pooled to the control grid resolution, normalized, and optionally modulated by a radial mask.
    
        Parameters
        ----------
        target_mask : np.ndarray
            Binary mask indicating the target region.
    
        Returns
        -------
        np.ndarray
            The direction field at the control grid resolution, shape (Wg, Hg, Dg, 3), normalized and modulated.
        """
        # Compute direction field at full resolution
        if self.target_point is not None:
            grad_z_full, grad_y_full, grad_x_full = self._field_to_point_fullres(
                target_mask.shape, self.target_point
            )
        else:
            if np.sum(target_mask) == 0:
                raise ValueError("No target voxels found. Check labels_to_target and input image.")
            grad_z_full, grad_y_full, grad_x_full = self._radial_field_fullres(
                target_mask
            )

        # Pool gradients to control grid resolution
        grad_z = DeformationTransform.pool_to_grid(grad_z_full, self.grid_space, reduction="mean")
        grad_y = DeformationTransform.pool_to_grid(grad_y_full, self.grid_space, reduction="mean")
        grad_x = DeformationTransform.pool_to_grid(grad_x_full, self.grid_space, reduction="mean")

        field = np.stack([grad_x, grad_y, grad_z], axis=-1).astype(np.float32)
        field = DeformationTransform.normalize_field(field)

        # Distance 
        if len(np.unique(target_mask)) > 1:
            dist = self.get_radial_mask(target_mask, outside_max_value=0.5)
            dist_pooled = DeformationTransform.pool_to_grid(dist, self.grid_space, reduction="mean")

            field *= dist_pooled[..., None]

        return field

    def generate_control_points_grid(self, sitk_image: sitk.Image) -> np.ndarray:
        """
        Generate the deformation control points grid for a given image.
    
        This method computes the direction field for the target mask at the control grid resolution,
        applies optional attention and keep masks, random field masking, border locking, and scaling.
        The resulting field is transposed and axes are optionally flipped or inverted as specified.
    
        Parameters
        ----------
        sitk_image : sitk.Image
            The input image as a SimpleITK Image.
    
        Returns
        -------
        np.ndarray
            The deformation control points grid, ready for use in B-spline deformation.
        """
        # Retrieve target, keep mask and attention masks
        target_mask, attention_mask_np, keep_mask = self.get_target_keep_mask(sitk_image)
        field = self.get_direction_field(target_mask)
            
        if self.attention_mask is not None:
            # Change the field amplitude where the attention mask is defined
            attention_grid = DeformationTransform.pool_to_grid(attention_mask_np, self.grid_space, reduction="max")
            # Apply a Gaussian decay based on the distance transform of the attention grid
            dist_attention = distance_transform_edt(attention_grid == 0)
            dist_attention = np.exp(-0.4 * dist_attention ** 2)

            field *= dist_attention[..., None]

        if keep_mask is not None:
            # Set field to zero in the keep regions
            keep_grid = DeformationTransform.pool_to_grid(keep_mask, self.grid_space, reduction="max")
            field[keep_grid.astype(bool)] = 0

        if self.random_field_mask:
            # Apply random mask to the deformation field
            field = self.apply_random_field_mask(field)
            field = DeformationTransform.normalize_field(field)

        # Lock borders of the field to zero
        field = self.lock_borders(field, self.locked_borders)

        # Apply displacement
        field = field * self.displacement
        
        # Revert direction for pull mode
        if self.mode == "pull":
            field *= -1.0

        # Axes gestion
        field = np.transpose(field, (2, 1, 0, 3))

        if self.flip_axes is not None:
            field = np.flip(field, axis=self.flip_axes)

        if self.inverse_axes is not None:
            for axe in self.inverse_axes:
                field[..., axe] *= -1
        return field.astype(np.float32)

    @staticmethod
    def lock_borders(tensor: torch.Tensor, locked: int) -> torch.Tensor:
        """
        Lock the borders of a tensor by setting them to zero, preserving the inner region.

        Parameters
        ----------
        tensor : torch.Tensor
            The input tensor to modify.

        locked : int
            The number of voxels to lock (set to zero) at the borders.

        Returns
        -------
        torch.Tensor
            The tensor with locked borders.
        """
        if locked > 0:
            if isinstance(tensor, torch.Tensor):
                # Extract the inner region
                inner = tensor[
                    locked : -locked,
                    locked : -locked,
                    locked : -locked,
                ].clone()

                # Zero out the whole tensor
                tensor.zero_()

                # Restore the inner region
                tensor[
                    locked : -locked,
                    locked : -locked,
                    locked : -locked,
                ] = inner

            elif isinstance(tensor, np.ndarray):
                inner = tensor[
                    locked : -locked,
                    locked : -locked,
                    locked : -locked,
                ].copy()

                tensor.fill(0)

                tensor[
                    locked : -locked,
                    locked : -locked,
                    locked : -locked,
                ] = inner

        return tensor

    def apply_bspline_deformation(
        self,
        image: sitk.Image,
        control_points: np.ndarray,
    ) -> tio.LabelMap:
        """
        Apply a B-spline deformation to the input image using the provided control points.

        Handles both scalar and multi-component images.
        
        Parameters
        ----------
        image : sitk.Image
            The input image to be deformed.

        control_points : np.ndarray
            Array of B-spline control point parameters.

        Returns
        -------
        tio.LabelMap
            The deformed image as a TorchIO LabelMap.

        """
        nb_components = image.GetNumberOfComponentsPerPixel()

        # Deform each component if multi-component, else deform scalar image
        if nb_components > 1:

            def deform_component(index: int) -> sitk.Image:
                sub_image = sitk.VectorIndexSelectionCast(image, index)
                transform = self.initialize_transform(
                    image, control_points, self.spline_order
                )
                return self.resample_image(
                    sub_image,
                    transform,
                    image,
                    self.interpolator,
                    background_value=self.background_value
                )

            with concurrent.futures.ThreadPoolExecutor() as executor:
                components = list(
                    executor.map(deform_component, range(nb_components))
                )

            deformed = sitk.Compose(components)

        else:
            transform = self.initialize_transform(
                image, control_points, self.spline_order
            )
            deformed = self.resample_image(
                image,
                transform,
                image,
                self.interpolator,
                background_value=self.background_value
            )

        tensor, affine = self.sitk_to_nib(deformed)
        return tio.LabelMap(tensor=tensor, affine=affine)

    def _check_image_size_vs_grid(self, sitk_image: sitk.Image) -> None:
        """
        Ensure the image is large enough for the chosen grid_space
        and spline_order.

        Raises
        ------
        ValueError
            If the image is too small for the current grid configuration.
        """
        size = sitk_image.GetSize()  # (X, Y, Z)
        min_dim = min(size)

        min_required_points = self.spline_order + 1
        max_allowed_grid = min_dim // min_required_points

        if self.grid_space > max_allowed_grid:
            raise ValueError(
                f"Image too small for grid_space={self.grid_space}. "
                f"Minimum dimension: {min_dim}. "
                f"Spline order: {self.spline_order}. "
                f"Maximum allowed grid_space: {max_allowed_grid}."
            )

    # Sitk deformation methods
    @staticmethod
    def initialize_transform(
        image_ref: sitk.Image,
        control_points: np.ndarray,
        spline_order: int
    ) -> sitk.BSplineTransform:
        """
        Initialize a B-spline transform using the provided control points.

        This method creates a B-spline transform for the given reference image and control points.
        The control points define the deformation grid, and the spline order determines the
        smoothness of the deformation.

        Parameters
        ----------
        image_ref : sitk.Image
            The reference image used to initialize the B-spline transform.

        control_points : np.ndarray
            A NumPy array representing the control points for the B-spline transform.

        spline_order : int
            The order of the B-spline interpolation. Higher orders result in smoother deformations.

        Returns
        -------
        sitk.BSplineTransform
            The initialized B-spline transform.
        """
        # mesh size must be number of grid cells
        mesh_size = [
            int(s - spline_order)
            for s in control_points.shape[:3]
        ]
        
        if any(m <= 0 for m in mesh_size):
            raise ValueError(
                f"Invalid mesh_size {mesh_size} from control_points.shape "
                f"{control_points.shape} and spline_order {spline_order}"
            )

        # mesh_size = np.subtract(control_points.shape[:3], spline_order).tolist()
        transform = sitk.BSplineTransformInitializer(image_ref, mesh_size, order=spline_order)
        transform.SetParameters(
            control_points.flatten(order="F").astype(np.float64).tolist()
        )
        return transform

    @staticmethod
    def resample_image(
        image: sitk.Image,
        transform: sitk.Transform,
        reference: sitk.Image,
        interpolator: Union[int, str],
        background_value: float = 0,
    ) -> sitk.Image:
        """
        Resample a SimpleITK image using a given transform and reference image.
        
        Parameters
        ----------
        image : sitk.Image
            The input image to be resampled.

        transform : sitk.Transform
            The geometric transform to apply during resampling.

        reference : sitk.Image
            The reference image defining output space, spacing, origin, and direction.

        interpolator : Union[int, str]
            Interpolator to use for resampling. Can be one of the following:
            - `sitk.sitkNearestNeighbor`
            - `sitk.sitkLinear`
            - `sitk.sitkBSpline`
            - or other SimpleITK interpolation methods.

        background_value : float, optional
            The default pixel value for areas outside the input image. Default is 50.

        Returns
        -------
        sitk.Image
            The resampled image with type `sitk.sitkFloat32` and default pixel value 50.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(background_value)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetTransform(transform)
        resampler.SetReferenceImage(reference)

        return resampler.Execute(image)

    # Resolution methods for parameters
    def _resolve_grid_space(self, grid_space: Union[int, Tuple[int, int]]) -> int:
        """
        Resolve grid_space to a valid integer value.

        Parameters
        ----------
        grid_space : Union[int, Tuple[int, int]]
            Either a fixed integer or a range (tuple) to randomly select from.

        Returns
        -------
        int
            Resolved grid_space value.
        """
        # is list or tuple
        if isinstance(grid_space, (list, tuple)):
            if len(grid_space) != 2:
                raise ValueError(f"If grid_space is a list or a tuple it must have exactly 2 elements (got {grid_space})")
            
            grid_space = tuple(grid_space)
            value = np.random.randint(grid_space[0], grid_space[1] + 1)
        else:
            value = grid_space

        if value <= 0:
            raise ValueError(f"grid_space must be positive and got {value}")

        return value

    def _resolve_displacement(self, displacement: Union[float, Tuple[float, float]]) -> float:
        """
        Resolve displacement as a percentage of grid_space.

        Parameters
        ----------
        displacement : Union[float, Tuple[float, float]]
            Either a fixed percentage or a range (tuple) to randomly select from.

        Returns
        -------
        float
            Calculated displacement value.
        """
        if isinstance(displacement, list):
            if len(displacement) != 2:
                raise ValueError(f"displacement list must have exactly 2 elements (got {len(displacement)} with values {displacement})")
            displacement = tuple(displacement)

        if isinstance(displacement, tuple):
            value = np.random.uniform(*displacement) * self.grid_space
        else:
            value = displacement * self.grid_space

        if value <= 0:
            raise ValueError(f"displacement ({value}) must be in the range [0, {self.grid_space}] [0, grid_space]")

        return value
    
    @classmethod
    def apply_to_tensor(cls, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        is_3d = tensor.dim() == 3
        if is_3d:
            tensor = tensor.unsqueeze(0)

        subject = tio.Subject(label=tio.LabelMap(tensor=tensor))
        result = cls(**kwargs)(subject)["label"].data

        if is_3d:
            result = result.squeeze(0)
        return result

    # Apply transform method
    def apply_transform(self, subject: Subject) -> Subject:
        """
        apply_transform

        Parameters
        ----------
        subject : Subject
            The subject object.

        Returns
        -------
        Subject
            The subject object transformed.

        """
        self.log_transform_info()
        for image in self.get_images(subject):
            sitk_image = tio.io.nib_to_sitk(
                image.data,
                image.affine,
                force_3d=True,
            )
            self._check_image_size_vs_grid(sitk_image)

            # Generate deformation control point
            control_points = self.generate_control_points_grid(
                sitk_image
            )

            # Apply B-spline deformation
            deformed = self.apply_bspline_deformation(
                sitk_image,
                control_points,
            )
            if torch.equal(deformed.data, image.data):
                raise ValueError("Deformation resulted in identical image data. Check parameters and input image. Consider adjusting grid_space, displacement, or label configuration.")

            image.set_data(deformed.data)

        return subject

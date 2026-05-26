import torch
from typing import Tuple, Union, List

from scipy.ndimage import generate_binary_structure, label
from skimage.measure import regionprops

def compute_coverage(
    label_tensor: torch.Tensor,
    lesion_mask: torch.Tensor,
) -> tuple:
    """
    Compute a coverage score for a lesion mask within a label tensor, with penalties
    for overlap with invalid labels or areas outside any label.

    The coverage score is the fraction of lesion voxels lying inside label '1'.

    Parameters
    ----------
    label_tensor : torch.Tensor
        Integer labels volume.
    lesion_mask : torch.Tensor
        Binary mask (same shape).

    Returns
    -------
    float
        Proportion in [0, 1].
    """
    if label_tensor.shape != lesion_mask.shape:
        raise RuntimeError(
            f"Shape mismatch: label_tensor {label_tensor.shape} vs lesion_mask {lesion_mask.shape}"
        )

    lesion_voxels = lesion_mask > 0
    total_lesion_voxels = torch.sum(lesion_voxels).item()
    if total_lesion_voxels == 0:
        return 0.0, 1.0  # No lesion voxels = fully outside

    inside = lesion_voxels & (label_tensor == 1)
    inside_proportion = torch.sum(inside).item() / total_lesion_voxels

    return inside_proportion

def get_random_position_in_label(
        label_tensor: torch.Tensor,
        target_labels: Union[int, List[int]],
        step: int = 1
    ) -> List[int]:
    """
    Select a random position in a tensor where the entries equal the specified target label.

    Parameters
    ----------
    label_tensor : torch.Tensor
        Tensor of integer labels, e.g. a 3D volume of voxel labels.

    target_labels: Union[int, List[int]]
        Labels in which lesions will be inserted.

    Returns
    -------
    List[int]
        A list of integer coordinates (e.g. [i, j, k]) corresponding to a randomly chosen
        position whose value in `label_tensor` equals `target_label`.

    Raises
    ------
    ValueError
        If no element in `label_tensor` matches `target_label`.
    """
    if isinstance(target_labels, int):
        target_labels = [target_labels]

    # Create a boolean mask where label_tensor matches any of the target_labels
    mask = torch.isin(label_tensor, torch.tensor(target_labels))
    # Get the indices of all matching voxels
    indices = mask.nonzero(as_tuple=False)

    # Raise an error if no matching voxel is found
    if indices.numel() == 0:
        raise ValueError(f"No voxel with labels {target_labels} found.")

    # Subsample the indices to reduce the number of candidates (every 10th voxel)
    step = 10
    indices = indices[::step]

    # Randomly select one index from the subsampled candidates
    rand_idx = torch.randint(0, indices.shape[0], (1,))
    position = indices[rand_idx].squeeze(0)

    return position.tolist()

def get_biggest_component(binary_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract the largest connected component from a binary tensor.

    Parameters
    ----------
    binary_tensor : torch.Tensor
        A binary mask tensor (dtype=torch.bool or containing 0/1 values) of shape
        (D, H, W) or similar..

    Returns
    -------
    torch.Tensor
        A boolean tensor of the same shape as `binary_tensor`, with True for voxels
        belonging to the largest connected component and False elsewhere.
    """
    
    structure = generate_binary_structure(rank=3, connectivity=3)
    labeled_array, num_features = label(binary_tensor, structure=structure)

    component_mask = torch.tensor(labeled_array == max(regionprops(labeled_array), key=lambda x: x.area).label)
    return component_mask

def compute_anatomical_resistance(
    labels_tensor: torch.Tensor,
    wm_label: int,
    start: Tuple[int, int, int],
    end: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Compute a resistance map based on anatomical tissue labels, then crop to a subregion.

    Parameters
    ----------
    labels_tensor : torch.Tensor
        A 3D integer tensor where each voxel's value indicates a tissue label.

    wm_label : int
        The integer label corresponding to white matter in `labels_tensor`.

    start : tuple of int
        Inclusive start coordinates of the subregion to extract, as (x_start, y_start, z_start).

    end : tuple of int
        Exclusive end coordinates of the subregion to extract, as (x_end, y_end, z_end).

    Returns
    -------
    torch.Tensor
        A float32 tensor of shape
        `(end[0]-start[0], end[1]-start[1], end[2]-start[2])` containing voxelwise resistances:
        
        - White matter (`wm_label`): 0.1  
        - Gray matter (label 10): 0.3  
        - CSF (label 32): 0.01  
        - All other labels: 1.0  
    """
    # TODO work on it to make it work or remove it since it is not used for now, it is supposed to give a resistance map based on the anatomical labels, with low resistance in WM and high resistance in other tissues, but it is not working as expected for now
    # Easy propagation == 0.1 hard propagation == 1
    gm_label = 10
    csf_label = 32

    resistance_map = torch.ones_like(labels_tensor, dtype=torch.float32)
    resistance_map[labels_tensor != wm_label] = 1.0

    resistance_map[labels_tensor == wm_label] = 0.1
    resistance_map[labels_tensor == gm_label] = 0.3
    resistance_map[labels_tensor == csf_label] = 0.01

    resistance_crop = resistance_map[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2]
    ]
    return resistance_crop
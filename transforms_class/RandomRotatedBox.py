"""
RandomRotatedCuboid Transform
Extract rotated random regions from TorchIO subjects using Affine.
"""

import logging
import os
import numpy as np
import torch
import torchio as tio

from utils.logger import BLUE, RESET, configure_logging

logger = logging.getLogger(__name__)


class RandomRotatedBox(tio.SpatialTransform):
    """
    Extract a randomly positioned and rotated box region from all images in a TorchIO subject.
    
    Creates a binary box, rotates it with Affine, and uses it as mask
    to extract the same region from all images in the subject.
    
    Parameters
    ----------
    box_size_reduction_range : tuple of float, optional
        Box size as percentage of min dimension (0.0-1.0). Default: (0.2, 0.5)
    rotation_range : tuple of float, optional
        Rotation angle range in degrees. Default: (-180, 180)
    """
    
    def __init__(
        self,
        box_size_reduction_range: tuple = (0.2, 0.5),
        rotation_range: tuple = (-180, 180),
        **kwargs
    ):
        super().__init__(**kwargs)
        configure_logging(dev_mode=True, module_name=__name__)
        
        self.box_size_reduction_range = box_size_reduction_range
        self.rotation_range = rotation_range
        
        self.volume_size = None
        self.box_mask = None
        self.box_position = None
        
        self.args_names = [
            'box_size_reduction_range',
            'rotation_range',
        ]
        
    def __random_volume_size(self):
        """Return a random value in the given range."""
        self.volume_size = [round(
                (1 - np.random.uniform(
                    self.box_size_reduction_range[0],
                    self.box_size_reduction_range[1]))
                * dim) for dim in self.image_size
        ]
        logger.info("%s[RandomRotatedBox]%s Random volume size: %s%s%s", BLUE, RESET, self.volume_size, BLUE, RESET)

    def __random_box_position(self):
        """Generate random starting position for the box."""
        d, h, w = self.image_size
        box_d, box_h, box_w = self.volume_size

        # Random position pour chaque dimension
        d_start = np.random.randint(0, max(1, d - box_d + 1))
        h_start = np.random.randint(0, max(1, h - box_h + 1))
        w_start = np.random.randint(0, max(1, w - box_w + 1))

        self.box_position = [d_start, h_start, w_start]
        print(f"Random box position: {self.box_position}")

    def create_binary_box(self):
        """Create binary box image at random position."""
        d, h, w = self.image_size
        box_data = torch.zeros(1, d, h, w, dtype=torch.float32)

        # Get random starting position
        self.__random_box_position()
        d_start, h_start, w_start = self.box_position
        d_end = min(d, d_start + self.volume_size[0])
        h_end = min(h, h_start + self.volume_size[1])
        w_end = min(w, w_start + self.volume_size[2])

        # Create box region
        box_data[:, d_start:d_end, h_start:h_end, w_start:w_end] = 1.0

        return tio.LabelMap(tensor=box_data)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Apply transform to all images in subject."""
        # Get volume size from first image
        images_list = self.get_images(subject)
        if not images_list:
            raise ValueError("Subject has no images to transform.")

        print('images_list', images_list)
        first_image = images_list[0]
        self.image_size = first_image.spatial_shape

        # Randomize parameters
        self.__random_volume_size()

        # Create binary box and rotate it with Affine
        box_image = self.create_binary_box()
        logger.info("%s[RandomRotatedBox]%s Rotation range: %s%s%s", BLUE, RESET, BLUE, self.rotation_range, RESET)
        affine_transform = tio.RandomAffine(degrees=self.rotation_range)
        rotated_box = affine_transform(box_image)
        self.box_mask = rotated_box.data > 0.5

        # Apply mask to all images
        for image in images_list:
            # Extract using rotated box mask
            cropped_tensor = image.data.clone()
            cropped_tensor[~self.box_mask] = 0
            
            image.set_data(cropped_tensor)
        
        return subject
    

# ============================================================================
# SIMPLE TEST
# ============================================================================

def test():
    test_folder = "test_RandomRotatedBox"
    print("=" * 70 + "\n", "TEST: Random Rotated Box\n", "=" * 70 + "\n")
    
    # Create subject with multiple images
    subject = tio.Subject(
        t1=tio.ScalarImage(tensor=torch.randn(1, 128, 128, 128)),
        t2=tio.ScalarImage(tensor=torch.randn(1, 128, 128, 128)),
        label=tio.LabelMap(tensor=torch.randint(0, 3, (1, 128, 128, 128)).float())
    )
    
    print(f"Original subject keys: {list(subject.keys())}\n")
    
    # Create transform
    transform = RandomRotatedBox(
        box_size_reduction_range=(0.1, 0.75),
        rotation_range=(-90, 90)
    )
    
    # Apply transform
    transformed_subject = transform(subject)
    
    # Save original and transformed images
    os.makedirs(test_folder, exist_ok=True)
    for image_name in subject.get_images_dict(intensity_only=False).keys():
        original_image = subject[image_name]
        transformed_image = transformed_subject[image_name]
        
        original_image.save(f"{test_folder}/original_{image_name}.nii.gz")
        transformed_image.save(f"{test_folder}/transformed_{image_name}.nii.gz")
    
    # Save parameters
    with open(f"{test_folder}/transform_parameters.txt", "w") as f:
        f.write(f"Rotation Range: {transform.rotation_range}\n")
        f.write(f"Volume Size: {transform.volume_size}\n")
        f.write(f"Box Position: {transform.box_position}")
    
    print(f"Transformed subject keys: {list(transformed_subject.keys())}")
    print(f"Saved in {test_folder}")
    print("=" * 70)


if __name__ == "__main__":
    test()
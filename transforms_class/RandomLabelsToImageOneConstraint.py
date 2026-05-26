import logging
import math
import random

import torchio as tio
from torchio.transforms.augmentation.intensity.random_labels_to_image import _check_mean_and_std_length
from transforms_class import LabelTransform
from utils.logger import ORANGE, RED, RESET, YELLOW
logger = logging.getLogger(__name__)
"""
RandomLabelsToImageOneConstraint

How it works
------------
Without constraints =>
  contrasted_labels = None → Fast path

With constraints =>
  contrasted_labels = {1: [10], 2: [10], 3: [10]}
  minimal_contrast = 0.1  →  zone_size = 0.2 (constant)
  max_zones = 3
  
    Limit: zones occupy at most 50% of the space [0.0, 1.0]
  
  
    STEP 1: Create source zones (fixed size = 0.2)
  ──────────────────────────────────────────────────
  0.0                                                          1.0
  |───[Zone 0]───────────[Zone 1]───────[Zone 2]──────────────|
  
  
    STEP 2: Randomly assign sources to zones
  ──────────────────────────────────────────────────
  0.0                                                          1.0
  |───[Zone 0]───────────[Zone 1]───────[Zone 2]──────────────|
         S1, S3                          S2
  
  zones_used = {0, 2}
  
  
    STEP 3: Find gaps (target_zones)
  ────────────────────────────────────
  0.0                                                          1.0
  |[──]------[──────────────────────────]------[─────────────]|
   Target           Target zone 1               Target zone 2
   zone 0
        
  
    RESULT: Sources and targets are SEPARATED into distinct zones
"""
class RandomLabelsToImageOneConstraint(tio.RandomLabelsToImage):
    def __init__(
        self,
        minimal_contrast,
        max_zones=None,
        contrasted_labels=None,
        step = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contrasted_labels = contrasted_labels
        self.minimal_contrast = minimal_contrast
        self.max_zones = max_zones
        self.zone_size = 2 * self.minimal_contrast
        self.step = step


        if minimal_contrast < abs(self.default_mean[1] - self.default_mean[0]) // 2:
            raise ValueError(f"Minimal contrast must be inferior to the middle of default_mean but got {minimal_contrast} and {abs(self.default_mean[1] - self.default_mean[0]) // 2}")

        # Useful for the LabelsToImage history
        self.args_names = [
            'label_key',
            'mean',
            'std',
            'image_key',
            'used_labels',
            'ignore_background',
            'discretize',
            'create_mask_index',
            'create_mask_name',
            'random_shape',
            'minimal_contrast',
            'max_zones',
            'contrasted_labels'
        ]

    def compute_label_to_distinguish(self, labels):
        # TODOC
        label_set = set(labels)
        label_to_mean = {}
        label_to_std = {}

        # Case where there is no constraint
        if self.contrasted_labels is None:
            for label in label_set:
                mean, std = self.get_params(label)
                label_to_mean[label] = mean
                label_to_std[label] = std
            return list(label_to_mean.values()), list(label_to_std.values())

        # Case with constraints
        allocation = self.__allocate_space_for_constraints(label_set)

        # Assign values from allocation
        for label in label_set:
            _, std = self.get_params(label)
            label_to_std[label] = std

            if label in allocation:
                label_to_mean[label] = allocation[label]
            else:
                # Label has no constraints: use default mean
                mean, _ = self.get_params(label)
                label_to_mean[label] = mean

        return list(label_to_mean.values()), list(label_to_std.values())

    def __allocate_space_for_constraints(self, label_set):
        """
        Orchestrate the space allocation process.
        Groups labels by constraints and divides space accordingly.
        TODOC
        """
        constraint_groups, unconstrained_labels = self._group_labels_by_constraints(label_set)

        allocation = self._divide_space(constraint_groups)
        return allocation

    def _group_labels_by_constraints(self, label_set):
        """
        Group labels by their constraint targets.
        Returns constraint_groups and unconstrained_labels.

        TODOC
        """
        constraint_groups = {}
        unconstrained_labels = set(label_set)

        for label, constraint_labels in self.contrasted_labels.items():
            label_int = int(label) if isinstance(label, str) else label

            if label_int not in label_set:
                continue

            # Remove from unconstrained set
            unconstrained_labels.discard(label_int)

            # Create immutable key for constraint targets
            constraint_key = frozenset(constraint_labels)

            if constraint_key not in constraint_groups:
                constraint_groups[constraint_key] = []

            constraint_groups[constraint_key].append(label_int)

        return constraint_groups, unconstrained_labels

    def _divide_space(self, constraint_groups):
        #TODOC
        allocation = {}
        for constraint_key, source_labels in constraint_groups.items():
            target_labels = list(constraint_key)
            
            # Create source zones using the max number of possible zones
            space_range = self.default_mean[1] - self.default_mean[0]
            max_zones_possible = (
                self.max_zones 
                if self.max_zones is not None 
                else math.ceil((space_range / self.zone_size) * 3 / 4)
            )
            
            # Number of zones to create for sources
            num_source_zones = min(max_zones_possible, len(source_labels))
            if num_source_zones == 0:
                logger.warning("Not enough space to create zones, using default assignment")
                for label in source_labels:
                    allocation[label] = self.sample_uniform(self.default_mean[0], self.default_mean[1])  # nosec B311
                for label in target_labels:
                    allocation[label] = self.sample_uniform(self.default_mean[0], self.default_mean[1])  # nosec B311
                continue
            
            # Define source zones randomly
            source_zones = []
            for zone_idx in range(num_source_zones):
                center = self.sample_uniform(self.default_mean[0] + self.minimal_contrast, self.default_mean[1] - self.minimal_contrast)  # nosec B311
                source_zones.append([center - self.minimal_contrast, center + self.minimal_contrast])

            
            
            # Find random values for sources in the union of source zones
            zones_used = set()
            for source_label in source_labels:
                value, zone_idx = self._random_from_union_of_ranges(source_zones)
                allocation[source_label] = value
                zones_used.add(zone_idx)
            
            # Determine target zones (gap spaces)
            target_zones = self._find_target_zones(source_zones, zones_used)
            
            
            # Assign targets to target zones
            if target_zones:
                for target_label in target_labels:
                    value, _ = self._random_from_union_of_ranges(target_zones)
                    allocation[target_label] = value
            else:
                logger.warning(f"No target zones available, using default assignment")
                for target_label in target_labels:
                    allocation[target_label] = self.sample_uniform(self.default_mean[0], self.default_mean[1])  # nosec B311
            
        
        return allocation


    def _merge_zones(self, zones):
        """
        Merge zones that overlap or touch.
        
        Example:
        zones = [[0.1, 0.2], [0.15, 0.25], [0.2, 0.35]]
        → merged = [[0.1, 0.35]]  (all merged into one)
        
        Logic:
        - [0.1, 0.2] and [0.15, 0.25] overlap (0.15 <= 0.2) ==> merge
        - [0.1, 0.25] and [0.2, 0.35] touch (0.2 <= 0.25) ==> merge

        TODOC
        """
        if not zones:
            return []
        
        # Sort by zone start
        sorted_zones = sorted(zones, key=lambda z: z[0])
        
        merged = [sorted_zones[0]]
        
        for current_zone in sorted_zones[1:]:
            last_merged = merged[-1]
            
            # Merge if overlap OR touching (equal boundary)
            if current_zone[0] <= last_merged[1]:
                last_merged[1] = max(last_merged[1], current_zone[1])
            else:
                merged.append(current_zone)
        
        return merged

    def _find_target_zones(self, source_zones, zones_used):
        """
        Find target zones = spaces not covered by used source zones
        
        Example:
        source_zones = [[0.05, 0.25], [0.20, 0.40], [0.75, 0.95]]
        zones_used = {0, 1, 2}
        self.default_mean = [0.0, 1.0]
        
        
        Merged zones: [[0.05, 0.40], [0.75, 0.95]]
        ==> target_zones = [[0.0, 0.05], [0.40, 0.75], [0.95, 1.0]]
        TODOC
        """
        if not zones_used:
            # No zone used: all space is available
            return [[self.default_mean[0], self.default_mean[1]]]
        space = self.minimal_contrast / 2
        
        # Collect used zones
        used_zones = [source_zones[i] for i in zones_used]
        
        # Merge zones that overlap or touch
        merged_zones = self._merge_zones(used_zones)
        
        target_zones = []
        
        # Gap before the first merged zone
        if merged_zones[0][0] - space > self.default_mean[0]:
            target_zones.append([self.default_mean[0], merged_zones[0][0] - space])
        
        # Gaps between merged zones
        for i in range(len(merged_zones) - 1):
            gap_start = merged_zones[i][1]
            gap_end = merged_zones[i + 1][0]
            if gap_start + space < gap_end - space:
                target_zones.append([gap_start + space, gap_end - space])
        
        # Gap after the last merged zone
        if merged_zones[-1][1] < self.default_mean[1] - space:
            target_zones.append([merged_zones[-1][1] + space, self.default_mean[1]])
        
        return target_zones

    def _random_from_union_of_ranges(self, zones):
        """
        TODOC
        """
        zones_lengths = []
        for zone in zones:
            zone_length = int(round((zone[1] - zone[0]) / self.step))
            zones_lengths.append(zone_length)
        total_length = sum(zones_lengths)
        rand_idx = random.randint(0, total_length - 1)  # nosec B311

        # find which zone the random index falls into


        cumulative_lengths = [sum(zones_lengths[:i+1]) for i in range(len(zones_lengths))]
        zone_idx = next(i for i, cl in enumerate(cumulative_lengths) if rand_idx < cl)
        offset_idx = rand_idx - (cumulative_lengths[zone_idx - 1] if zone_idx > 0 else 0)
        value = zones[zone_idx][0] + offset_idx * self.step
        
        return value, zone_idx

    @staticmethod
    def add_labels(labels, constraints, dictionary):
        for label in labels:
            dictionary[label] = list(set(dictionary.get(label, [])) | set(constraints))
        return dictionary

    def resolve_anatomy(self, name_map: dict[str, int]) -> None:
        """
        Resolve the anatomy labels to their corresponding integer IDs.

        Parameters
        ----------
        name_map : dict[str, int]
            A mapping from string labels to integer IDs.
        """
        if self.contrasted_labels is not None:
            new_constrasted_labels = {}
            for target_label, constraint_labels in self.contrasted_labels.items():
                new_label = LabelTransform.resolve_value(target_label, name_map)
                new_constraint_labels = LabelTransform.resolve_value(constraint_labels, name_map)
                if isinstance(new_label, list):
                    RandomLabelsToImageOneConstraint.add_labels(new_label, new_constraint_labels, new_constrasted_labels)
                else:
                    new_constrasted_labels[new_label] = new_constraint_labels
            self.contrasted_labels = new_constrasted_labels

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if self.label_key is None:
            iterable = subject.get_images_dict(intensity_only=False).items()
            for name, image in iterable:
                if isinstance(image, tio.LabelMap):
                    self.label_key = name
                    break
            else:
                message = f'No label maps found in subject: {subject}'
                raise RuntimeError(message)

        arguments = {
            'label_key': self.label_key,
            'mean': [],
            'std': [],
            'image_key': self.image_key,
            'used_labels': self.used_labels,
            'discretize': self.discretize,
            'ignore_background': self.ignore_background,
        }

        label_map = subject[self.label_key].data

        # Find out if we face a partial-volume image or a label map.
        # One-hot-encoded label map is considered as a partial-volume image
        all_discrete = label_map.eq(label_map.float().round()).all()
        same_num_dims = label_map.squeeze().dim() < label_map.dim()
        is_discretized = all_discrete and same_num_dims

        if not is_discretized and self.discretize:
            # Take label with highest value in voxel
            max_label, label_map = label_map.max(dim=0, keepdim=True)
            # Remove values where all labels are 0 (i.e. missing labels)
            label_map[max_label == 0] = -1
            is_discretized = True

        if is_discretized:
            labels = label_map.unique().long().tolist()
            if -1 in labels:
                labels.remove(-1)
        else:
            labels = range(label_map.shape[0])

        # Raise error if mean and std are not defined for every label
        _check_mean_and_std_length(labels, self.mean, self.std)  # type: ignore[arg-type]  # noqa: B950

        means = arguments['mean']
        stds = arguments['std']
        assert isinstance(means, list)
        assert isinstance(stds, list)

        arguments['mean'], arguments['std'] = self.compute_label_to_distinguish(labels)

        transform = tio.LabelsToImage(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        assert isinstance(transformed, tio.Subject)
        return transformed
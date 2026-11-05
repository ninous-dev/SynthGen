import logging
import random
from typing import Dict, Set, Tuple, List
from utils.logger import BLUE, GREEN, RED, RESET, YELLOW, configure_logging

import numpy as np

import torchio as tio
from torchio.transforms.augmentation.intensity.random_labels_to_image import _check_mean_and_std_length
from transforms_class import LabelTransform

logger = logging.getLogger(__name__)

"""
RandomLabelsToImageConstraint

Algorithm:
    1. Random allocation for each label
    2. Iteratively adjust to satisfy constraints (max 5 iterations):
       - Adjust TARGET labels respecting minimal_contrast
       - If blocking remains, adjust SOURCE labels respecting minimal_contrast
"""


class RandomLabelsToImageConstraint(tio.RandomLabelsToImage):
    def __init__(
        self,
        minimal_contrast,
        max_zones=None,
        contrasted_labels=None,
        step=0.01,
        highlight_contrasted_labels=False,
        verbose_dev=True,
        max_iterations=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contrasted_labels = contrasted_labels
        self.minimal_contrast = minimal_contrast if isinstance(minimal_contrast, (int, float)) else random.uniform(minimal_contrast[0], minimal_contrast[1])
        self.max_zones = max_zones
        self.zone_size = 2 * self.minimal_contrast
        self.step = step
        self.highlight_contrasted_labels = highlight_contrasted_labels
        self.max_iterations = max_iterations
        
        configure_logging(dev_mode=verbose_dev, module_name=__name__)

        if self.minimal_contrast < abs(self.default_mean[1] - self.default_mean[0]) // 2:
            raise ValueError(
                f"Minimal contrast must be inferior to the middle of default_mean "
                f"but got {self.minimal_contrast} and "
                f"{abs(self.default_mean[1] - self.default_mean[0]) // 2}"
            )

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

    def compute_label_to_distinguish(self, labels: np.array) -> Tuple[List, List]:
        """
        Compute mean and standard deviation parameters for each label.
        
        Retrieves or computes intensity parameters (mean and standard deviation) 
        for each unique label. Handles both unconstrained cases (default parameters) 
        and constrained cases where label means are allocated based on predefined 
        constraints.
        
        Parameters
        ----------
        labels: np.array
            The input labels for which to compute intensity parameters.

        Returns
        -------
        Tuple[List, List]
            A tuple containing two lists: the first list contains the mean values,
            and the second list contains the standard deviation values for each label.
        """
        label_set = set(labels)

        # Always initialize ALL labels with get_params (default random draw)
        means_alloc, stds_alloc = self._initialize_allocation(label_set)

        # If constraints exist, adjust means iteratively
        if self.contrasted_labels is not None:
            constraint_groups = self._group_labels_by_constraints(label_set)
            means_alloc = self._iterative_adjust(means_alloc, constraint_groups, self.max_iterations)
            logger.info(
                "%s[RandomLabelsToImageConstraint]%s Final allocation:\n%s",
                BLUE, RESET, self._format_allocation(means_alloc)
            )

        return list(means_alloc.values()), list(stds_alloc.values())

    def _group_labels_by_constraints(self, label_set: Set[int]) -> Dict:
        """
        Group labels by their constraint targets.
        
        Parameters
        ----------
        label_set: Set[int]
            The set of labels to consider for grouping.
        
        Returns
        -------
        Dict
            A dictionary where keys are frozensets of constraint labels (targets) and
            values are lists of source labels that must be contrasted from those targets.
        """
        constraint_groups = {}

        for label, constraint_labels in self.contrasted_labels.items():
            label_int = int(label) if isinstance(label, str) else label

            if label_int not in label_set:
                continue

            # Use frozenset to ensure that the order of constraint labels does not affect grouping
            constraint_key = frozenset(constraint_labels)

            if constraint_key not in constraint_groups:
                constraint_groups[constraint_key] = []

            constraint_groups[constraint_key].append(label_int)

        return constraint_groups

    def _initialize_allocation(self, label_set: Set[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Initialize intensity values for ALL labels using ``get_params``.

        Parameters
        ----------
        label_set : Set[int]
            Full set of labels present in the current image.

        Returns
        -------
        Tuple[Dict[int, float], Dict[int, float]]
            ``(means, stds)`` — one dict per label, drawn from ``get_params``.
        """
        means, stds = {}, {}
        for label in label_set:
            mean, std = self.get_params(label)
            means[label] = mean
            stds[label]  = std
        return means, stds

    def _iterative_adjust(
        self,
        allocation: Dict[int, float],
        constraint_groups: Dict,
        max_iterations: int
    ) -> Dict[int, float]:
        """
        Iteratively adjust label assignments to satisfy contrast constraints.
        
        Strategy
        --------
            For each iteration (max 5):
                1. Adjust all TARGET labels respecting minimal_contrast
                2. If blocking remains, adjust blocking SOURCE labels respecting minimal_contrast
                3. Continue until constraints satisfied or max iterations

        Parameters
        ----------
        allocation: Dict[int, float]
            Current intensity allocation for each label.
        constraint_groups: Dict
            Grouping of labels by their constraints.
        max_iterations: int
            Maximum iterations (default: 5).
            
        Returns
        -------
        Dict[int, float]
            Final intensity allocation after iterative adjustments.
        """
        for iteration in range(max_iterations):
            # Detect blocking pairs
            blocking_list = self._detect_blocking_labels(allocation, constraint_groups)
            
            # Check if solved
            if self._is_solved(blocking_list):
                return allocation
            
            # Adjust TARGET labels
            blocking_targets, _ = self._get_blocking_labels(blocking_list)
            for target in blocking_targets:
                # Get sources that constrain this target
                constraint_sources = self._get_constraint(target, constraint_groups, role="target")
                new_position = self._find_best_position(target, allocation, constraint_sources)
                allocation[target] = new_position
            
            # Check again after target adjustment
            blocking_list = self._detect_blocking_labels(allocation, constraint_groups)
            if self._is_solved(blocking_list):
                return allocation

            # If still blocking, adjust SOURCE labels
            _, blocking_sources = self._get_blocking_labels(blocking_list)
            if blocking_sources:
                for source in blocking_sources:
                    # Get targets that this source must be contrasted from
                    constraint_targets = self._get_constraint(source, constraint_groups, role="source")
                    new_position = self._find_best_position(source, allocation, constraint_targets)
                    allocation[source] = new_position
                    
        # Final check
        final_test = self._detect_blocking_labels(allocation, constraint_groups)
        if final_test:
            logger.warning(
                f"{YELLOW}[RandomLabelsToImageConstraint]{RESET} "
                f"Could not resolve all constraints: {len(final_test)} violations remain. "
                f"The constraint remaining are {final_test}. "
                f"Consider increasing intensity range or decreasing minimal_contrast."
            )
        else:
            logger.debug(
                f"{GREEN}[RandomLabelsToImageConstraint]{RESET} "
                f"all constraints satisfied"
            )
        
        return allocation

    def _get_constraint(self, label: int, constraint_groups: Dict, role: str) -> Set[int]:
        """
        Get the counterpart labels for a given label based on its role in the constraint groups.

        Parameters
        ----------
        label : int
            The label whose counterparts are looked up.
        constraint_groups : Dict
            Grouping of labels by their constraints (keys are frozensets of targets,
            values are lists of sources).
        role : str
            Either ``"target"`` (return the sources that constrain this target)
            or ``"source"`` (return the targets that this source must be contrasted from).

        Returns
        -------
        Set[int]
            The set of counterpart labels, or an empty set if not found.
        """
        for constraint_key, source_labels in constraint_groups.items():
            if role == "target" and label in constraint_key:
                return set(source_labels)
            if role == "source" and label in source_labels:
                return set(constraint_key)
        return set()

    def _get_blocking_labels(
        self,
        blocking_list: List[Tuple[int, int, float]]
    ) -> Tuple[Set[int], Set[int]]:
        """
        Identify blocking TARGET and SOURCE labels from blocking_list.
        
        Parameters
        ----------
        blocking_list: List[Tuple[int, int, float]]
            A list of tuples (target_label, source_label, contrast) for each pair that violates the minimal contrast constraint.
        
        Returns
        -------
        Tuple[Set[int], Set[int]]
            Two sets: the first contains TARGET labels that violate constraints, 
            and the second contains SOURCE labels that are blocking those targets.
        """
        blocking_targets = set()
        blocking_sources = set()
        
        for target, source, _ in blocking_list:
            blocking_targets.add(target)
            blocking_sources.add(source)
        
        return blocking_targets, blocking_sources

    def _detect_blocking_labels(
        self,
        allocation: Dict[int, float],
        constraint_groups: Dict
    ) -> List[Tuple[int, int, float]]:
        """
        Detect all source-target pairs blocking the minimum contrast constraint
        
        Parameters
        ----------
        allocation: Dict[int, float]
            Current intensity allocation for each label.
        constraint_groups: Dict
            Grouping of labels by their constraints.
        
        Returns
        -------
        List[Tuple[int, int, float]]
            A list of tuples (target_label, source_label, contrast) for each pair that blocks the minimal contrast constraint.
        """
        blocking_list = []
        
        # Iterate through each constraint group
        for constraint_key, source_labels in constraint_groups.items():
            target_labels = list(constraint_key)
            
            # Check each (target, source) pair
            for target in target_labels:
                for source in source_labels:
                    if target not in allocation.keys() or source not in allocation.keys():
                        continue  # Skip if either label is not in the current allocation
                    contrast = abs(allocation[target] - allocation[source])

                    # Record blocking if constraint not satisfied
                    if contrast < self.minimal_contrast:
                        blocking_list.append((target, source, contrast))
        
        return blocking_list

    def _is_solved(
        self,
        blocking_list: List[Tuple[int, int, float]]
    ) -> bool:
        """
        Check if all constraints are currently satisfied.
        
        Parameters
        ----------
        blocking_list : List[Tuple[int, int, float]]
            List of constraint blocking tuples
        
        Returns
        -------
        is_solved : bool
            True if blocking list is empty, False otherwise
        """
        return len(blocking_list) == 0

    def _find_best_position(
        self,
        label: int,
        allocation: Dict[int, float],
        constraint_labels: Set[int] = None,
    ) -> float:
        """
        Find optimal intensity position using max-min distance criterion while respecting minimal_contrast.
        
        Evaluates 100 candidate positions and selects the one that:
        - Maintains minimal_contrast with constraint_labels (required constraint)
        - Maximizes the minimum distance to other labels (optimization goal)
        
        Parameters
        ----------
        label: int
            The label for which to find the best position.
        allocation: Dict[int, float]
            Current intensity allocation for all labels.
        constraint_labels: Set[int]
            Labels that must maintain minimal_contrast with this label.
            If None, no mandatory constraints are applied.
            
        Returns
        -------
        float
            The optimal intensity value for the given label that respects minimal_contrast constraints
            and maximizes the minimum distance to other labels.
        """
        candidates = np.linspace(
            self.default_mean[0],
            self.default_mean[1],
            100
        )
        
        best_score = -float('inf')
        best_value = allocation[label]
        
        constraint_labels = constraint_labels or set()

        # shuffle candidates
        np.random.shuffle(candidates)
        
        # Evaluate each candidate position
        for candidate in candidates:
            # Check if candidate respects minimal_contrast with constraint_labels
            respects_constraint = True
            for constrain_label in constraint_labels:
                if constrain_label not in allocation:
                    continue  # Skip if the constraint label is not in the current allocation
                contrast = abs(candidate - allocation[constrain_label])
                if contrast < self.minimal_contrast:
                    respects_constraint = False
                    break
            
            # Skip candidates that violate constraints
            if not respects_constraint:
                continue
            
            # Compute distances to all other labels
            distances = [
                abs(candidate - allocation[other])
                for other in allocation
                if other != label
            ]
            
            # Max-min criterion: maximize the minimum distance
            min_distance = min(distances)
            
            if min_distance > best_score:
                best_score = min_distance
                best_value = candidate
        
        return best_value

    def _format_allocation(self, allocation: Dict[int, float]) -> str:
        """
        Format allocation for readable logging output.
        
        Parameters
        ----------
        allocation: Dict[int, float]
            Current intensity allocation for each label.
        
        Returns
        -------
        str
            A formatted string representation of the allocation, showing each label, its assigned intensity, and a visual bar representing the intensity value.
        """
        # Show only constrained labels (keys + their constraint targets)
        if self.contrasted_labels is not None:
            constrained = set(self.contrasted_labels.keys())
            for v in self.contrasted_labels.values():
                constrained.update(v)
            allocation = {k: v for k, v in allocation.items() if k in constrained}

        lines = []
        for label, value in sorted(allocation.items()):
            bar_length = int(value * 40)
            bar = '█' * bar_length
            rest = '▒' * (40 - bar_length)
            lines.append(f"{' ' * 45}Label {label:3d}: {value:.2f} {bar}{rest}")
        
        return "\n".join(lines)

    @staticmethod
    def add_labels(
        labels: Set[int], 
        constraints: Set[int], 
        dictionary: Dict[int, List[int]]
    ) -> Dict[int, List[int]]:
        """
        Add constraint associations to labels in a dictionary.
        
        Parameters
        ----------
        labels: Set[int]
            The set of labels to which the constraints should be added.
        constraints: Set[int]
            The set of constraint labels to associate with the given labels.
        dictionary: Dict[int, List[int]]
            The existing dictionary mapping labels to their associated constraint labels.
            
        Returns
        -------
        Dict[int, List[int]]
            An updated dictionary where each label in `labels` has the `constraints` added to its associated constraint labels, ensuring no duplicates.
        """
        for label in labels:
            dictionary[label] = list(set(dictionary.get(label, [])) | set(constraints))
        return dictionary

    def resolve_anatomy(self, name_map: dict[str, int]) -> None:
        """
        Resolve the anatomy labels to their corresponding integer IDs.
        
        Parameters
        ----------
        name_map: dict[str, int]
            A mapping from label names (strings) to their corresponding integer IDs.
        """
        if self.contrasted_labels is not None:
            new_constrasted_labels = {}
            for target_label, constraint_labels in self.contrasted_labels.items():
                new_label = LabelTransform.resolve_value(target_label, name_map)
                new_constraint_labels = LabelTransform.resolve_value(constraint_labels, name_map)

                if isinstance(new_label, list):
                    RandomLabelsToImageConstraint.add_labels(
                        new_label, 
                        new_constraint_labels, 
                        new_constrasted_labels
                    )
                else:
                    new_constrasted_labels[new_label] = new_constraint_labels
            self.contrasted_labels = new_constrasted_labels

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        logger.info("%s[RandomLabelsToImageConstraint]%s Started", BLUE, RESET)
        if self.label_key is None:
            iterable = subject.get_images_dict(intensity_only=False).items()
            for name, image in iterable:
                if isinstance(image, tio.LabelMap):
                    self.label_key = name
                    break
            else:
                message = f'No label maps found in subject: {subject}'
                raise RuntimeError(message)

        # Resolve contrasted_labels
        if "contrasted_lesion_labels_to_remove" in subject:
            for label_to_remove in subject["contrasted_lesion_labels_to_remove"]:
                if self.contrasted_labels is not None:
                    # Remove from keys
                    if label_to_remove in self.contrasted_labels:
                        del self.contrasted_labels[label_to_remove]

                    # Remove from values
                    for _, constraint_labels in self.contrasted_labels.items():
                        if label_to_remove in constraint_labels:
                            constraint_labels.remove(label_to_remove)

                    # Remove entries with empty constraint lists
                    self.contrasted_labels = {
                        key: values
                        for key, values in self.contrasted_labels.items()
                        if len(values) > 0
                    }
            logger.info("%s[RandomLabelsToImageConstraint]%s Resolved contrasted_labels after removing empty lesion labels: %s", 
                        BLUE, RESET, self.contrasted_labels)
                        
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

        # Check if label map is already discretized
        all_discrete = label_map.eq(label_map.float().round()).all()
        same_num_dims = label_map.squeeze().dim() < label_map.dim()
        is_discretized = all_discrete and same_num_dims

        if is_discretized:
            labels = label_map.unique().long().tolist()
            if -1 in labels:
                labels.remove(-1)
        else:
            labels = range(label_map.shape[0])

        # Check that mean and std lists have the same length as the number of labels
        _check_mean_and_std_length(labels, self.mean, self.std)

        means = arguments['mean']
        stds = arguments['std']
        assert isinstance(means, list)
        assert isinstance(stds, list)

        arguments['mean'], arguments['std'] = self.compute_label_to_distinguish(labels)

        transform = tio.LabelsToImage(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        assert isinstance(transformed, tio.Subject)
        return transformed
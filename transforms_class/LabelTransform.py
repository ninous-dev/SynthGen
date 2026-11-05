import logging
import torchio as tio
from typing import List, Union
import math

from utils.logger import configure_logging
logger = logging.getLogger(__name__)

class LabelTransform(tio.LabelTransform):

    def __init__(self, verbose_dev: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set logging level to INFO for non-dev mode
        configure_logging(dev_mode=verbose_dev, module_name=__name__)
        
        self.anatomy_args = [
            "labels_to_target",
            "labels_to_keep",
            "center_labels",
            "background_value",
            "lesion_label",
            "lesion_labels",
            "labels_for_edema",
            "label_within",
            "label_to_dilate",
            "label_within_delete",
            "ventricular_label"
        ]
        
    def check_anatomy_resolved(self):
        """
        Check if the anatomy labels have been resolved.

        Raises
        ------
        ValueError
            If the anatomy labels are not resolved.
        """
        def validate_labels(labels, label_name):
            if isinstance(labels, float):
                return int(labels)  # Convert single float to int if it's a whole number
            elif isinstance(labels, int):
                return labels  # Already an integer, no conversion needed
            elif isinstance(labels, list) and all(isinstance(v, (int, float)) for v in labels):
                return [int(v) for v in labels]  # Convert list of floats to ints if they are whole numbers
            elif not isinstance(labels, list) or not all(isinstance(v, int) for v in labels):
                raise ValueError(f"{label_name} must be resolved to integers, got {labels}.")

        for anatomy_arg in self.anatomy_args:
            if hasattr(self, anatomy_arg) and getattr(self, anatomy_arg) is not None:
                validate_labels(getattr(self, anatomy_arg), anatomy_arg)
                
    def _is_int_value(self, value) -> bool:
        """Check if value is int or list of ints (not strings)."""
        if isinstance(value, int):
            return True
        elif isinstance(value, list):
            return all(isinstance(v, int) for v in value)
        return False
    
    def _flatten_labels(self, labels) -> list:
        """Helper to flatten label values to a list of numbers."""
        if isinstance(labels, (int, float)):
            return [labels]
        elif isinstance(labels, list):
            return labels
        return []

    def resolve_anatomy(self, name_map: dict[str, int]) -> None:
        """
        Resolve the anatomy labels to their corresponding integer IDs.

        Parameters
        ----------
        name_map : dict[str, int]
            A mapping from string labels to integer IDs.
        """
        for anatomy_arg in self.anatomy_args:
            if hasattr(self, anatomy_arg) and getattr(self, anatomy_arg) is not None:
                current_value = getattr(self, anatomy_arg)

                # Keep track of labels_for_edema mapping ONLY if instance is TumorTransform
                if anatomy_arg == "labels_for_edema" and type(self).__name__ == "TumorTransform":
                    if not hasattr(self, "original_edema_labels"):
                        setattr(self, "original_edema_labels", {})

                    original_dict = getattr(self, "original_edema_labels")
                    
                    # First time only: create keys with original items and resolve them
                    if not original_dict and isinstance(current_value, list):
                        for item in current_value:
                            resolved = LabelTransform.resolve_value(item, name_map)
                            original_dict[str(item)] = resolved
                    
                    # Not first time: resolve the existing values again
                    elif original_dict:
                        for key in original_dict.keys():
                            current_resolved = original_dict[key]
                            new_resolved = LabelTransform.resolve_value(current_resolved, name_map)
                            original_dict[key] = new_resolved
                
                # Resolve the value
                setattr(self, anatomy_arg, LabelTransform.resolve_value(
                    getattr(self, anatomy_arg), name_map
                ))

    @staticmethod
    def resolve_value(
        value: Union[int, str, List[Union[int, str]]],
        name_map: dict[str, int],
    ) -> Union[int, List[int]]:
        if isinstance(value, float):
            return value

        elif isinstance(value, int):
            if value in name_map.keys():
                return name_map[value]
            return value

        elif isinstance(value, str):
            r = name_map[value]

            # Check for NaN values in the resolved result
            if isinstance(r, float):
                r = [r]  # Convert to list for uniform processing
            if isinstance(r, (list, tuple, set)):
                r = [item for item in r if not (isinstance(item, float) and math.isnan(item))]
                # convert to set to remove duplicates, then back to list
                r = list(set(r))
            return r

        elif isinstance(value, list):
            resolved = []
            for v in value:
                r = LabelTransform.resolve_value(v, name_map)
                if r is not None:
                    if isinstance(r, (list, set, tuple)):
                        resolved.extend(list(r))
                    else:
                        resolved.append(r)
            resolved = list(set(resolved))  # Remove duplicates
            return resolved

        raise TypeError(f"Unsupported value type {type(value)} for value={value} with name_map={list(name_map.keys())}")

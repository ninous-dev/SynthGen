import logging
import torchio as tio
from typing import List, Union
import math

from utils.logger import configure_logging
logger = logging.getLogger(__name__)

class LabelTransform(tio.LabelTransform):

    def __init__(self, dev_mode: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not dev_mode:
            # Set logging level to INFO for non-dev mode
            configure_logging(dev_mode=dev_mode, module_name=__name__)

        self.anatomy_args = [
            "labels_to_target",
            "labels_to_keep",
            "center_labels",
            "background_value",
            "lesion_label",
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
                setattr(self, anatomy_arg, LabelTransform.resolve_value(getattr(self, anatomy_arg), name_map))
                    
                    
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
            return resolved

        raise TypeError(f"Unsupported value type {type(value)} for value={value} with name_map={list(name_map.keys())}")

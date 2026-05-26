# -----------------------------------------------------------------------------
# JSON Encoding
# -----------------------------------------------------------------------------

import json
import numpy as np
import torch

class ArrayTensorJSONEncoder(json.JSONEncoder):
    """
    Encode NumPy arrays and Torch tensors as JSON lists, floats as strings.
    """
    def default(self, obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        if isinstance(obj, (float, np.floating)):
            return str(obj)
        return super().default(obj)

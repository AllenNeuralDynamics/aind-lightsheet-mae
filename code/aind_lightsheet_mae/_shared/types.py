"""
Defines the data types used in
the repository
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch

# IO types
PathLike = Union[str, Path]
ArrayLike = Union[torch.Tensor, np.ndarray]
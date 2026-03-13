"""Shared interface for MRI image and k-space distortions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDistortion(ABC):
    """Compact base interface for distortion operators."""

    def __call__(self, data: Any, **kwargs: Any) -> np.ndarray:
        return self.apply(data, **kwargs)

    @abstractmethod
    def apply(self, data: Any, **kwargs: Any) -> np.ndarray:
        """Apply the distortion and return a NumPy array."""

    def to_numpy(self, data: Any) -> np.ndarray:
        if (
            hasattr(data, "detach")
            and hasattr(data, "cpu")
            and hasattr(data, "numpy")
        ):
            data = data.detach().cpu().numpy()
        array = np.asarray(data)
        if array.ndim < 2:
            raise ValueError(
                "Distortions require arrays with at least two dimensions"
            )
        return array

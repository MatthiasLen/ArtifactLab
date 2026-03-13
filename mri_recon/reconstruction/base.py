"""Abstract interfaces shared by MRI reconstruction algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseReconstructor(ABC):
    """Compact base interface for MRI reconstruction algorithms.

    Implementations operate on the standard sample dictionaries returned by the
    package datasets, most importantly :class:`mri_recon.datasets.FastMRIDataset`.
    """

    def __init__(
        self,
        kspace_field: str = "kspace",
        mask_field: str = "mask",
        target_field: str = "target",
    ) -> None:
        self.kspace_field = kspace_field
        self.mask_field = mask_field
        self.target_field = target_field

    def __call__(self, sample: dict[str, Any], **kwargs: Any) -> Any:
        """Alias :meth:`apply_reconstruction` for a compact call-site syntax."""

        return self.apply_reconstruction(sample, **kwargs)

    @abstractmethod
    def apply_reconstruction(self, sample: dict[str, Any], **kwargs: Any) -> Any:
        """Reconstruct an image from a standard sample dictionary."""

    def get_kspace(self, sample: dict[str, Any]) -> Any:
        """Return the k-space measurements stored in *sample*."""

        return self._require_field(sample, self.kspace_field)

    def get_mask(self, sample: dict[str, Any]) -> Any:
        """Return the optional sampling mask stored in *sample*."""

        return sample.get(self.mask_field)

    def get_target(self, sample: dict[str, Any]) -> Any:
        """Return the optional reference image stored in *sample*."""

        return sample.get(self.target_field)

    def to_numpy(self, data: Any) -> Any:
        """Convert tensors or array-like values into NumPy arrays when possible."""

        if data is None:
            return data
        if hasattr(data, "detach") and hasattr(data, "cpu") and hasattr(data, "numpy"):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    def _require_field(self, sample: dict[str, Any], field: str) -> Any:
        if field not in sample:
            raise KeyError(f"Sample does not contain field {field!r}")
        return sample[field]

"""Classic MRI reconstruction algorithms."""

from __future__ import annotations

from typing import Any

from .base import BaseReconstructor

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised via runtime guard.
    np = None


def _require_numpy() -> None:
    if np is None:
        raise ImportError(
            "Classic MRI reconstruction requires numpy. Install dependencies from "
            "requirements.txt before using these reconstructors."
        )


def _fft2c(image: np.ndarray) -> np.ndarray:
    """Return a centered orthonormal 2D FFT over the last two axes."""

    _require_numpy()
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def _ifft2c(kspace: np.ndarray) -> np.ndarray:
    """Return a centered orthonormal 2D inverse FFT over the last two axes."""

    _require_numpy()
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


class ZeroFilledReconstructor(BaseReconstructor):
    """Classical zero-filled inverse FFT MRI reconstruction."""

    def __init__(self, coil_axis: int = 0, magnitude: bool = True) -> None:
        super().__init__()
        self.coil_axis = coil_axis
        self.magnitude = magnitude

    def apply_reconstruction(self, sample: dict[str, Any], **kwargs: Any) -> np.ndarray:
        """Apply a centered inverse FFT and optionally return a magnitude image."""

        del kwargs
        _require_numpy()
        reconstructed = _ifft2c(np.asarray(self.get_kspace(sample)))
        if reconstructed.ndim > 2:
            return np.sqrt(np.sum(np.abs(reconstructed) ** 2, axis=self.coil_axis))
        if self.magnitude:
            return np.abs(reconstructed)
        return reconstructed


class LandweberReconstructor(BaseReconstructor):
    """Simple iterative Landweber reconstruction for Cartesian single-coil MRI."""

    def __init__(
        self,
        num_iterations: int = 10,
        step_size: float = 1.0,
        l2_weight: float = 0.0,
        magnitude: bool = True,
    ) -> None:
        super().__init__()
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.l2_weight = l2_weight
        self.magnitude = magnitude

    def apply_reconstruction(self, sample: dict[str, Any], **kwargs: Any) -> np.ndarray:
        """Iteratively solve a masked least-squares MRI reconstruction problem."""

        del kwargs
        _require_numpy()
        measured_kspace = np.asarray(self.get_kspace(sample))
        mask = self._prepare_mask(sample, measured_kspace.shape)
        reconstruction = _ifft2c(mask * measured_kspace)
        for _ in range(self.num_iterations):
            residual = mask * (_fft2c(reconstruction) - measured_kspace)
            gradient = _ifft2c(residual) + self.l2_weight * reconstruction
            reconstruction = reconstruction - self.step_size * gradient
        if self.magnitude:
            return np.abs(reconstruction)
        return reconstruction

    def _prepare_mask(self, sample: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray:
        mask = self.get_mask(sample)
        if mask is None:
            return np.abs(np.asarray(self.get_kspace(sample))) > 0
        mask_array = np.asarray(mask).astype(bool)
        while mask_array.ndim < len(shape):
            mask_array = np.expand_dims(mask_array, axis=0)
        return np.broadcast_to(mask_array, shape)

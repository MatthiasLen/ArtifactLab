"""Reconstruction methods specialized for undersampled k-space."""

from __future__ import annotations

from typing import Any

from .base import BaseReconstructor
from .classic import _fft2c, _ifft2c

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised via runtime guard.
    np = None


def _require_numpy() -> None:
    """Ensure NumPy is available before running undersampled methods."""

    if np is None:
        raise ImportError(
            "Undersampled MRI reconstruction requires numpy. Install "
            "dependencies from requirements.txt before using these methods."
        )


class POCSReconstructor(BaseReconstructor):
    """POCS reconstruction for undersampled Cartesian k-space.

    This method alternates between:
    1) sparsity promotion via complex soft-thresholding in image domain,
    2) exact data consistency at acquired k-space locations.
    """

    def __init__(
        self,
        num_iterations: int = 25,
        l1_weight: float = 1e-3,
        magnitude: bool = True,
    ) -> None:
        super().__init__()
        self.num_iterations = num_iterations
        self.l1_weight = l1_weight
        self.magnitude = magnitude

    def apply_reconstruction(self, sample: dict[str, Any], **kwargs: Any) -> Any:
        """Run POCS iterations with thresholding and k-space projection.

        Args:
            sample: Standard sample dictionary containing undersampled k-space
                and optional sampling mask.
            **kwargs: Unused optional arguments for interface compatibility.

        Returns:
            Reconstructed image using POCS updates, optionally converted to
            magnitude.
        """

        del kwargs
        _require_numpy()
        measured_kspace = np.asarray(self.get_kspace(sample))
        mask = self._prepare_mask(sample, measured_kspace.shape)

        # Normalize k-space so one l1_weight value works across scans/scales.
        scale = float(np.max(np.abs(measured_kspace)))
        if scale <= 0.0:
            reconstruction = np.zeros_like(_ifft2c(mask * measured_kspace))
            return reconstruction if not self.magnitude else np.abs(reconstruction)
        normalized_kspace = measured_kspace / scale

        # Initialization: inverse FFT of masked measurements (zero-filled baseline).
        reconstruction = _ifft2c(mask * normalized_kspace)
        for _ in range(self.num_iterations):
            # Sparsity prior step: shrink coefficients toward zero.
            reconstruction = self._soft_threshold_complex(
                reconstruction,
                self.l1_weight,
            )
            kspace_estimate = _fft2c(reconstruction)
            # Data-consistency step: keep measured samples, update only missing ones.
            kspace_estimate = (
                mask * normalized_kspace
                + (1.0 - mask) * kspace_estimate
            )
            reconstruction = _ifft2c(kspace_estimate)

        # Restore original physical scale after normalized-domain optimization.
        reconstruction = reconstruction * scale

        if self.magnitude:
            return np.abs(reconstruction)
        return reconstruction

    def _prepare_mask(self, sample: dict[str, Any], shape: tuple[int, ...]) -> Any:
        """Build a float mask broadcastable to k-space shape.

        Args:
            sample: Input sample dictionary.
            shape: Desired k-space shape.

        Returns:
            Float mask with entries in {0, 1} and shape matching k-space.
        """

        mask = self.get_mask(sample)
        if mask is None:
            # Fallback for datasets without explicit masks.
            return (np.abs(np.asarray(self.get_kspace(sample))) > 0).astype(np.float32)
        mask_array = np.asarray(mask).astype(np.float32)
        while mask_array.ndim < len(shape):
            mask_array = np.expand_dims(mask_array, axis=0)
        return np.broadcast_to(mask_array, shape)

    def _soft_threshold_complex(self, image: Any, threshold: float) -> Any:
        """Apply complex soft-thresholding to an image.

        Args:
            image: Complex-valued image array.
            threshold: Non-negative shrinkage parameter.

        Returns:
            Thresholded complex image where magnitudes are shrunk while phases
            are preserved.
        """

        magnitude = np.abs(image)
        shrunk = np.maximum(magnitude - threshold, 0.0)
        return np.where(magnitude > 0, shrunk * image / magnitude, 0.0)


class FISTAL1Reconstructor(BaseReconstructor):
    """FISTA-L1 reconstruction for undersampled Cartesian k-space.

    It minimizes:
    0.5 * || M F x - y ||_2^2 + lambda * ||x||_1
    with accelerated proximal-gradient updates and a data-consistency projection.
    """

    def __init__(
        self,
        num_iterations: int = 25,
        l1_weight: float = 1e-3,
        step_size: float = 1.0,
        magnitude: bool = True,
    ) -> None:
        super().__init__()
        self.num_iterations = num_iterations
        self.l1_weight = l1_weight
        self.step_size = step_size
        self.magnitude = magnitude

    def apply_reconstruction(self, sample: dict[str, Any], **kwargs: Any) -> Any:
        """Run normalized FISTA-L1 updates with explicit data consistency.

        Args:
            sample: Standard sample dictionary with undersampled k-space and
                optional sampling mask.
            **kwargs: Unused optional arguments for interface compatibility.

        Returns:
            Reconstructed image after accelerated proximal-gradient iterations,
            optionally converted to magnitude.
        """

        del kwargs
        _require_numpy()
        measured_kspace = np.asarray(self.get_kspace(sample))
        mask = self._prepare_mask(sample, measured_kspace.shape)

        # Normalize k-space so regularization strength is scan-scale invariant.
        scale = float(np.max(np.abs(measured_kspace)))
        if scale <= 0.0:
            reconstruction = np.zeros_like(_ifft2c(mask * measured_kspace))
            return reconstruction if not self.magnitude else np.abs(reconstruction)
        normalized_kspace = measured_kspace / scale

        # FISTA initialization from the zero-filled reconstruction.
        x_prev = _ifft2c(mask * normalized_kspace)
        y_curr = x_prev.copy()
        t_prev = 1.0
        for _ in range(self.num_iterations):
            # Gradient of data-fidelity term under unitary FFT conventions.
            gradient = _ifft2c(mask * (_fft2c(y_curr) - normalized_kspace))
            # Proximal L1 step (soft-threshold) on the gradient update.
            x_next = self._soft_threshold_complex(
                y_curr - self.step_size * gradient,
                self.step_size * self.l1_weight,
            )

            # Projection onto measurement-consistent set in k-space.
            x_next_kspace = _fft2c(x_next)
            x_next_kspace = (
                mask * normalized_kspace
                + (1.0 - mask) * x_next_kspace
            )
            x_next = _ifft2c(x_next_kspace)

            # Nesterov acceleration (FISTA momentum recursion).
            t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_prev * t_prev) ** 0.5)
            y_curr = x_next + ((t_prev - 1.0) / t_next) * (x_next - x_prev)
            x_prev = x_next
            t_prev = t_next

        # Restore original scale for downstream comparison and plotting.
        reconstruction = x_prev * scale

        if self.magnitude:
            return np.abs(reconstruction)
        return reconstruction

    def _prepare_mask(self, sample: dict[str, Any], shape: tuple[int, ...]) -> Any:
        """Build a float mask broadcastable to k-space shape.

        Args:
            sample: Input sample dictionary.
            shape: Desired k-space shape.

        Returns:
            Float mask with entries in {0, 1} and shape matching k-space.
        """

        mask = self.get_mask(sample)
        if mask is None:
            return (np.abs(np.asarray(self.get_kspace(sample))) > 0).astype(np.float32)
        mask_array = np.asarray(mask).astype(np.float32)
        while mask_array.ndim < len(shape):
            mask_array = np.expand_dims(mask_array, axis=0)
        return np.broadcast_to(mask_array, shape)

    def _soft_threshold_complex(self, image: Any, threshold: float) -> Any:
        """Apply complex soft-thresholding to an image.

        Args:
            image: Complex-valued image array.
            threshold: Non-negative shrinkage parameter.

        Returns:
            Thresholded complex image where magnitudes are shrunk while phases
            are preserved.
        """

        magnitude = np.abs(image)
        shrunk = np.maximum(magnitude - threshold, 0.0)
        return np.where(magnitude > 0, shrunk * image / magnitude, 0.0)

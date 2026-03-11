"""Classic MRI reconstruction algorithms."""

from __future__ import annotations

from typing import Any

from .base import BaseReconstructor

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised via runtime guard.
    np = None


def _require_numpy() -> None:
    """Ensure NumPy is available before running numerical reconstruction code."""

    if np is None:
        raise ImportError(
            "Classic MRI reconstruction requires numpy. Install dependencies from "
            "requirements.txt before using these reconstructors."
        )


def _fft2c(image: np.ndarray) -> np.ndarray:
    """Return centered orthonormal 2D FFT over spatial axes.

    Args:
        image: Complex or real-valued image array. The FFT is applied over the
            last two axes.

    Returns:
        A centered k-space array with orthonormal FFT scaling.
    """

    _require_numpy()
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def _ifft2c(kspace: np.ndarray) -> np.ndarray:
    """Return centered orthonormal 2D inverse FFT over spatial axes.

    Args:
        kspace: Complex-valued k-space array. The inverse FFT is applied over
            the last two axes.

    Returns:
        A centered image-domain array with orthonormal FFT scaling.
    """

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
        """Apply a zero-filled inverse FFT baseline.

        Args:
            sample: Standard sample dictionary containing at least a ``kspace``
                entry and optionally a ``mask``.
            **kwargs: Unused optional arguments for interface compatibility.

        Returns:
            Reconstructed image. For multicoil input this returns RSS
            magnitude across ``coil_axis``. For single-coil it returns either
            magnitude or complex image based on ``self.magnitude``.
        """

        del kwargs
        _require_numpy()
        # Zero-filling means we directly inverse-transform the measured k-space.
        reconstructed = _ifft2c(np.asarray(self.get_kspace(sample)))
        if reconstructed.ndim > 2:
            # For multicoil data, aggregate coil images with root-sum-of-squares.
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
        """Iteratively solve a masked least-squares MRI problem.

        Args:
            sample: Standard sample dictionary containing k-space and optional
                sampling mask.
            **kwargs: Unused optional arguments for interface compatibility.

        Returns:
            Reconstructed image after ``num_iterations`` gradient steps,
            returned as magnitude if ``self.magnitude`` is enabled.
        """

        del kwargs
        _require_numpy()
        measured_kspace = np.asarray(self.get_kspace(sample))
        mask = self._prepare_mask(sample, measured_kspace.shape)

        # Start from zero-filled inverse as a physically meaningful initial guess.
        reconstruction = _ifft2c(mask * measured_kspace)
        for _ in range(self.num_iterations):
            # Residual in measurement domain: how far current estimate misses data.
            residual = mask * (_fft2c(reconstruction) - measured_kspace)
            # Gradient of data term plus optional L2 regularization.
            gradient = _ifft2c(residual) + self.l2_weight * reconstruction
            # Landweber update (gradient descent step).
            reconstruction = reconstruction - self.step_size * gradient
        if self.magnitude:
            return np.abs(reconstruction)
        return reconstruction

    def _prepare_mask(self, sample: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray:
        """Return a broadcastable boolean sampling mask.

        Args:
            sample: Input sample dictionary.
            shape: Desired k-space shape to broadcast to.

        Returns:
            Boolean mask with the same shape as the k-space array.
        """

        mask = self.get_mask(sample)
        if mask is None:
            # If no explicit mask exists, infer support from nonzero k-space.
            return np.abs(np.asarray(self.get_kspace(sample))) > 0
        mask_array = np.asarray(mask).astype(bool)
        while mask_array.ndim < len(shape):
            mask_array = np.expand_dims(mask_array, axis=0)
        return np.broadcast_to(mask_array, shape)


class ConjugateGradientReconstructor(BaseReconstructor):
    """Conjugate-gradient MRI reconstruction for masked Cartesian sampling."""

    def __init__(
        self,
        num_iterations: int = 20,
        l2_weight: float = 0.0,
        tolerance: float = 1e-8,
        magnitude: bool = True,
    ) -> None:
        super().__init__()
        self.num_iterations = num_iterations
        self.l2_weight = l2_weight
        self.tolerance = tolerance
        self.magnitude = magnitude

    def apply_reconstruction(self, sample: dict[str, Any], **kwargs: Any) -> np.ndarray:
        """Solve normal equations with conjugate gradient in image space.

        Args:
            sample: Standard sample dictionary containing k-space and optional
                sampling mask.
            **kwargs: Unused optional arguments for interface compatibility.

        Returns:
            Reconstructed image from conjugate-gradient iterations, optionally
            converted to magnitude.
        """

        del kwargs
        _require_numpy()
        measured_kspace = np.asarray(self.get_kspace(sample))
        mask = self._prepare_mask(sample, measured_kspace.shape)

        # Solve (F^H M F + lambda I) x = F^H M y with CG in image space.
        right_hand_side = _ifft2c(mask * measured_kspace)
        estimate = np.zeros_like(right_hand_side)
        residual = right_hand_side - self._apply_normal_operator(estimate, mask)
        direction = residual.copy()
        residual_norm = np.vdot(residual, residual).real

        for _ in range(self.num_iterations):
            # Stop when the normal-equation residual is sufficiently small.
            if residual_norm <= self.tolerance:
                break
            normal_direction = self._apply_normal_operator(direction, mask)
            denominator = np.vdot(direction, normal_direction).real
            # Non-positive curvature indicates degeneracy/numerical instability.
            if denominator <= 0:
                break
            step = residual_norm / denominator
            estimate = estimate + step * direction
            residual = residual - step * normal_direction
            next_residual_norm = np.vdot(residual, residual).real
            if next_residual_norm <= self.tolerance:
                break
            # Conjugacy factor controlling new search direction.
            beta = next_residual_norm / residual_norm
            direction = residual + beta * direction
            residual_norm = next_residual_norm

        if self.magnitude:
            return np.abs(estimate)
        return estimate

    def _prepare_mask(self, sample: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray:
        """Return a broadcastable boolean sampling mask.

        Args:
            sample: Input sample dictionary.
            shape: Desired k-space shape.

        Returns:
            Boolean mask with dimensions matching k-space.
        """

        mask = self.get_mask(sample)
        if mask is None:
            return np.abs(np.asarray(self.get_kspace(sample))) > 0
        mask_array = np.asarray(mask).astype(bool)
        while mask_array.ndim < len(shape):
            mask_array = np.expand_dims(mask_array, axis=0)
        return np.broadcast_to(mask_array, shape)

    def _apply_normal_operator(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply normal operator ``F^H M F + lambda I`` to an image.

        Args:
            image: Current image iterate.
            mask: Sampling mask in k-space.

        Returns:
            Output of the normal operator for CG updates.
        """

        projected = _ifft2c(mask * _fft2c(image))
        if self.l2_weight == 0:
            return projected
        return projected + self.l2_weight * image


class TikhonovReconstructor(BaseReconstructor):
    """Closed-form Tikhonov reconstruction in Fourier domain."""

    def __init__(self, l2_weight: float = 1e-3, magnitude: bool = True) -> None:
        super().__init__()
        self.l2_weight = l2_weight
        self.magnitude = magnitude

    def apply_reconstruction(self, sample: dict[str, Any], **kwargs: Any) -> np.ndarray:
        """Solve a ridge-regularized least-squares problem in k-space.

        Args:
            sample: Standard sample dictionary containing k-space and optional
                mask.
            **kwargs: Unused optional arguments for interface compatibility.

        Returns:
            Reconstructed image obtained by closed-form filtering in k-space,
            optionally converted to magnitude.
        """

        del kwargs
        _require_numpy()
        measured_kspace = np.asarray(self.get_kspace(sample))
        mask = self._prepare_mask(sample, measured_kspace.shape)

        # Closed-form in Fourier domain: x = F^-1((M y) / (M + lambda)).
        denominator = mask.astype(measured_kspace.dtype) + self.l2_weight
        stabilized_kspace = (mask * measured_kspace) / denominator
        reconstructed = _ifft2c(stabilized_kspace)
        if self.magnitude:
            return np.abs(reconstructed)
        return reconstructed

    def _prepare_mask(self, sample: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray:
        """Return a broadcastable boolean sampling mask.

        Args:
            sample: Input sample dictionary.
            shape: Desired k-space shape.

        Returns:
            Boolean mask with dimensions matching k-space.
        """

        mask = self.get_mask(sample)
        if mask is None:
            return np.abs(np.asarray(self.get_kspace(sample))) > 0
        mask_array = np.asarray(mask).astype(bool)
        while mask_array.ndim < len(shape):
            mask_array = np.expand_dims(mask_array, axis=0)
        return np.broadcast_to(mask_array, shape)

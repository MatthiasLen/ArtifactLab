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


class TVPDHGReconstructor(BaseReconstructor):
    """Total-variation reconstruction via Primal-Dual Hybrid Gradient (PDHG).

    The solver minimizes
        0.5 * || M F x - y ||_2^2 + lambda * TV(x)
    for Cartesian single-coil acquisitions, where ``M`` is the sampling mask.
    """

    def __init__(
        self,
        num_iterations: int = 120,
        tv_weight: float = 1e-3,
        tau: float = 0.2,
        sigma: float = 0.2,
        theta: float = 1.0,
        magnitude: bool = True,
    ) -> None:
        super().__init__()
        self.num_iterations = num_iterations
        self.tv_weight = tv_weight
        self.tau = tau
        self.sigma = sigma
        self.theta = theta
        self.magnitude = magnitude

    def apply_reconstruction(self, sample: dict[str, Any], **kwargs: Any) -> Any:
        """Run PDHG updates for TV-regularized reconstruction.

        Args:
            sample: Standard sample dictionary with k-space and optional mask.
            **kwargs: Unused optional arguments for interface compatibility.

        Returns:
            Reconstructed image, optionally converted to magnitude.
        """

        del kwargs
        _require_numpy()
        measured_kspace = np.asarray(self.get_kspace(sample))
        mask = self._prepare_mask(sample, measured_kspace.shape)

        scale = float(np.max(np.abs(measured_kspace)))
        if scale <= 0.0:
            reconstruction = np.zeros_like(_ifft2c(mask * measured_kspace))
            return reconstruction if not self.magnitude else np.abs(reconstruction)

        # Solve in a normalized domain so one tv_weight works across scans.
        normalized_kspace = measured_kspace / scale

        # Initialize primal and dual variables from the zero-filled baseline.
        x = _ifft2c(mask * normalized_kspace)
        x_bar = x.copy()
        dual_data = np.zeros_like(measured_kspace)
        dual_grad_x = np.zeros_like(x)
        dual_grad_y = np.zeros_like(x)

        for _ in range(self.num_iterations):
            # Dual ascent: data-fidelity and TV channels.
            forward_data = mask * _fft2c(x_bar)
            grad_x, grad_y = self._gradient(x_bar)

            dual_data = self._prox_l2_data_conjugate(
                dual_data + self.sigma * forward_data,
                normalized_kspace,
                self.sigma,
            )

            dual_grad_x, dual_grad_y = self._project_onto_tv_dual_ball(
                dual_grad_x + self.sigma * grad_x,
                dual_grad_y + self.sigma * grad_y,
                self.tv_weight,
            )

            # Primal descent with K^T p = A^H p_data + grad^T p_tv.
            x_prev = x
            x = x - self.tau * (
                _ifft2c(mask * dual_data)
                + self._gradient_adjoint(dual_grad_x, dual_grad_y)
            )
            x_bar = x + self.theta * (x - x_prev)

        reconstruction = x * scale
        if self.magnitude:
            return np.abs(reconstruction)
        return reconstruction

    def _prepare_mask(self, sample: dict[str, Any], shape: tuple[int, ...]) -> Any:
        """Build a float mask broadcastable to k-space shape.

        If no explicit mask is present, default to fully sampled support. This
        is the correct fallback for fully sampled fastMRI volumes where missing
        mask metadata does not imply missing k-space lines.
        """

        mask = self.get_mask(sample)
        if mask is None:
            return np.ones(shape, dtype=np.float32)
        mask_array = np.asarray(mask).astype(np.float32)
        while mask_array.ndim < len(shape):
            mask_array = np.expand_dims(mask_array, axis=0)
        return np.broadcast_to(mask_array, shape)

    def _prox_l2_data_conjugate(
        self,
        dual_var: Any,
        measured: Any,
        sigma: float,
    ) -> Any:
        """Apply prox_{sigma f*} for f(u)=0.5||u-y||_2^2."""

        return (dual_var - sigma * measured) / (1.0 + sigma)

    def _project_onto_tv_dual_ball(self, qx: Any, qy: Any, radius: float) -> tuple[Any, Any]:
        """Project vector field onto the isotropic TV dual ball."""

        norm = np.sqrt(np.abs(qx) ** 2 + np.abs(qy) ** 2)
        scale = np.maximum(1.0, norm / max(radius, 1e-12))
        return qx / scale, qy / scale

    def _gradient(self, image: Any) -> tuple[Any, Any]:
        """Forward differences with Neumann boundary handling."""

        grad_x = np.zeros_like(image)
        grad_y = np.zeros_like(image)
        grad_x[..., :-1, :] = image[..., 1:, :] - image[..., :-1, :]
        grad_y[..., :, :-1] = image[..., :, 1:] - image[..., :, :-1]
        return grad_x, grad_y

    def _gradient_adjoint(self, qx: Any, qy: Any) -> Any:
        """Return the exact adjoint of ``_gradient`` (``grad^T``).

        For forward differences ``Dx[i] = x[i+1] - x[i]``, the adjoint is
        ``D^T p = [-p0, p0-p1, ..., p_{n-2}]``. Using this exact sign pattern
        is critical for PDHG convergence; the opposite sign turns the primal
        update into ascent and produces noise-like outputs.
        """

        adj = np.zeros_like(qx)

        # Adjoint along x/spatial-vertical axis.
        adj[..., 0, :] -= qx[..., 0, :]
        adj[..., 1:-1, :] += qx[..., :-2, :] - qx[..., 1:-1, :]
        adj[..., -1, :] += qx[..., -2, :]

        # Adjoint along y/spatial-horizontal axis.
        adj[..., :, 0] -= qy[..., :, 0]
        adj[..., :, 1:-1] += qy[..., :, :-2] - qy[..., :, 1:-1]
        adj[..., :, -1] += qy[..., :, -2]
        return adj

"""Resolution-related distortions in k-space."""

from __future__ import annotations

import numpy as np

from mri_recon.distortions.base import BaseDistortion


def _frequency_grids(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return fftshifted Cartesian frequency grids in cycles/pixel.

    The returned arrays have shape ``(ny, nx)`` and are aligned with a centered
    k-space convention (DC at image center).
    """

    ny, nx = shape[-2:]
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    return np.meshgrid(kx, ky, indexing="xy")


def _radial_frequency(shape: tuple[int, ...]) -> np.ndarray:
    """Return radial frequency normalized to ``[0, 1]`` on the sampled grid."""

    kx, ky = _frequency_grids(shape)
    radius = np.sqrt(kx * kx + ky * ky)
    max_radius = float(np.max(radius))
    if max_radius <= 0.0:
        return np.zeros_like(radius)
    return radius / max_radius


def _interp_axis_last(
    data: np.ndarray,
    source_positions: np.ndarray,
) -> np.ndarray:
    """Linearly sample along the last axis with zero outside bounds."""

    reshaped = data.reshape(-1, data.shape[-1])
    sampled = np.empty(
        (reshaped.shape[0], source_positions.size),
        dtype=data.dtype,
    )
    grid = np.arange(data.shape[-1], dtype=np.float64)

    for row_index, row in enumerate(reshaped):
        sampled_real = np.interp(
            source_positions,
            grid,
            row.real,
            left=0.0,
            right=0.0,
        )
        sampled_imag = np.interp(
            source_positions,
            grid,
            row.imag,
            left=0.0,
            right=0.0,
        )
        sampled[row_index] = sampled_real + 1j * sampled_imag

    return sampled.reshape(*data.shape[:-1], source_positions.size)


class IsotropicResolutionReduction(BaseDistortion):
    """Low-pass truncation with a circular mask.

    This applies
    ``M_out(kx, ky) = M(kx, ky) * 1[r(kx, ky) <= K]``
    where ``K`` is ``radius_fraction`` on the normalized radial grid.
    """

    def __init__(self, radius_fraction: float = 0.6) -> None:
        if not 0.0 < radius_fraction <= 1.0:
            raise ValueError("radius_fraction must be in (0, 1]")
        self.radius_fraction = radius_fraction

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        kspace = self.to_numpy(data)
        # Hard radial cutoff: removes high-frequency detail isotropically.
        mask = _radial_frequency(kspace.shape) <= self.radius_fraction
        return kspace * mask


class AnisotropicResolutionChange(BaseDistortion):
    """Low-pass truncation with independent axis limits.

    This applies
    ``M_out = M * 1[|kx| <= Kx] * 1[|ky| <= Ky]``.
    """

    def __init__(
        self,
        kx_fraction: float = 0.8,
        ky_fraction: float = 0.6,
    ) -> None:
        if not 0.0 < kx_fraction <= 1.0:
            raise ValueError("kx_fraction must be in (0, 1]")
        if not 0.0 < ky_fraction <= 1.0:
            raise ValueError("ky_fraction must be in (0, 1]")
        self.kx_fraction = kx_fraction
        self.ky_fraction = ky_fraction

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        kspace = self.to_numpy(data)
        kx, ky = _frequency_grids(kspace.shape)
        kx_max = np.max(np.abs(kx)) or 1.0
        ky_max = np.max(np.abs(ky)) or 1.0
        # Different axis cutoffs produce direction-dependent blur.
        mask = (np.abs(kx) <= self.kx_fraction * kx_max) & (
            np.abs(ky) <= self.ky_fraction * ky_max
        )
        return kspace * mask


class ZeroFillDistortion(BaseDistortion):
    """Zero-pad centered k-space to increase grid size.

    Zero-filling changes sampling density in image space (smaller pixels) but
    does not increase the acquired bandwidth.
    """

    def __init__(self, pad_factor: float | tuple[float, float] = 2.0) -> None:
        if isinstance(pad_factor, tuple):
            fy, fx = pad_factor
        else:
            fy = fx = pad_factor
        if fy < 1.0 or fx < 1.0:
            raise ValueError("pad_factor must be >= 1")
        self.fy = float(fy)
        self.fx = float(fx)

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        kspace = self.to_numpy(data)
        ny, nx = kspace.shape[-2:]
        out_ny = max(int(round(ny * self.fy)), ny)
        out_nx = max(int(round(nx * self.fx)), nx)
        output = np.zeros(
            (*kspace.shape[:-2], out_ny, out_nx),
            dtype=kspace.dtype,
        )
        y0 = (out_ny - ny) // 2
        x0 = (out_nx - nx) // 2
        # Keep DC and existing samples centered in the larger array.
        output[..., y0:y0 + ny, x0:x0 + nx] = kspace
        return output


class PhaseEncodeDecimation(BaseDistortion):
    """Decimate phase-encode lines to reduce FOV and induce aliasing.

    Keeps every ``factor``-th ky line, with the center line preserved.
    """

    def __init__(self, factor: int = 2) -> None:
        if factor < 1:
            raise ValueError("factor must be >= 1")
        self.factor = factor

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        kspace = self.to_numpy(data)
        ny = kspace.shape[-2]
        mask = np.zeros(ny, dtype=bool)
        offset = (ny // 2) % self.factor
        mask[offset:: self.factor] = True
        # Unsampled phase-encode lines are set to zero.
        return kspace * mask[:, None]


class VariableDensityBandwidthReduction(BaseDistortion):
    """Smooth radial taper for controlled bandwidth reduction.

    Uses ``H(r) = exp(-(r/kappa)^2)`` on normalized radial frequency ``r``.
    """

    def __init__(self, kappa: float = 0.7) -> None:
        if not 0.0 < kappa:
            raise ValueError("kappa must be > 0")
        self.kappa = kappa

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        kspace = self.to_numpy(data)
        radius = _radial_frequency(kspace.shape)
        # Smooth taper reduces ringing compared to hard truncation.
        taper = np.exp(-((radius / self.kappa) ** 2))
        return kspace * taper


class CoordinateScaling(BaseDistortion):
    """Resample onto scaled k-space coordinates.

    Implements ``M_out(kx, ky) = M(alpha_x * kx, alpha_y * ky)`` with linear
    interpolation and zero padding outside sampled support.
    """

    def __init__(self, alpha_x: float = 1.0, alpha_y: float = 1.0) -> None:
        if alpha_x <= 0.0 or alpha_y <= 0.0:
            raise ValueError("alpha_x and alpha_y must be > 0")
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        kspace = self.to_numpy(data)
        ny, nx = kspace.shape[-2:]
        cx = (nx - 1) / 2.0
        cy = (ny - 1) / 2.0

        # Map output-grid coordinates to source-grid coordinates.
        source_x = (np.arange(nx) - cx) * self.alpha_x + cx
        source_y = (np.arange(ny) - cy) * self.alpha_y + cy

        # First interpolate along kx, then along ky.
        sampled_x = _interp_axis_last(kspace, source_x)
        sampled_xy = _interp_axis_last(
            np.swapaxes(sampled_x, -2, -1),
            source_y,
        )
        return np.swapaxes(sampled_xy, -2, -1)

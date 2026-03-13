"""Artifact-oriented k-space distortions for Cartesian MRI."""

from __future__ import annotations

import numpy as np

from mri_recon.distortions.base import BaseDistortion


def _kx_ky_axes(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return fftshifted frequency axes in cycles/pixel."""

    ny, nx = shape[-2:]
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    return kx, ky


class GibbsRingingDistortion(BaseDistortion):
    """Hard rectangular truncation that induces Gibbs ringing.

    Applies
    ``M_out = M * 1[|kx| <= Kx] * 1[|ky| <= Ky]``.
    """

    def __init__(
        self,
        kx_fraction: float = 0.7,
        ky_fraction: float = 0.7,
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
        kx, ky = _kx_ky_axes(kspace.shape)
        kx_max = np.max(np.abs(kx)) or 1.0
        ky_max = np.max(np.abs(ky)) or 1.0

        # Rectangular support cutoff keeps center band and removes outer lines.
        mask = (
            (np.abs(kx)[None, :] <= self.kx_fraction * kx_max)
            & (np.abs(ky)[:, None] <= self.ky_fraction * ky_max)
        )
        return kspace * mask


class AliasingWrapAroundDistortion(BaseDistortion):
    """Undersample phase-encode lines by factor ``R`` to create wrap-around."""

    def __init__(self, factor: int = 2) -> None:
        if factor < 1:
            raise ValueError("factor must be >= 1")
        self.factor = factor

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        kspace = self.to_numpy(data)
        ny = kspace.shape[-2]
        mask = np.zeros(ny, dtype=bool)

        # Preserve the centered ky line and then keep every R-th line.
        offset = (ny // 2) % self.factor
        mask[offset:: self.factor] = True
        return kspace * mask[:, None]


class EPINHalfGhostDistortion(BaseDistortion):
    """Apply odd/even line mismatch that produces EPI N/2 ghosting.

    Odd ky lines are multiplied by
    ``exp(i * phi) * exp(i * 2*pi*delta_x*kx)``.
    """

    def __init__(
        self,
        phase_offset_rad: float = 0.0,
        delta_x_pixels: float = 0.0,
    ) -> None:
        self.phase_offset_rad = float(phase_offset_rad)
        self.delta_x_pixels = float(delta_x_pixels)

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        kspace = np.asarray(self.to_numpy(data)).copy()
        ny = kspace.shape[-2]
        odd_lines = (np.arange(ny) % 2) == 1

        if self.phase_offset_rad != 0.0:
            # Constant odd/even phase mismatch.
            kspace[..., odd_lines, :] *= np.exp(1j * self.phase_offset_rad)

        if self.delta_x_pixels != 0.0:
            # Alternating readout shift encoded as kx phase ramp.
            kx, _ = _kx_ky_axes(kspace.shape)
            ramp = np.exp(1j * 2.0 * np.pi * self.delta_x_pixels * kx)
            kspace[..., odd_lines, :] *= ramp[None, :]

        return kspace


class LineByLineMotionGhostDistortion(BaseDistortion):
    """Apply line-wise translation phase terms to mimic motion ghosts.

    For each ky line ``l``, applies
    ``exp(-i*2*pi*(kx*dx_l + ky_l*dy_l))``.
    """

    def __init__(
        self,
        max_shift_x_pixels: float = 1.0,
        max_shift_y_pixels: float = 0.5,
        pattern: str = "ramp",
        block_size: int = 8,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.max_shift_x_pixels = float(max_shift_x_pixels)
        self.max_shift_y_pixels = float(max_shift_y_pixels)
        self.pattern = pattern
        self.block_size = block_size

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        kspace = self.to_numpy(data)
        ny = kspace.shape[-2]
        kx, ky = _kx_ky_axes(kspace.shape)
        shift_x, shift_y = self._line_shifts(ny)

        phase_x = np.exp(-1j * 2.0 * np.pi * shift_x[:, None] * kx[None, :])
        phase_y = np.exp(-1j * 2.0 * np.pi * shift_y[:, None] * ky[:, None])
        phase = phase_x * phase_y
        return kspace * phase

    def _line_shifts(self, ny: int) -> tuple[np.ndarray, np.ndarray]:
        line_index = np.arange(ny, dtype=np.float64)

        if self.pattern == "ramp":
            shift_x = np.linspace(
                -self.max_shift_x_pixels,
                self.max_shift_x_pixels,
                ny,
            )
            shift_y = np.linspace(
                -self.max_shift_y_pixels,
                self.max_shift_y_pixels,
                ny,
            )
            return shift_x, shift_y

        if self.pattern == "sinusoidal":
            phase = 2.0 * np.pi * line_index / max(ny, 1)
            shift_x = self.max_shift_x_pixels * np.sin(phase)
            shift_y = self.max_shift_y_pixels * np.cos(phase)
            return shift_x, shift_y

        if self.pattern == "step":
            block = (line_index // self.block_size).astype(int)
            sign = np.where((block % 2) == 0, 1.0, -1.0)
            shift_x = self.max_shift_x_pixels * sign
            shift_y = self.max_shift_y_pixels * sign
            return shift_x, shift_y

        raise ValueError("pattern must be one of: ramp, sinusoidal, step")


class OffResonanceDistortion(BaseDistortion):
    """Apply line-wise off-resonance phase accrual during readout.

    Uses
    ``M_l(kx) <- M_l(kx) * exp(i * 2*pi * Omega_l * t(kx))``
    with configurable ``Omega_l`` pattern along ky.
    """

    def __init__(
        self,
        omega_max: float = 1.0,
        omega_pattern: str = "linear",
        readout_time_scale: float = 1.0,
    ) -> None:
        self.omega_max = float(omega_max)
        self.omega_pattern = omega_pattern
        self.readout_time_scale = float(readout_time_scale)

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        kspace = self.to_numpy(data)
        ny = kspace.shape[-2]
        kx, _ = _kx_ky_axes(kspace.shape)

        omega = self._omega_profile(ny)
        kx_max = np.max(np.abs(kx)) or 1.0
        readout_time = self.readout_time_scale * (kx / kx_max)

        # Per-line phase accrual over readout time creates blur/distortion.
        phase = np.exp(
            1j * 2.0 * np.pi * omega[:, None] * readout_time[None, :]
        )
        return kspace * phase

    def _omega_profile(self, ny: int) -> np.ndarray:
        line_index = np.arange(ny, dtype=np.float64)

        if self.omega_pattern == "linear":
            return np.linspace(-self.omega_max, self.omega_max, ny)

        if self.omega_pattern == "sinusoidal":
            phase = 2.0 * np.pi * line_index / max(ny, 1)
            return self.omega_max * np.sin(phase)

        raise ValueError("omega_pattern must be one of: linear, sinusoidal")

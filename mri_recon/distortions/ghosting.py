"""Ghosting-related distortions in k-space."""

from __future__ import annotations

import torch

from mri_recon.distortions.base import (
    BaseDistortion,
    _validate_cartesian_kspace_tensor,
)


class PhaseEncodeGhostingDistortion(BaseDistortion):
    """Periodic line-wise phase inconsistency that creates image ghosts.

    The distortion multiplies every ``line_period``-th k-space line along the
    chosen axis by a constant complex factor. For Cartesian MRI, alternating
    line phase errors are a simple and realistic model for ghosting caused by
    timing mismatch, odd-even echo inconsistency, or periodic acquisition
    instability.

    :param int line_period: Period of the corrupted line pattern. Must be at
        least ``2``.
    :param int line_offset: Offset of the first corrupted line within the
        period. Must satisfy ``0 <= line_offset < line_period``.
    :param float phase_error_radians: Constant phase error applied to the
        selected lines.
    :param float corrupted_line_scale: Multiplicative magnitude applied to the
        selected lines. Must be positive.
    :param int ghost_axis: K-space axis on which to apply the periodic line
        inconsistency. The default ``-2`` corresponds to the phase-encode axis
        for 2D Cartesian k-space.
    """

    def __init__(
        self,
        line_period: int = 2,
        line_offset: int = 1,
        phase_error_radians: float = torch.pi / 2,
        corrupted_line_scale: float = 1.0,
        ghost_axis: int = -2,
    ) -> None:
        super().__init__()
        if line_period < 2:
            raise ValueError("line_period must be at least 2")
        if not 0 <= line_offset < line_period:
            raise ValueError("line_offset must satisfy 0 <= line_offset < line_period")
        if corrupted_line_scale <= 0.0:
            raise ValueError("corrupted_line_scale must be positive")
        if ghost_axis not in (-2, -1):
            raise ValueError("ghost_axis must be -2 or -1 for 2D k-space")

        self.line_period = int(line_period)
        self.line_offset = int(line_offset)
        self.phase_error_radians = float(phase_error_radians)
        self.corrupted_line_scale = float(corrupted_line_scale)
        self.ghost_axis = ghost_axis

    def _line_factor(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        magnitude = torch.tensor(
            self.corrupted_line_scale,
            dtype=dtype,
            device=device,
        )
        phase = torch.tensor(
            self.phase_error_radians,
            dtype=dtype,
            device=device,
        )
        return torch.polar(magnitude, phase)

    def _line_modulation(
        self,
        y_complex: torch.Tensor,
        axis_size: int,
        complex_axis: int,
        *,
        conjugate: bool = False,
    ) -> torch.Tensor:
        selected_lines = torch.arange(axis_size, device=y_complex.device)
        selected_lines = (selected_lines - self.line_offset) % self.line_period == 0

        # Start from an identity modulation and only alter the corrupted lines.
        modulation = torch.ones(
            axis_size,
            dtype=y_complex.dtype,
            device=y_complex.device,
        )

        line_factor = self._line_factor(
            dtype=y_complex.real.dtype,
            device=y_complex.device,
        )

        if conjugate:
            # The adjoint of a diagonal complex modulation uses conjugated phases.
            line_factor = torch.conj(line_factor)

        modulation[selected_lines] = line_factor

        # Reshape the 1D line pattern so it broadcasts across batch, coil, and
        # the untouched spatial axis of the complex k-space tensor.
        view_shape = [1] * y_complex.ndim
        view_shape[complex_axis] = axis_size
        return modulation.view(*view_shape)

    def _apply_line_modulation(
        self,
        y: torch.Tensor,
        *,
        conjugate: bool = False,
    ) -> torch.Tensor:
        if self.phase_error_radians == 0.0 and self.corrupted_line_scale == 1.0:
            return y

        _validate_cartesian_kspace_tensor(y)

        # Convert the stored real/imaginary channels into a complex tensor so
        # ghosting can be expressed as a diagonal multiplication in k-space.
        y_complex = torch.view_as_complex(y.movedim(1, -1).contiguous())

        # After removing the real/imaginary channel, the targeted k-space axis
        # shifts left by one position in the complex-valued tensor.
        axis = self.ghost_axis % y.ndim
        axis_size = y.shape[axis]
        complex_axis = axis - 1

        y_ghosted = y_complex * self._line_modulation(
            y_complex,
            axis_size,
            complex_axis,
            conjugate=conjugate,
        )
        return torch.view_as_real(y_ghosted).movedim(-1, 1).contiguous()

    def A(self, y: torch.Tensor) -> torch.Tensor:
        return self._apply_line_modulation(y, conjugate=False)

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self._apply_line_modulation(y, conjugate=True)

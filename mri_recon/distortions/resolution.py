"""Resolution-related distortions in k-space."""

from __future__ import annotations

import torch

from mri_recon.distortions.base import (
    BaseDistortion,
    _normalized_axis_frequencies,
    _radial_frequency,
)


class IsotropicResolutionReduction(BaseDistortion):
    """Low-pass truncation with a circular mask.

    This applies
    ``M_out(kx, ky) = M(kx, ky) * 1[r(kx, ky) <= K]``
    where ``K`` is ``radius_fraction`` on the normalized radial grid.

    :param float radius_fraction: Normalized cutoff radius in ``(0, 1]``.
        Frequencies outside this radius are set to zero.
    """

    def __init__(self, radius_fraction: float = 0.6) -> None:
        super().__init__()
        if not 0.0 < radius_fraction <= 1.0:
            raise ValueError("radius_fraction must be in (0, 1]")
        self.radius_fraction = radius_fraction

    def A(self, y: torch.Tensor) -> torch.Tensor:
        # Hard radial cutoff: removes high-frequency detail isotropically.
        mask = _radial_frequency(y.shape) <= self.radius_fraction
        mask = mask.to(y.device)
        return y * mask

    def A_adjoint(
        self,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return self.A(y)


class AnisotropicResolutionReduction(BaseDistortion):
    """Axis-aligned low-pass truncation with independent cutoffs.

    This applies a rectangular mask
    ``M_out(kx, ky) = M(kx, ky) * 1[|kx| <= Kx] * 1[|ky| <= Ky]``
    where ``Kx`` and ``Ky`` are the normalized cutoffs along the readout and
    phase-encode frequency axes respectively.

    :param float kx_radius_fraction: Normalized cutoff along the horizontal
        frequency axis in ``(0, 1]``.
    :param float ky_radius_fraction: Normalized cutoff along the vertical
        frequency axis in ``(0, 1]``.
    """

    def __init__(
        self,
        kx_radius_fraction: float = 1.0,
        ky_radius_fraction: float = 0.4,
    ) -> None:
        super().__init__()
        if not 0.0 < kx_radius_fraction <= 1.0:
            raise ValueError("kx_radius_fraction must be in (0, 1]")
        if not 0.0 < ky_radius_fraction <= 1.0:
            raise ValueError("ky_radius_fraction must be in (0, 1]")

        self.kx_radius_fraction = kx_radius_fraction
        self.ky_radius_fraction = ky_radius_fraction

    def _mask(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        normalized_kx, normalized_ky = _normalized_axis_frequencies(shape)

        # A rectangular passband models direction-dependent resolution loss,
        # such as stronger truncation along phase encode than along readout.
        mask = (normalized_kx <= self.kx_radius_fraction) & (
            normalized_ky <= self.ky_radius_fraction
        )
        return mask.to(device)

    def A(self, y: torch.Tensor) -> torch.Tensor:
        return y * self._mask(y.shape, y.device)

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.A(y)

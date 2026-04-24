"""Resolution-related distortions in k-space."""

from __future__ import annotations

import torch

from mri_recon.distortions.base import BaseDistortion


def _frequency_grids(shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fftshifted Cartesian frequency grids in cycles/pixel.

    The returned arrays have shape ``(ny, nx)`` and are aligned with a centered
    k-space convention (DC at image center).
    """
    ny, nx = shape[-2:]
    kx = torch.fft.fftshift(torch.fft.fftfreq(nx))
    ky = torch.fft.fftshift(torch.fft.fftfreq(ny))
    return torch.meshgrid(kx, ky, indexing="xy")


def _radial_frequency(shape: tuple[int, ...]) -> torch.Tensor:
    """Return radial frequency normalized to ``[0, 1]`` on the sampled grid."""

    kx, ky = _frequency_grids(shape)
    radius = torch.sqrt(kx * kx + ky * ky)
    max_radius = float(torch.max(radius))
    if max_radius <= 0.0:
        return torch.zeros_like(radius)
    return radius / max_radius


class IsotropicResolutionReduction(BaseDistortion):
    """Low-pass truncation with a circular mask.

    This applies
    ``M_out(kx, ky) = M(kx, ky) * 1[r(kx, ky) <= K]``
    where ``K`` is ``radius_fraction`` on the normalized radial grid.
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
        return self(y)

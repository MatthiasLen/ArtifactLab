"""Resolution-related distortions in k-space."""

from __future__ import annotations

import torch

from mri_recon.distortions.base import BaseDistortion, _radial_frequency


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
        return self(y)

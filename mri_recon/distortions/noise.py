"""Noise-related k-space distortions."""

from __future__ import annotations

import torch

from mri_recon.distortions.base import BaseDistortion


class GaussianNoiseDistortion(BaseDistortion):
    """Add additive Gaussian noise to the two-channel k-space tensor.

    Independent zero-mean Gaussian noise is added to each stored channel of the
    real-valued k-space representation.

    :param float sigma: Standard deviation of the additive noise.
    """

    def __init__(self, sigma: float = 0.00001) -> None:
        super().__init__()
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        self.sigma = sigma

    def A(self, y: torch.Tensor) -> torch.Tensor:
        if self.sigma == 0.0:
            return y
        return y + self.sigma * torch.randn_like(y)


class ComplexGaussianNoiseDistortion(BaseDistortion):
    """Additive complex Gaussian noise applied in k-space.

    This adds independent zero-mean Gaussian noise to the real and imaginary
    components of the complex-valued k-space measurements.

    :param float sigma: Standard deviation of the Gaussian noise added to each
        of the real and imaginary components.
    """

    def __init__(self, sigma: float = 0.00001) -> None:
        super().__init__()
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        self.sigma = sigma

    def A(self, y: torch.Tensor) -> torch.Tensor:
        if self.sigma == 0.0:
            return y

        y_complex = torch.view_as_complex(y.movedim(1, -1).contiguous())

        noise_real = self.sigma * torch.randn_like(y_complex.real)
        noise_imag = self.sigma * torch.randn_like(y_complex.imag)
        y_noisy = y_complex + noise_real + 1j * noise_imag

        return torch.view_as_real(y_noisy).movedim(-1, 1).contiguous()

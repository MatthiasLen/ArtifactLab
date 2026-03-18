"""Sharpness-related distortions in k-space."""

from __future__ import annotations

import numpy as np

from mri_recon.distortions.base import BaseDistortion
from mri_recon.distortions.resolution import (
    _frequency_grids,
    _radial_frequency,
)


class Apodization(BaseDistortion):
    """Apply smooth windows to suppress high frequencies.

    Multiplies k-space by a smooth window ``H(kx, ky)``.
    - Gaussian: ``exp(-(kx/kx0)^2 - (ky/ky0)^2)``
    - Hamming/Hann/Kaiser: separable 1D windows along each axis
    """

    def __init__(
        self,
        window: str = "gaussian",
        kappa_x: float = 0.7,
        kappa_y: float = 0.7,
        beta: float = 8.0,
    ) -> None:
        self.window = window.lower()
        if kappa_x <= 0.0 or kappa_y <= 0.0:
            raise ValueError("kappa_x and kappa_y must be > 0")
        self.kappa_x = kappa_x
        self.kappa_y = kappa_y
        self.beta = beta

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        kspace = self.to_numpy(data)
        ny, nx = kspace.shape[-2:]
        if self.window == "gaussian":
            kx, ky = _frequency_grids(kspace.shape)
            kx_norm = np.abs(kx) / (np.max(np.abs(kx)) or 1.0)
            ky_norm = np.abs(ky) / (np.max(np.abs(ky)) or 1.0)
            # Isotropic or anisotropic Gaussian attenuation in frequency space.
            win = np.exp(
                -(
                    (kx_norm / self.kappa_x) ** 2
                    + (ky_norm / self.kappa_y) ** 2
                )
            )
        elif self.window == "hamming":
            win = np.hamming(ny)[:, None] * np.hamming(nx)[None, :]
        elif self.window == "hann":
            win = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
        elif self.window == "kaiser":
            win = (
                np.kaiser(ny, self.beta)[:, None]
                * np.kaiser(nx, self.beta)[None, :]
            )
        else:
            raise ValueError(
                "window must be one of: gaussian, hamming, hann, kaiser"
            )
        return kspace * win


class DirectionalSharpnessControl(BaseDistortion):
    """Axis-specific apodization strength.

    Uses Gaussian apodization with different x/y scales to control directional
    blur and sharpness.
    """

    def __init__(self, kappa_x: float = 0.9, kappa_y: float = 0.5) -> None:
        self.apodization = Apodization(
            window="gaussian",
            kappa_x=kappa_x,
            kappa_y=kappa_y,
        )

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        return self.apodization.apply(data, **kwargs)


class HighFrequencyBoost(BaseDistortion):
    """Mild high-frequency amplification.

    Applies ``H(r) = 1 + beta * r^power`` on normalized radial frequency ``r``.
    """

    def __init__(self, beta: float = 0.2, power: float = 1.0) -> None:
        if beta < 0.0:
            raise ValueError("beta must be >= 0")
        if power < 1.0:
            raise ValueError("power must be >= 1")
        self.beta = beta
        self.power = power

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        kspace = self.to_numpy(data)
        radius = _radial_frequency(kspace.shape)
        # Gain increases with radius; high bands are emphasized most.
        gain = 1.0 + self.beta * (radius**self.power)
        return kspace * gain


class UnsharpMaskKspace(BaseDistortion):
    """Unsharp masking implemented directly in k-space.

    Uses ``H = 1 + beta * (1 - L)`` where ``L`` is a smooth low-pass window.
    """

    def __init__(self, beta: float = 0.2, lowpass_kappa: float = 0.6) -> None:
        if beta < 0.0:
            raise ValueError("beta must be >= 0")
        if lowpass_kappa <= 0.0:
            raise ValueError("lowpass_kappa must be > 0")
        self.beta = beta
        self.lowpass_kappa = lowpass_kappa

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        kspace = self.to_numpy(data)
        radius = _radial_frequency(kspace.shape)
        lowpass = np.exp(-((radius / self.lowpass_kappa) ** 2))
        # Preserve low frequencies, amplify residual high-frequency content.
        gain = 1.0 + self.beta * (1.0 - lowpass)
        return kspace * gain


class RegularizedInverseBlur(BaseDistortion):
    """Wiener-like regularized inverse of a known blur window.

    Applies ``H = conj(L) / (|L|^2 + lambda)`` where ``L`` is a known blur in
    k-space and ``lambda`` controls noise amplification.
    """

    def __init__(
        self,
        l2_weight: float = 1e-3,
        blur_window: np.ndarray | None = None,
        lowpass_kappa: float = 0.6,
    ) -> None:
        if l2_weight < 0.0:
            raise ValueError("l2_weight must be >= 0")
        if lowpass_kappa <= 0.0:
            raise ValueError("lowpass_kappa must be > 0")
        self.l2_weight = l2_weight
        self.blur_window = blur_window
        self.lowpass_kappa = lowpass_kappa

    def apply(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        kspace = self.to_numpy(data)
        if self.blur_window is None:
            radius = _radial_frequency(kspace.shape)
            blur = np.exp(-((radius / self.lowpass_kappa) ** 2))
        else:
            blur = np.asarray(self.blur_window)
            if blur.shape != kspace.shape[-2:]:
                raise ValueError(
                    "blur_window shape must match the "
                    "last two input dimensions"
                )
        # Regularized inverse filter avoids division by near-zero frequencies.
        inverse_filter = np.conj(blur) / (np.abs(blur) ** 2 + self.l2_weight)
        return kspace * inverse_filter

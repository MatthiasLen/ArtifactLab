"""Resolution-related distortions in k-space."""

from __future__ import annotations

import torch

from mri_recon.distortions.base import (
    BaseDistortion,
    _normalized_axis_frequencies,
    _radial_frequency,
)


def _smooth_radial_low_pass_mask(
    shape: tuple[int, ...],
    device: torch.device,
    radius_fraction: float,
    transition_fraction: float,
    profile: str,
    beta: float | None = None,
) -> torch.Tensor:
    """Build a smooth circular low-pass mask on the normalized radial grid.

    The mask is one inside the passband, zero outside the cutoff, and follows
    the requested taper profile inside the transition band.

    :param tuple[int, ...] shape: Input k-space tensor shape.
    :param torch.device device: Device on which to allocate the mask.
    :param float radius_fraction: Normalized radial cutoff in ``(0, 1]``.
    :param float transition_fraction: Fraction of the cutoff radius reserved
        for the taper band in ``[0, 1]``.
    :param str profile: Taper profile name. Supported values are ``"hann"``
        and ``"kaiser"``.
    :param float | None beta: Kaiser window shape parameter. Required when
        ``profile`` is ``"kaiser"``.
    :returns: Real-valued radial mask with entries in ``[0, 1]``.
    :rtype: torch.Tensor
    """
    radius = _radial_frequency(shape).to(device)
    transition_start = radius_fraction * (1.0 - transition_fraction)

    if transition_fraction == 0.0:
        return (radius <= radius_fraction).to(dtype=radius.dtype)

    mask = torch.zeros_like(radius)
    passband = radius <= transition_start
    transition_band = (radius > transition_start) & (radius <= radius_fraction)

    mask[passband] = 1.0
    if not torch.any(transition_band):
        return mask

    scaled_radius = (radius[transition_band] - transition_start) / (
        radius_fraction - transition_start
    )

    if profile == "hann":
        mask[transition_band] = 0.5 * (1.0 + torch.cos(torch.pi * scaled_radius))
        return mask

    if profile == "kaiser":
        if beta is None or beta <= 0.0:
            raise ValueError("beta must be positive for a Kaiser taper")

        beta_tensor = radius.new_tensor(beta)
        denominator = torch.i0(beta_tensor)
        edge_value = 1.0 / denominator
        raw = torch.i0(beta_tensor * torch.sqrt(1.0 - scaled_radius.square())) / denominator
        mask[transition_band] = (raw - edge_value) / (1.0 - edge_value)
        return mask.clamp(0.0, 1.0)

    raise ValueError(f"Unknown taper profile {profile!r}")


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


class HannTaperResolutionReduction(BaseDistortion):
    """Radial low-pass reduction with a Hann transition band.

    The mask equals ``1`` in the low-frequency passband, tapers smoothly to
    ``0`` with a raised-cosine profile near the cutoff, and is exactly ``0``
    beyond ``radius_fraction``.

    See https://en.wikipedia.org/wiki/Hann_function for details on the Hann window.

    :param float radius_fraction: Normalized cutoff radius in ``(0, 1]``.
        Frequencies outside this radius are fully suppressed.
    :param float transition_fraction: Fraction of the cutoff radius occupied by
        the smooth transition in ``[0, 1]``. ``0`` recovers the hard cutoff.
    """

    def __init__(
        self,
        radius_fraction: float = 0.6,
        transition_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        if not 0.0 < radius_fraction <= 1.0:
            raise ValueError("radius_fraction must be in (0, 1]")
        if not 0.0 <= transition_fraction <= 1.0:
            raise ValueError("transition_fraction must be in [0, 1]")
        self.radius_fraction = radius_fraction
        self.transition_fraction = transition_fraction

    def _mask(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return _smooth_radial_low_pass_mask(
            shape=shape,
            device=device,
            radius_fraction=self.radius_fraction,
            transition_fraction=self.transition_fraction,
            profile="hann",
        )

    def A(self, y: torch.Tensor) -> torch.Tensor:
        return y * self._mask(y.shape, y.device)

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.A(y)


class KaiserTaperResolutionReduction(BaseDistortion):
    """Radial low-pass reduction with a Kaiser transition band.

    The mask equals ``1`` in the low-frequency passband, tapers smoothly to
    ``0`` with a Kaiser-profile transition near the cutoff, and is exactly
    ``0`` beyond ``radius_fraction``.

    See https://en.wikipedia.org/wiki/Kaiser_window for details on the Kaiser window.

    :param float radius_fraction: Normalized cutoff radius in ``(0, 1]``.
        Frequencies outside this radius are fully suppressed.
    :param float transition_fraction: Fraction of the cutoff radius occupied by
        the smooth transition in ``[0, 1]``. ``0`` recovers the hard cutoff.
    :param float beta: Positive Kaiser shape parameter. Larger values create a
        steeper transition inside the taper band.
    """

    def __init__(
        self,
        radius_fraction: float = 0.6,
        transition_fraction: float = 0.25,
        beta: float = 8.6,
    ) -> None:
        super().__init__()
        if not 0.0 < radius_fraction <= 1.0:
            raise ValueError("radius_fraction must be in (0, 1]")
        if not 0.0 <= transition_fraction <= 1.0:
            raise ValueError("transition_fraction must be in [0, 1]")
        if beta <= 0.0:
            raise ValueError("beta must be positive")

        self.radius_fraction = radius_fraction
        self.transition_fraction = transition_fraction
        self.beta = beta

    def _mask(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return _smooth_radial_low_pass_mask(
            shape=shape,
            device=device,
            radius_fraction=self.radius_fraction,
            transition_fraction=self.transition_fraction,
            profile="kaiser",
            beta=self.beta,
        )

    def A(self, y: torch.Tensor) -> torch.Tensor:
        return y * self._mask(y.shape, y.device)

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.A(y)

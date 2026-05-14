"""Resolution-related distortions in k-space."""

from __future__ import annotations

import torch

from mri_recon.distortions.base import (
    SelfAdjointMultiplicativeMaskDistortion,
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


class IsotropicResolutionReduction(SelfAdjointMultiplicativeMaskDistortion):
    """Isotropic in-plane resolution reduction with a circular hard cutoff.

    This distortion keeps only a centered circular region of k-space:
    ``M_out(kx, ky) = M(kx, ky) * 1[r(kx, ky) <= K]``, where ``K`` is the
    retained radial support on the normalized frequency grid.

    In MRI terms, this models reduced in-plane spatial resolution at fixed
    field of view by removing high-frequency content equally in all in-plane
    directions. The reconstructed image keeps the same matrix size, but fine
    detail is lost because the maximum sampled k-space extent is smaller. The
    resulting effect is isotropic blur from limited k-space support.

    The parameter ``radius_fraction`` controls how much of the centered
    low-frequency region is retained. Smaller values preserve only the k-space
    core and therefore produce stronger blur and a broader point-spread
    function. A value of ``1.0`` keeps the full sampled support and recovers
    the identity operator.

    Compared with :class:`AnisotropicResolutionReduction`, this class applies
    the same reduction in all directions rather than separately along readout
    and phase encode. Compared with :class:`CartesianUndersampling`, it models
    resolution loss by shrinking the sampled support, not by skipping lines
    within the original support.

    :param float radius_fraction: Fraction of the original centered radial
        k-space support retained in ``(0, 1]``. Smaller values correspond to
        stronger isotropic resolution reduction.
    """

    def __init__(self, radius_fraction: float = 0.6) -> None:
        super().__init__()
        if not 0.0 < radius_fraction <= 1.0:
            raise ValueError("radius_fraction must be in (0, 1]")
        self.radius_fraction = radius_fraction

    def _mask(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        # Hard radial cutoff: removes high-frequency detail isotropically.
        mask = _radial_frequency(shape) <= self.radius_fraction
        return mask.to(device)


class AnisotropicResolutionReduction(SelfAdjointMultiplicativeMaskDistortion):
    """Axis-aligned Cartesian resolution reduction with independent cutoffs.

    This distortion keeps only a centered rectangular region of Cartesian
    k-space and zeros the remaining outer frequencies:
    ``M_out(kx, ky) = M(kx, ky) * 1[|kx| <= Kx] * 1[|ky| <= Ky]``.

    In MRI terms, this models reduced in-plane acquisition resolution at fixed
    field of view. The reconstructed image keeps the same matrix size, but its
    effective spatial resolution decreases because less high-frequency k-space
    support is retained. The resulting artifact is blur from limited k-space
    extent, not aliasing from undersampling.

    The two parameters control the retained centered k-space support along the
    Cartesian encoding axes. ``kx_radius_fraction`` corresponds to the retained
    readout-direction frequency extent, while ``ky_radius_fraction``
    corresponds to the retained phase-encode-direction frequency extent.
    Reducing either parameter broadens the point-spread function along the
    corresponding image direction. A typical MRI-like setting keeps
    ``kx_radius_fraction`` close to ``1.0`` and reduces
    ``ky_radius_fraction``, reflecting that protocols often sacrifice more
    phase-encode resolution than readout resolution.

    This is a hard rectangular cutoff. If a softer edge is desired, use one of
    the taper-based resolution distortions instead.

    :param float kx_radius_fraction: Fraction of the original centered
        k-space extent retained along the horizontal frequency axis in
        ``(0, 1]``. This corresponds to the retained readout-direction
        resolution support. ``1.0`` keeps the full sampled readout extent.
    :param float ky_radius_fraction: Fraction of the original centered
        k-space extent retained along the vertical frequency axis in
        ``(0, 1]``. This corresponds to the retained phase-encode-direction
        resolution support. Smaller values produce stronger blur along the
        corresponding image direction.
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

        # Centered rectangular support models reduced acquired Cartesian
        # resolution, often stronger along phase encode than along readout.
        mask = (normalized_kx <= self.kx_radius_fraction) & (
            normalized_ky <= self.ky_radius_fraction
        )
        return mask.to(device)


class HannTaperResolutionReduction(SelfAdjointMultiplicativeMaskDistortion):
    """Isotropic resolution reduction with a Hann-tapered radial cutoff.

    This distortion keeps a centered circular low-frequency region and tapers
    the mask smoothly to zero near the cutoff using a raised-cosine Hann
    profile. Inside the passband the mask equals ``1``; inside the transition
    band it decreases smoothly from ``1`` to ``0``; beyond the cutoff it is
    exactly ``0``.

    In MRI terms, this is still a resolution-reduction operator: it suppresses
    high spatial frequencies and therefore lowers effective in-plane spatial
    resolution at fixed field of view. Relative to
    :class:`IsotropicResolutionReduction`, the main difference is not the type
    of resolution loss but the edge behavior of the k-space support. The smooth
    transition reduces ringing that a hard truncation can introduce, at the
    cost of making the support edge less sharp.

    ``radius_fraction`` sets the outer radial extent of the retained support.
    ``transition_fraction`` sets how much of that outer region is devoted to
    the smooth taper. Setting ``transition_fraction=0`` recovers the hard
    circular cutoff. Larger transition fractions make the cutoff gentler and
    behave more like k-space apodization.

    Compared with :class:`CartesianUndersampling`, this class does not skip
    phase-encode lines within the original support. It reduces resolution by
    attenuating and removing high frequencies, leading primarily to blur rather
    than undersampling aliasing.

    See https://en.wikipedia.org/wiki/Hann_function for details on the Hann window.

    :param float radius_fraction: Fraction of the original centered radial
        k-space support retained in ``(0, 1]``. Frequencies outside this radius
        are fully suppressed.
    :param float transition_fraction: Fraction of the cutoff radius occupied by
        the smooth Hann transition in ``[0, 1]``. ``0`` recovers the hard
        circular cutoff.
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


class KaiserTaperResolutionReduction(SelfAdjointMultiplicativeMaskDistortion):
    """Isotropic resolution reduction with a Kaiser-tapered radial cutoff.

    This distortion keeps a centered circular low-frequency region and tapers
    the mask smoothly to zero near the cutoff using a Kaiser-window profile.
    Inside the passband the mask equals ``1``; inside the transition band it
    decreases smoothly from ``1`` to ``0``; beyond the cutoff it is exactly
    ``0``.

    In MRI terms, this lowers effective in-plane spatial resolution at fixed
    field of view by reducing the retained high-frequency k-space extent. As
    with :class:`HannTaperResolutionReduction`, the purpose of the taper is to
    soften the hard support edge and reduce ringing, while still producing blur
    from lost high-frequency information.

    ``radius_fraction`` sets the outer radial extent of the retained support.
    ``transition_fraction`` controls the width of the taper band. ``beta``
    controls the shape of the Kaiser taper within that band: larger values
    produce a steeper transition, while smaller positive values produce a more
    gradual roll-off. Setting ``transition_fraction=0`` recovers the hard
    circular cutoff regardless of ``beta``.

    Compared with :class:`CartesianUndersampling`, this class reduces
    resolution by shrinking and tapering the effective k-space support, rather
    than by skipping lines from the original grid. The dominant image effect is
    blur and apodization, not aliasing from sub-Nyquist sampling.

    See https://en.wikipedia.org/wiki/Kaiser_window for details on the Kaiser window.

    :param float radius_fraction: Fraction of the original centered radial
        k-space support retained in ``(0, 1]``. Frequencies outside this radius
        are fully suppressed.
    :param float transition_fraction: Fraction of the cutoff radius occupied by
        the smooth Kaiser transition in ``[0, 1]``. ``0`` recovers the hard
        circular cutoff.
    :param float beta: Positive Kaiser shape parameter controlling how steeply
        the taper falls inside the transition band.
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


class RadialHighPassEmphasisDistortion(SelfAdjointMultiplicativeMaskDistortion):
    """Radially boost high frequencies with a smooth monotone gain field.

    The mask equals ``1`` in the low-frequency core, rises smoothly across a
    fixed transition band, and reaches ``1 + alpha`` at the sampled edge. This
    behaves like a gentle high-frequency shelf rather than amplifying all
    nonzero frequencies.

    :param float alpha: Non-negative gain added at the k-space edge.
    :param float boost_start_radius: Normalized radius in ``[0, 1)`` where the
        high-frequency shelf begins to rise.
    :param float boost_end_radius: Normalized radius in ``(0, 1]`` where the
        shelf reaches its full gain.
    """

    BOOST_START_RADIUS = 0.4
    BOOST_END_RADIUS = 0.9

    def __init__(
        self,
        alpha: float = 0.4,
        boost_start_radius: float = BOOST_START_RADIUS,
        boost_end_radius: float = BOOST_END_RADIUS,
    ) -> None:
        super().__init__()
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 <= boost_start_radius < 1.0:
            raise ValueError("boost_start_radius must be in [0, 1)")
        if not 0.0 < boost_end_radius <= 1.0:
            raise ValueError("boost_end_radius must be in (0, 1]")
        if boost_start_radius >= boost_end_radius:
            raise ValueError("boost_start_radius must be smaller than boost_end_radius")

        self.alpha = alpha
        self.boost_start_radius = boost_start_radius
        self.boost_end_radius = boost_end_radius

    def _mask(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        radius = _radial_frequency(shape).to(device)
        transition = (radius - self.boost_start_radius) / (
            self.boost_end_radius - self.boost_start_radius
        )
        transition = transition.clamp(0.0, 1.0)
        transition = transition * transition * (3.0 - 2.0 * transition)
        return 1.0 + self.alpha * transition

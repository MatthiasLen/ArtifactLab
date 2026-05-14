"""Bias-field related distortions in k-space."""

from __future__ import annotations

import torch

from mri_recon.distortions.base import BaseDistortion, _normalized_frequency_grids


def _gaussian_bias_gain_field(
    shape: tuple[int, ...],
    device: torch.device,
    width_x_fraction: float,
    width_y_fraction: float,
    center_x_fraction: float,
    center_y_fraction: float,
    edge_gain: float,
) -> torch.Tensor:
    """Build a normalized Gaussian multiplicative gain field on the sampled grid."""

    kx, ky = _normalized_frequency_grids(shape)
    kx = kx.to(device)
    ky = ky.to(device)
    dx = (kx - center_x_fraction) / width_x_fraction
    dy = (ky - center_y_fraction) / width_y_fraction
    gaussian = torch.exp(-0.5 * (dx * dx + dy * dy))

    edge_value = float(gaussian.min())
    if edge_value < 1.0:
        gaussian = (gaussian - edge_value) / (1.0 - edge_value)
    gaussian = gaussian.clamp(0.0, 1.0)

    gain = edge_gain + (1.0 - edge_gain) * gaussian
    return gain / gain.max()


class GaussianKspaceBiasField(BaseDistortion):
    """Smooth centered multiplicative bias field in k-space.

    The gain is radial, equals ``1`` at DC, and smoothly decays toward
    ``edge_gain`` at the edge of the sampled k-space grid.

    :param float width_fraction: Radial width of the Gaussian envelope on the
        normalized k-space grid.
    :param float edge_gain: Gain approached near the edge of k-space. Must lie
        in ``(0, 1]``.
    """

    def __init__(self, width_fraction: float = 0.35, edge_gain: float = 0.4) -> None:
        super().__init__()
        if width_fraction <= 0.0:
            raise ValueError("width_fraction must be positive")
        if not 0.0 < edge_gain <= 1.0:
            raise ValueError("edge_gain must be in (0, 1]")
        self.width_fraction = width_fraction
        self.edge_gain = edge_gain

    def _gain_field(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return _gaussian_bias_gain_field(
            shape=shape,
            device=device,
            width_x_fraction=self.width_fraction,
            width_y_fraction=self.width_fraction,
            center_x_fraction=0.0,
            center_y_fraction=0.0,
            edge_gain=self.edge_gain,
        )

    def A(self, y: torch.Tensor) -> torch.Tensor:
        gain = self._gain_field(y.shape, y.device)
        return y * gain

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.A(y)


class OffCenterAnisotropicGaussianKspaceBiasField(BaseDistortion):
    """Off-center anisotropic Gaussian multiplicative bias field in k-space.

    The gain peaks at an offset location in k-space, decays with different widths
    along ``kx`` and ``ky``, and is normalized to unit maximum on the sampled grid.

    Note: This class can also approximate a readout-decay-like blur when used in a
    centered anisotropic configuration. To mimic stronger attenuation along the
    readout direction, keep the Gaussian centered at DC with
    ``center_x_fraction=0.0`` and ``center_y_fraction=0.0``, choose a narrower
    width along the readout axis than along the orthogonal axis, and reduce
    ``edge_gain`` below ``1.0``. For example, a setting such as
    ``width_x_fraction < width_y_fraction`` with a moderate ``edge_gain``
    produces a smooth directional loss of high-frequency content that can
    resemble readout-decay blur.

    This remains a phenomenological approximation rather than an explicit
    time-ordered readout-decay model. It applies a smooth multiplicative
    k-space weighting, not a physically parameterized echo-train decay.

    :param float width_x_fraction: Gaussian width along the normalized ``kx``
        direction. Smaller values produce stronger attenuation away from the
        center along ``kx``. When ``kx`` is the readout axis, choosing
        ``width_x_fraction < width_y_fraction`` approximates readout-direction
        blur.
    :param float width_y_fraction: Gaussian width along the normalized ``ky``
        direction. Larger values preserve more support along ``ky`` relative
        to ``kx``.
    :param float center_x_fraction: Center offset along normalized ``kx`` in
        ``[-1, 1]``. Use ``0.0`` for a centered readout-decay-like
        approximation.
    :param float center_y_fraction: Center offset along normalized ``ky`` in
        ``[-1, 1]``. Use ``0.0`` for a centered readout-decay-like
        approximation.
    :param float edge_gain: Baseline gain far from the Gaussian peak. Must lie
        in ``(0, 1]``. Smaller values strengthen the peripheral attenuation
        and therefore the resulting directional blur.
    """

    def __init__(
        self,
        width_x_fraction: float = 0.2,
        width_y_fraction: float = 0.35,
        center_x_fraction: float = 0.15,
        center_y_fraction: float = -0.1,
        edge_gain: float = 0.3,
    ) -> None:
        super().__init__()
        if width_x_fraction <= 0.0 or width_y_fraction <= 0.0:
            raise ValueError("width fractions must be positive")
        if abs(center_x_fraction) > 1.0 or abs(center_y_fraction) > 1.0:
            raise ValueError("center fractions must be in [-1, 1]")
        if not 0.0 < edge_gain <= 1.0:
            raise ValueError("edge_gain must be in (0, 1]")
        self.width_x_fraction = width_x_fraction
        self.width_y_fraction = width_y_fraction
        self.center_x_fraction = center_x_fraction
        self.center_y_fraction = center_y_fraction
        self.edge_gain = edge_gain

    def _gain_field(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return _gaussian_bias_gain_field(
            shape=shape,
            device=device,
            width_x_fraction=self.width_x_fraction,
            width_y_fraction=self.width_y_fraction,
            center_x_fraction=self.center_x_fraction,
            center_y_fraction=self.center_y_fraction,
            edge_gain=self.edge_gain,
        )

    def A(self, y: torch.Tensor) -> torch.Tensor:
        gain = self._gain_field(y.shape, y.device)
        return y * gain

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.A(y)

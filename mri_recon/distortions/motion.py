"""Motion-related distortions in k-space."""

from __future__ import annotations

import torch

from mri_recon.distortions.base import BaseDistortion, _frequency_grids


class TranslationMotionDistortion(BaseDistortion):
    """Rigid in-plane translation represented as a phase ramp in k-space.

    Translating the object in image space by ``(shift_y_pixels, shift_x_pixels)``
    corresponds to multiplying k-space by a unit-modulus complex phase ramp.

    :param float shift_x_pixels: Horizontal image-space translation in pixels.
    :param float shift_y_pixels: Vertical image-space translation in pixels.
    """

    def __init__(self, shift_x_pixels: float = 8.0, shift_y_pixels: float = 4.0) -> None:
        super().__init__()
        self.shift_x_pixels = shift_x_pixels
        self.shift_y_pixels = shift_y_pixels

    def _phase_ramp(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        kx, ky = _frequency_grids(shape)
        kx = kx.to(device)
        ky = ky.to(device)

        phase = -2.0 * torch.pi * (kx * self.shift_x_pixels + ky * self.shift_y_pixels)
        return torch.polar(torch.ones_like(phase), phase)

    def A(self, y: torch.Tensor) -> torch.Tensor:
        if self.shift_x_pixels == 0.0 and self.shift_y_pixels == 0.0:
            return y

        y_complex = torch.view_as_complex(y.movedim(1, -1).contiguous())
        y_moved = y_complex * self._phase_ramp(y.shape, y.device)
        return torch.view_as_real(y_moved).movedim(-1, 1).contiguous()

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        if self.shift_x_pixels == 0.0 and self.shift_y_pixels == 0.0:
            return y

        y_complex = torch.view_as_complex(y.movedim(1, -1).contiguous())
        y_moved = y_complex * torch.conj(self._phase_ramp(y.shape, y.device))
        return torch.view_as_real(y_moved).movedim(-1, 1).contiguous()

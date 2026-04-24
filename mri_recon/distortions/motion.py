"""Motion-related distortions in k-space."""

from __future__ import annotations

import torch

from mri_recon.distortions.base import BaseDistortion, _frequency_grids


def _validate_cartesian_kspace_tensor(y: torch.Tensor) -> None:
    """Validate the repository's 2D Cartesian k-space tensor convention."""

    if y.ndim not in (4, 5):
        raise ValueError(
            "Expected k-space with shape (B, 2, H, W) or (B, 2, N, H, W), "
            f"got tensor with shape {tuple(y.shape)}"
        )
    if y.shape[1] != 2:
        raise ValueError(
            "Expected real/imaginary channel dimension of size 2 at axis 1, "
            f"got shape {tuple(y.shape)}"
        )
    if not torch.is_floating_point(y):
        raise TypeError(f"Expected floating-point real/imaginary tensor, got dtype {y.dtype}")
    if y.shape[-2] <= 0 or y.shape[-1] <= 0:
        raise ValueError(f"Spatial k-space dimensions must be positive, got shape {tuple(y.shape)}")


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

        _validate_cartesian_kspace_tensor(y)
        y_complex = torch.view_as_complex(y.movedim(1, -1).contiguous())
        y_moved = y_complex * self._phase_ramp(y.shape, y.device)
        return torch.view_as_real(y_moved).movedim(-1, 1).contiguous()

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        if self.shift_x_pixels == 0.0 and self.shift_y_pixels == 0.0:
            return y

        _validate_cartesian_kspace_tensor(y)
        y_complex = torch.view_as_complex(y.movedim(1, -1).contiguous())
        y_moved = y_complex * torch.conj(self._phase_ramp(y.shape, y.device))
        return torch.view_as_real(y_moved).movedim(-1, 1).contiguous()


class SegmentedTranslationMotionDistortion(BaseDistortion):
    """Piecewise translation motion applied across k-space acquisition segments.

    This approximates acquisition-time motion using only Cartesian k-space data.
    The phase-encode dimension is split into contiguous segments, and each segment
    is modulated by the phase ramp corresponding to a different image-space
    translation. The resulting k-space is internally inconsistent across segments,
    which produces realistic ghosting and motion artifacts in the reconstruction.

    Intuition: we want to model a Cartesian MRI scan that is not acquired all at once.
    Instead, one line block of k-space is measured, then the next, and so on. If the patient
    moves between those acquisition blocks, each block is still individually consistent
    with a translated object, but the full stitched k-space no longer corresponds
    to any single static image. Reconstruction therefore tries to explain mutually
    inconsistent k-space segments with one image, which manifests as ghosting,
    blurring, and replicated structure rather than a simple global shift.

    :param tuple[float, ...] shift_x_pixels: Horizontal image-space translations,
        one per segment, in pixels.
    :param tuple[float, ...] shift_y_pixels: Vertical image-space translations,
        one per segment, in pixels.
    :param int segment_axis: K-space axis to segment. The default ``-2`` applies
        motion changes across the phase-encode lines.
    """

    def __init__(
        self,
        shift_x_pixels: tuple[float, ...] = (0.0, 2.0, 5.0, 5.0),
        shift_y_pixels: tuple[float, ...] = (0.0, 1.0, 2.0, 2.0),
        segment_axis: int = -2,
    ) -> None:
        super().__init__()
        # Each segment must have one horizontal and one vertical translation.
        if len(shift_x_pixels) == 0 or len(shift_y_pixels) == 0:
            raise ValueError("shift_x_pixels and shift_y_pixels must be non-empty")

        if len(shift_x_pixels) != len(shift_y_pixels):
            raise ValueError("shift_x_pixels and shift_y_pixels must have the same length")

        # Restrict to the two spatial k-space axes for 2D Cartesian data.
        if segment_axis not in (-2, -1):
            raise ValueError("segment_axis must be -2 or -1 for 2D k-space")

        self.shift_x_pixels = tuple(float(shift) for shift in shift_x_pixels)
        self.shift_y_pixels = tuple(float(shift) for shift in shift_y_pixels)
        self.segment_axis = segment_axis

    def _phase_ramp(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        shift_x_pixels: float,
        shift_y_pixels: float,
    ) -> torch.Tensor:
        # Build the centered Cartesian frequency coordinates used by the MRI FFT.
        kx, ky = _frequency_grids(shape)
        kx = kx.to(device)
        ky = ky.to(device)
        # A spatial translation becomes a linear phase term in k-space.
        phase = -2.0 * torch.pi * (kx * shift_x_pixels + ky * shift_y_pixels)
        return torch.polar(torch.ones_like(phase), phase)

    def _segment_slices(self, shape: tuple[int, ...]) -> list[slice]:
        # Resolve which k-space axis is segmented and split it into contiguous blocks.
        axis = self.segment_axis % len(shape)
        axis_size = shape[axis]
        if len(self.shift_x_pixels) > axis_size:
            raise ValueError(
                f"Cannot split axis of length {axis_size} into {len(self.shift_x_pixels)} non-empty segments"
            )
        boundaries = torch.linspace(0, axis_size, len(self.shift_x_pixels) + 1, dtype=torch.int64)
        return [
            slice(int(boundaries[i]), int(boundaries[i + 1])) for i in range(len(boundaries) - 1)
        ]

    def _apply_segmented_phase(self, y: torch.Tensor, conjugate: bool = False) -> torch.Tensor:
        # If every segment has zero motion, the operator is exactly the identity.
        if all(shift == 0.0 for shift in self.shift_x_pixels) and all(
            shift == 0.0 for shift in self.shift_y_pixels
        ):
            return y

        _validate_cartesian_kspace_tensor(y)

        # Convert the real/imaginary channel representation into a complex tensor so
        # that the phase modulation can be applied as a complex multiplication.
        y_complex = torch.view_as_complex(y.movedim(1, -1).contiguous())

        # Start from a copy of the original k-space and overwrite one acquisition
        # segment at a time with the motion-corrupted version.
        y_distorted = y_complex.clone()

        # After removing the real/imaginary channel, the segmented axis shifts left
        # by one position in the complex-valued tensor.
        complex_axis = (self.segment_axis % y.ndim) - 1

        for segment_slice, shift_x, shift_y in zip(
            self._segment_slices(y.shape), self.shift_x_pixels, self.shift_y_pixels, strict=True
        ):
            # Build the phase ramp for the translation active during this segment.
            ramp = self._phase_ramp(y.shape, y.device, shift_x, shift_y)
            if conjugate:
                # The adjoint uses the complex-conjugated phase ramp.
                ramp = torch.conj(ramp)

            # Select the current segment in the complex k-space tensor.
            selection = [slice(None)] * y_complex.ndim
            selection[complex_axis] = segment_slice

            if complex_axis == y_complex.ndim - 2:
                # Segment along phase-encode lines: keep only the corresponding rows.
                ramp_segment = ramp[segment_slice, :]
            else:
                # Segment along readout samples: keep only the corresponding columns.
                ramp_segment = ramp[:, segment_slice]

            # Replace this block with the same data multiplied by its segment-specific
            # translation phase. Broadcasting handles batch and coil dimensions.
            y_distorted[tuple(selection)] = y_complex[tuple(selection)] * ramp_segment

        # Convert back to the repository's two-channel real/imaginary convention.
        return torch.view_as_real(y_distorted).movedim(-1, 1).contiguous()

    def A(self, y: torch.Tensor) -> torch.Tensor:
        return self._apply_segmented_phase(y, conjugate=False)

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self._apply_segmented_phase(y, conjugate=True)

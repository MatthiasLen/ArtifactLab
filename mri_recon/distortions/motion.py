"""Motion-related distortions in k-space."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mri_recon.distortions.base import (
    BaseDistortion,
    _frequency_grids,
    _validate_cartesian_kspace_tensor,
)


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


class RotationalMotionDistortion(BaseDistortion):
    """Rigid in-plane rotation applied by resampling centered Cartesian k-space.

    A rigid image-space rotation by ``angle_radians`` corresponds to the same
    coordinate rotation in the centered Fourier domain. On the discrete sampled
    grid this becomes an interpolation problem over the stored real and
    imaginary k-space channels.

    :param float angle_radians: In-plane rotation angle in radians.
    """

    def __init__(self, angle_radians: float = torch.pi / 12) -> None:
        super().__init__()
        self.angle_radians = float(angle_radians)

    def _reshape_kspace_channels(self, y: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Flatten batch and optional coil axes for channel-wise resampling."""

        if y.ndim == 4:
            return y, ()

        leading_shape = (y.shape[0], y.shape[2])
        y_flat = y.permute(0, 2, 1, 3, 4).reshape(-1, 2, y.shape[-2], y.shape[-1])
        return y_flat, leading_shape

    def _restore_kspace_channels(
        self, y_flat: torch.Tensor, leading_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Restore flattened batch and coil axes after resampling."""

        if not leading_shape:
            return y_flat

        batch_size, coil_count = leading_shape
        return y_flat.reshape(batch_size, coil_count, 2, *y_flat.shape[-2:]).permute(0, 2, 1, 3, 4)

    def _rotation_grid(
        self, shape: tuple[int, ...], device: torch.device, angle_radians: float
    ) -> torch.Tensor:
        """Build a sampling grid for a centered in-plane coordinate rotation."""

        batch_size = shape[0]

        # The affine matrix acts in normalized centered coordinates, so no
        # translation term is needed here.
        angle = torch.tensor(angle_radians, device=device, dtype=torch.float32)
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        theta = torch.tensor(
            [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0]],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)
        theta = theta.expand(batch_size, -1, -1)
        return F.affine_grid(
            theta,
            size=[batch_size, 2, shape[-2], shape[-1]],
            align_corners=False,
        )

    def _rotate_kspace(self, y: torch.Tensor, angle_radians: float) -> torch.Tensor:
        """Rotate the stored real and imaginary k-space channels together."""

        y_flat, leading_shape = self._reshape_kspace_channels(y)
        grid = self._rotation_grid(y_flat.shape, y.device, angle_radians)
        # Real and imaginary channels are resampled together so the complex
        # k-space phase remains internally consistent.
        rotated_channels = F.grid_sample(
            y_flat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return self._restore_kspace_channels(rotated_channels, leading_shape)

    def _apply_rotation(self, y: torch.Tensor, angle_radians: float) -> torch.Tensor:
        """Apply the centered k-space rotation and return the rotated samples."""

        if angle_radians == 0.0:
            return y

        _validate_cartesian_kspace_tensor(y)
        return self._rotate_kspace(y, angle_radians)

    def _apply_rotation_adjoint(self, y: torch.Tensor, angle_radians: float) -> torch.Tensor:
        """Apply the exact adjoint of the implemented interpolation operator."""

        if angle_radians == 0.0:
            return y

        _validate_cartesian_kspace_tensor(y)

        # Reverse-angle resampling is not the adjoint once interpolation and
        # zero-padding enter the operator. Use the vector-Jacobian product of
        # the actual forward map so reconstruction methods see the correct
        # linear adjoint of the implemented distortion.
        with torch.enable_grad():
            probe = torch.zeros_like(y, requires_grad=True)

            def forward_fn(probe_input: torch.Tensor) -> torch.Tensor:
                return self._apply_rotation(probe_input, angle_radians)

            _, adjoint = torch.autograd.functional.vjp(
                forward_fn,
                probe,
                v=y.detach(),
                create_graph=False,
                strict=False,
            )

        return adjoint.detach()

    def A(self, y: torch.Tensor) -> torch.Tensor:
        """Rotate the centered k-space samples corresponding to image motion."""

        return self._apply_rotation(y, self.angle_radians)

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the exact adjoint of the implemented rotation operator."""

        return self._apply_rotation_adjoint(y, self.angle_radians)


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

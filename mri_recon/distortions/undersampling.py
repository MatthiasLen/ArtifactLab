"""Cartesian k-space undersampling distortions."""

from __future__ import annotations

import torch

from mri_recon.distortions.base import SelfAdjointMultiplicativeMaskDistortion


PATTERNS = {"uniform_random", "variable_density_random", "equispaced"}
PARTIAL_FOURIER_SIDES = {"low", "high"}

# Lorentzian offset that controls how steeply the variable-density weights
# fall off with normalized distance from k-space center.
# Smaller values → steeper density gradient; 0.05 gives ~20x weight ratio
# between the ACS boundary and the k-space edge.
_VD_WEIGHT_OFFSET: float = 0.05


class CartesianUndersampling(SelfAdjointMultiplicativeMaskDistortion):
    """Cartesian k-space undersampling along one encoding direction.

    This distortion simulates sub-Nyquist MRI acquisition by keeping only a
    subset of Cartesian k-space lines along a chosen axis, phase encode by
    default. A contiguous low-frequency center region (ACS) may be retained
    fully, while the peripheral lines are sampled using a configurable random
    or equispaced pattern.

    This is fundamentally different from resolution reduction. In
    resolution-reduction distortions, the maximum retained k-space extent is
    reduced, which primarily causes blur because high spatial frequencies are
    absent. In Cartesian undersampling, the original k-space extent is still
    targeted, but many lines inside that extent are skipped. The dominant
    consequence is aliasing or incoherent undersampling artifact, not a simple
    broader point-spread function.

    ``keep_fraction`` controls the total sampling budget along the chosen axis.
    ``center_fraction`` reserves part of that budget for a fully sampled ACS
    block near k-space center, which is important for many parallel imaging and
    learned reconstruction pipelines. ``pattern`` controls how the remaining
    peripheral lines are selected. ``axis`` determines which Cartesian encoding
    direction is undersampled. ``seed`` makes the random patterns reproducible.

    The mask is deterministic for a given shape, device, and seed.

    :param float keep_fraction: Fraction of lines kept along the undersampled
        axis in ``(0, 1]``. For example, ``0.25`` keeps 25% of the lines and
        corresponds to approximately 4x acceleration when applied to a single
        encoding direction.
    :param float center_fraction: Fraction of lines along the undersampled axis
        reserved for a contiguous, fully sampled ACS region in ``[0, 1]``.
        Defaults to ``0.5 * keep_fraction`` so that part of the sampling budget
        remains available for peripheral sampling. ``0`` disables the ACS block.
    :param str pattern: Peripheral sampling pattern. Supported values are
        ``"uniform_random"``, ``"variable_density_random"``, and
        ``"equispaced"``. Variable-density sampling favors low-frequency lines
        near the ACS boundary; equispaced sampling is deterministic.
    :param int axis: Axis along which to undersample. The default ``-2``
        corresponds to phase encode for the repository's standard 2D k-space
        convention. Other values allow undersampling along readout or depth.
    :param int | None seed: Random seed for reproducible mask generation.
        Ignored by the deterministic ``"equispaced"`` pattern.
    """

    def __init__(
        self,
        keep_fraction: float = 0.25,
        center_fraction: float | None = None,
        pattern: str = "variable_density_random",
        axis: int = -2,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")

        if axis not in (-1, -2, -3):
            raise ValueError(f"axis must be -1, -2, or -3, got {axis}")

        if pattern not in PATTERNS:
            raise ValueError(f"pattern must be one of {sorted(PATTERNS)}, got {pattern!r}")

        if center_fraction is None:
            # Reserve half of the kept lines for a contiguous ACS block and
            # leave the remainder for peripheral sampling.
            center_fraction = 0.5 * keep_fraction
        elif not 0.0 <= center_fraction <= 1.0:
            raise ValueError(f"center_fraction must be in [0, 1], got {center_fraction}")

        if center_fraction > keep_fraction:
            raise ValueError(
                f"center_fraction ({center_fraction}) must not exceed "
                f"keep_fraction ({keep_fraction})"
            )

        self.keep_fraction = keep_fraction
        self.center_fraction = center_fraction
        self.pattern = pattern
        self.axis = axis
        self.seed = seed
        self._cached_mask = None
        self._cached_shape = None
        self._cached_device: torch.device | None = None

    def _mask(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate a binary Cartesian undersampling mask.

        The mask is applied along the specified axis (phase-encode by default).
        It keeps a contiguous center region fully sampled when requested and
        randomly undersamples the periphery to achieve the desired keep_fraction.

        :param tuple[int, ...] shape: k-space tensor shape.
        :param torch.device device: Device for the mask.
        :returns: Binary mask broadcastable to shape.
        :rtype: torch.Tensor
        """
        # Cache the mask keyed on both shape and device so that repeated GPU
        # forward passes do not perform a CPU→GPU copy on every call.
        if (
            self._cached_mask is not None
            and self._cached_shape == shape
            and self._cached_device == device
        ):
            return self._cached_mask

        # Get the size along the undersampling axis
        axis_size = shape[self.axis]

        # Set random seed for reproducibility
        rng_state = None
        if self.seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(self.seed)

        try:
            # Generate the 1D mask along the specified axis
            mask_1d = self._generate_1d_mask(axis_size)

            # Expand the 1D mask to match the full shape
            mask = self._expand_mask_to_shape(mask_1d, shape)
        finally:
            # Restore random state if we set a seed
            if self.seed is not None and rng_state is not None:
                torch.set_rng_state(rng_state)

        # Move to the target device and cache, including the device.
        mask = mask.to(device)
        self._cached_mask = mask
        self._cached_shape = shape
        self._cached_device = device

        return mask

    def _generate_1d_mask(self, axis_size: int) -> torch.Tensor:
        """Generate a 1D binary mask along the undersampling axis.

        :param int axis_size: Size of the axis along which to undersample.
        :returns: 1D binary mask of shape (axis_size,).
        :rtype: torch.Tensor
        """
        # Calculate number of lines to keep and center region size.
        num_keep = max(1, int(round(axis_size * self.keep_fraction)))
        num_center = int(round(axis_size * self.center_fraction))

        # Ensure center is not larger than total kept lines
        num_center = min(num_center, num_keep)

        # Initialize mask (all zeros)
        mask = torch.zeros(axis_size, dtype=torch.float32)

        # Keep the contiguous center region (ACS) when requested.
        center_start = (axis_size - num_center) // 2
        center_end = center_start + num_center
        if num_center > 0:
            mask[center_start:center_end] = 1.0

        peripheral_indices = self._peripheral_indices(center_start, center_end, axis_size)

        # Sample from the peripheral region with the requested pattern.
        if num_keep > num_center:
            num_peripheral = num_keep - num_center
            selected_indices = self._select_peripheral_indices(
                peripheral_indices=peripheral_indices,
                num_peripheral=num_peripheral,
                axis_size=axis_size,
            )
            mask[selected_indices] = 1.0

        return mask

    def _peripheral_indices(
        self,
        center_start: int,
        center_end: int,
        axis_size: int,
    ) -> torch.Tensor:
        """Return line indices outside the contiguous ACS region."""

        return torch.cat(
            [
                torch.arange(0, center_start, dtype=torch.long),
                torch.arange(center_end, axis_size, dtype=torch.long),
            ]
        )

    def _select_peripheral_indices(
        self,
        peripheral_indices: torch.Tensor,
        num_peripheral: int,
        axis_size: int,
    ) -> torch.Tensor:
        """Select peripheral lines according to the configured sampling pattern."""

        if self.pattern == "uniform_random":
            permutation = torch.randperm(len(peripheral_indices))[:num_peripheral]
            return peripheral_indices[permutation]

        if self.pattern == "variable_density_random":
            center = 0.5 * (axis_size - 1)
            distances = torch.abs(peripheral_indices.to(torch.float32) - center)
            normalized_distances = distances / max(center, 1.0)

            # Favor low-frequency lines near the ACS boundary while preserving
            # non-zero probability across the periphery.
            weights = 1.0 / (_VD_WEIGHT_OFFSET + normalized_distances.square())
            selected_positions = torch.multinomial(
                weights,
                num_samples=num_peripheral,
                replacement=False,
            )
            return peripheral_indices[selected_positions]

        if self.pattern == "equispaced":
            return self._select_equispaced_indices(peripheral_indices, num_peripheral)

        raise RuntimeError(f"Unsupported pattern {self.pattern!r}")

    def _select_equispaced_indices(
        self,
        peripheral_indices: torch.Tensor,
        num_peripheral: int,
    ) -> torch.Tensor:
        """Select evenly spaced peripheral indices.

        Lines are spaced uniformly within the peripheral index array.
        Because the peripheral array has a gap at the ACS region, the spacing
        between selected k-space lines will be approximately doubled near the
        ACS boundary compared to the spacing in the outer periphery.
        This pattern is fully deterministic and unaffected by ``seed``.
        """

        if num_peripheral >= len(peripheral_indices):
            return peripheral_indices

        # The early-return above guarantees step > 1.0, which in turn means
        # floor((idx + 0.5) * step) is strictly increasing, so no collision
        # resolution is needed.
        step = len(peripheral_indices) / num_peripheral
        positions = (
            (torch.arange(num_peripheral, dtype=torch.float32) * step + 0.5 * step).floor().long()
        )

        return peripheral_indices[positions]

    def _expand_mask_to_shape(
        self, mask_1d: torch.Tensor, target_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Expand a 1D mask to the full k-space shape.

        The 1D mask is applied along the specified axis and broadcast to all
        other dimensions.

        :param torch.Tensor mask_1d: 1D binary mask.
        :param tuple[int, ...] target_shape: Target shape for expansion.
        :returns: Mask broadcastable to target_shape.
        :rtype: torch.Tensor
        """
        # Convert negative axis to positive
        ndim = len(target_shape)
        axis = self.axis if self.axis >= 0 else ndim + self.axis

        # Create the full shape with 1s for broadcast dimensions
        expand_shape = list(target_shape)
        for i in range(len(expand_shape)):
            if i != axis:
                expand_shape[i] = 1

        # Reshape the 1D mask to the expand shape
        mask = mask_1d.reshape(expand_shape)

        return mask


class PartialFourierDistortion(SelfAdjointMultiplicativeMaskDistortion):
    """Asymmetric contiguous Cartesian mask for partial Fourier acquisition.

    This distortion simulates partial Fourier MRI acquisition by keeping a
    contiguous asymmetric region of k-space along one encoding axis while
    preserving a centered low-frequency block. Unlike symmetric resolution
    reduction, the retained support extends farther on one side of k-space than
    the other. Unlike Cartesian undersampling, the retained region is
    contiguous rather than sparse throughout the original support.

    The distortion models the acquired k-space mask only. It does not attempt
    to reconstruct or infer the missing region with homodyne, POCS, or any
    other partial-Fourier-specific reconstruction method.

    :param float partial_fraction: Fraction of lines retained along the chosen
        axis in ``[0.5, 1]``. ``1.0`` recovers the identity operator.
    :param float center_fraction: Fraction of lines reserved for a centered,
        fully retained low-frequency block in ``[0, 1]``. This block must not
        exceed ``partial_fraction``.
    :param int axis: Axis along which to apply the asymmetric truncation. The
        default ``-2`` matches the repository's standard phase-encode axis.
    :param str side: Side that retains more support outside the centered block.
        Supported values are ``"low"`` and ``"high"``.
    """

    def __init__(
        self,
        partial_fraction: float = 0.7,
        center_fraction: float = 0.1,
        axis: int = -2,
        side: str = "high",
    ) -> None:
        super().__init__()

        if not 0.5 <= partial_fraction <= 1.0:
            raise ValueError(f"partial_fraction must be in [0.5, 1], got {partial_fraction}")
        if not 0.0 <= center_fraction <= 1.0:
            raise ValueError(f"center_fraction must be in [0, 1], got {center_fraction}")
        if center_fraction > partial_fraction:
            raise ValueError(
                f"center_fraction ({center_fraction}) must not exceed "
                f"partial_fraction ({partial_fraction})"
            )
        if axis not in (-1, -2, -3):
            raise ValueError(f"axis must be -1, -2, or -3, got {axis}")
        if side not in PARTIAL_FOURIER_SIDES:
            raise ValueError(f"side must be one of {sorted(PARTIAL_FOURIER_SIDES)}, got {side!r}")

        self.partial_fraction = partial_fraction
        self.center_fraction = center_fraction
        self.axis = axis
        self.side = side
        self._cached_mask = None
        self._cached_shape = None
        self._cached_device: torch.device | None = None

    def _mask(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate a deterministic partial Fourier mask.

        :param tuple[int, ...] shape: k-space tensor shape.
        :param torch.device device: Device for the mask.
        :returns: Binary mask broadcastable to ``shape``.
        :rtype: torch.Tensor
        """
        if (
            self._cached_mask is not None
            and self._cached_shape == shape
            and self._cached_device == device
        ):
            return self._cached_mask

        axis_size = shape[self.axis]
        mask_1d = self._generate_1d_mask(axis_size)
        mask = self._expand_mask_to_shape(mask_1d, shape).to(device)

        self._cached_mask = mask
        self._cached_shape = shape
        self._cached_device = device
        return mask

    def _generate_1d_mask(self, axis_size: int) -> torch.Tensor:
        """Generate a 1D contiguous asymmetric partial Fourier mask."""
        num_keep = max(1, int(round(axis_size * self.partial_fraction)))
        num_center = int(round(axis_size * self.center_fraction))
        num_center = min(num_center, num_keep)

        mask = torch.zeros(axis_size, dtype=torch.float32)

        center_start = (axis_size - num_center) // 2
        center_end = center_start + num_center
        remaining = num_keep - num_center

        low_available = center_start
        high_available = axis_size - center_end

        if self.side == "high":
            extra_high = min(remaining, high_available)
            extra_low = remaining - extra_high
        else:
            extra_low = min(remaining, low_available)
            extra_high = remaining - extra_low

        start = center_start - extra_low
        end = center_end + extra_high

        mask[start:end] = 1.0
        return mask

    def _expand_mask_to_shape(
        self, mask_1d: torch.Tensor, target_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Expand a 1D mask to the full k-space shape."""
        ndim = len(target_shape)
        axis = self.axis if self.axis >= 0 else ndim + self.axis

        expand_shape = list(target_shape)
        for i in range(len(expand_shape)):
            if i != axis:
                expand_shape[i] = 1

        return mask_1d.reshape(expand_shape)

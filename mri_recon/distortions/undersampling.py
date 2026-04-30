"""Cartesian k-space undersampling distortions."""

from __future__ import annotations

from math import floor

import torch

from mri_recon.distortions.base import SelfAdjointMultiplicativeMaskDistortion


PATTERNS = {"uniform_random", "variable_density_random", "equispaced"}


class CartesianUndersampling(SelfAdjointMultiplicativeMaskDistortion):
    """Cartesian k-space undersampling along phase-encode direction.

    This distortion simulates true MRI acquisition undersampling by applying a
    binary sampling mask along the phase-encode direction (default). The mask
    keeps a contiguous center region (ACS - Auto-Calibration Signal) fully
    sampled and randomly undersamples the periphery.

    The mask is deterministic given a shape and seed, ensuring reproducibility.

    :param float keep_fraction: Fraction of phase-encode lines to keep in
        ``(0, 1]``. For example, 0.25 keeps 25% of phase-encode lines.
    :param float center_fraction: Fraction of phase-encode lines reserved for
        the contiguous, fully-sampled ACS region in ``(0, 1]``. Defaults to
        ``0.5 * keep_fraction``, leaving part of the acquisition budget for
        randomized peripheral line sampling.
    :param str pattern: Peripheral sampling pattern. Supported values are
        ``"uniform_random"``, ``"variable_density_random"``, and
        ``"equispaced"``. Defaults to ``"variable_density_random"``.
    :param int axis: Axis along which to apply undersampling. Default is -2
        (phase-encode for 4D tensors), can also be -3 for 5D tensors.
    :param int | None seed: Random seed for reproducible mask generation.
        If None, uses unseeded randomness (not recommended for reproducibility).
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

        if axis not in (-2, -3):
            raise ValueError(f"axis must be -2 or -3, got {axis}")

        if pattern not in PATTERNS:
            raise ValueError(f"pattern must be one of {sorted(PATTERNS)}, got {pattern!r}")

        if center_fraction is None:
            # Reserve half of the kept lines for a contiguous ACS block and
            # leave the remainder for peripheral sampling.
            center_fraction = 0.5 * keep_fraction
        elif not 0.0 < center_fraction <= 1.0:
            raise ValueError(f"center_fraction must be in (0, 1], got {center_fraction}")

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

    def _mask(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate a binary Cartesian undersampling mask.

        The mask is applied along the specified axis (phase-encode by default).
        It keeps a contiguous center region fully sampled and randomly undersamples
        the periphery to achieve the desired keep_fraction.

        :param tuple[int, ...] shape: k-space tensor shape.
        :param torch.device device: Device for the mask.
        :returns: Binary mask broadcastable to shape.
        :rtype: torch.Tensor
        """
        # Cache the mask if shape hasn't changed
        if self._cached_mask is not None and self._cached_shape == shape:
            return self._cached_mask.to(device)

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

        # Cache the mask
        self._cached_mask = mask
        self._cached_shape = shape

        return mask.to(device)

    def _generate_1d_mask(self, axis_size: int) -> torch.Tensor:
        """Generate a 1D binary mask along the undersampling axis.

        :param int axis_size: Size of the axis along which to undersample.
        :returns: 1D binary mask of shape (axis_size,).
        :rtype: torch.Tensor
        """
        # Calculate number of lines to keep and center region size
        num_keep = max(1, int(round(axis_size * self.keep_fraction)))
        num_center = max(1, int(round(axis_size * self.center_fraction)))

        # Ensure center is not larger than total kept lines
        num_center = min(num_center, num_keep)

        # Initialize mask (all zeros)
        mask = torch.zeros(axis_size, dtype=torch.float32)

        # Keep the contiguous center region (ACS)
        center_start = (axis_size - num_center) // 2
        center_end = center_start + num_center
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
            weights = 1.0 / (0.05 + normalized_distances.square())
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
        """Select approximately evenly spaced peripheral indices."""

        if num_peripheral >= len(peripheral_indices):
            return peripheral_indices

        step = len(peripheral_indices) / num_peripheral
        candidate_positions = []
        seen_positions = set()
        for idx in range(num_peripheral):
            position = min(len(peripheral_indices) - 1, floor((idx + 0.5) * step))
            while position in seen_positions and position + 1 < len(peripheral_indices):
                position += 1
            while position in seen_positions and position - 1 >= 0:
                position -= 1
            if position in seen_positions:
                continue
            seen_positions.add(position)
            candidate_positions.append(position)

        return peripheral_indices[torch.tensor(candidate_positions, dtype=torch.long)]

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

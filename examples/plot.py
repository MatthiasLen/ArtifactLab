"""Plotting helpers shared by example scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _kspace_to_log_magnitude(kspace: torch.Tensor) -> torch.Tensor:
    """Convert k-space tensor to log-magnitude image for visualization."""

    if kspace.ndim == 4:
        kspace = kspace[0]
    if kspace.ndim != 3 or kspace.shape[0] != 2:
        raise ValueError(
            f"Expected k-space with shape (2, H, W) or (1, 2, H, W), got {tuple(kspace.shape)}"
        )

    kspace = kspace.detach().cpu()
    kspace_complex = torch.view_as_complex(kspace.permute(1, 2, 0).contiguous())
    magnitude = torch.log1p(torch.abs(kspace_complex))

    lower = torch.quantile(magnitude, 0.05)
    upper = torch.quantile(magnitude, 0.995)
    if float(upper) > float(lower):
        magnitude = magnitude.clamp(lower, upper)
        magnitude = (magnitude - lower) / (upper - lower)
    else:
        mag_max = float(magnitude.max())
        if mag_max > 0.0:
            magnitude = magnitude / mag_max

    return torch.sqrt(magnitude)


def save_kspace_plot(
    clean_kspace: torch.Tensor,
    distorted_kspace: torch.Tensor,
    save_fn: Path,
    distortion_label: str,
) -> None:
    """Save side-by-side log-magnitude visualizations of clean and distorted k-space."""

    images = [
        ("Original k-space", _kspace_to_log_magnitude(clean_kspace)),
        ("Distorted k-space", _kspace_to_log_magnitude(distorted_kspace)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    fig.suptitle(f"Distortion: {distortion_label}")
    for ax, (title, image) in zip(axes, images, strict=True):
        ax.imshow(image.numpy(), cmap="magma")
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(save_fn, dpi=200, bbox_inches="tight")
    plt.close(fig)

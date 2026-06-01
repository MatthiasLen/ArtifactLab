"""Plotting helpers shared across examples and utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np
import deepinv as dinv


def _kspace_to_log_magnitude(kspace: torch.Tensor) -> torch.Tensor:
    """Convert k-space tensor to a log-magnitude image for visualization."""

    if kspace.ndim == 5:
        # show only middle coil for visualization
        # (1, 2, C, H, W) -> (2, H, W)
        kspace = kspace[0, :, kspace.shape[2] // 2]
    elif kspace.ndim == 4:
        # (1, 2, H, W) -> (2, H, W)
        kspace = kspace[0]
    elif kspace.ndim == 3:
        pass
        # (2, H, W) -> (2, H, W)
    else:
        raise ValueError(
            f"Expected k-space with shape (2, H, W) or (1, 2, H, W) or (1, 2, C, H, W),got {tuple(kspace.shape)}"
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

    print("transforming k-space to log-magnitude images for visualization...")
    print(f"\tclean k-space shape: {clean_kspace.shape}")
    print(f"\tdistorted k-space shape: {distorted_kspace.shape}")

    images = [
        ("Original k-space", _kspace_to_log_magnitude(clean_kspace)),
        ("Distorted k-space", _kspace_to_log_magnitude(distorted_kspace)),
    ]

    print(f"clean k-space magnitude shape: {images[0][1].shape}")
    print(f"distorted k-space magnitude shape: {images[1][1].shape}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    fig.suptitle(f"Distortion: {distortion_label}")
    for ax, (title, image) in zip(axes, images, strict=True):
        ax.imshow(image.numpy(), cmap="magma")
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(save_fn, dpi=200, bbox_inches="tight")
    plt.close(fig)


def convert_image_for_save(im: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor image complex tensor to a real-valued NumPy array 
    by calculating the magnitude.
    (B, 2, H, W)  or (B, H, W) with complex type -> (B, H, W)

    Args:
        im (torch.Tensor): The input image tensor.

    Returns:
        np.ndarray: The converted image array.
    """
    if torch.is_complex(im) or im.shape[1] == 2:
        im = dinv.utils.signals.complex_abs(im, dim=1, keepdim=False)
    return im.numpy()

"""Plotting helpers shared across examples and utilities."""

from __future__ import annotations

from ast import literal_eval
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np
import deepinv as dinv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mri_recon


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
    return im.detach().cpu().numpy()


def find_all(string, substring):
    indices = []
    start = 0
    while True:
        start = string.find(substring, start)
        if start == -1:
            break
        indices.append(start)
        start += len(substring)  # Move past the last found substring
    return indices


def get_metadata_from_filename(filename: str) -> dict:
    # remove parent directories from filename
    filename = os.path.basename(filename)

    # remove file extension from filename
    filename = filename.replace(".tiff", "")

    add = ""
    if "_N4" in filename:
        filename = filename.replace("_N4", "")
        add = "+ N4 BF Corr"

    if "_uncorrected" in filename:
        filename = filename.replace("_uncorrected", "")
    elif "_corrected" in filename:
        if "GaussianNoise" in filename:
            add += "+ Noise Corr"
        elif "ReduceResolution" not in filename:
            add += "+ corr"
        filename = filename.replace("_corrected", "")

    filename_parts = filename.split("_")
    # [image_or_kspace, dataset_part1, (dataset_part2,) sample_name, distortion_or_reference, (reconstruction)]

    metadata = {
        "image_type": "unknown",  # image or kspace
        "dataset": "unknown",  # dataset name
        "sample_name": "unknown",  # sample_name
        "distortion": "unknown",
        "parameters": "",
        "add": add,
        "reconstruction_method": "unknown",
    }
    if "reference" in filename_parts:
        if len(filename_parts) == 4:
            metadata = {
                "image_type": filename_parts[0],  # image or kspace
                "dataset": filename_parts[1],  # dataset name
                "sample_name": filename_parts[2],  # sample_name
                "distortion": "reference",
                "parameters": [],
                "add": add,
                "reconstruction_method": "reference",
            }

        elif len(filename_parts) == 5:
            metadata = {
                "image_type": filename_parts[0],  # image or kspace
                "dataset": filename_parts[1] + "_" + filename_parts[2],  # dataset name
                "sample_name": filename_parts[3],  # sample_name
                "distortion": "reference",
                "parameters": [],
                "add": add,
                "reconstruction_method": "reference",
            }
        else:
            print(f"Warning: Unexpected filename format for reference example path: {filename}")

    else:
        if len(filename_parts) == 5:
            metadata = {
                "image_type": filename_parts[0],  # image or kspace
                "dataset": filename_parts[1],  # dataset name
                "sample_name": filename_parts[2],  # sample_name
                "distortion": filename_parts[3],  # distortion with parameters
                "parameters": [],
                "add": add,
                "reconstruction_method": filename_parts[4],  # reconstruction
            }
        elif len(filename_parts) == 6:
            metadata = {
                "image_type": filename_parts[0],  # image or kspace
                "dataset": filename_parts[1] + "_" + filename_parts[2],  # dataset name
                "sample_name": filename_parts[3],  # sample_name
                "distortion": filename_parts[4],  # distortion with parameters
                "parameters": [],
                "add": add,
                "reconstruction_method": filename_parts[5],  # reconstruction
            }
        else:
            print(f"Warning: Unexpected filename format for path: {filename}")

    # split distortion_type and parameters:
    if "=" in metadata["distortion"]:
        distortion = metadata["distortion"].split("=")[0][:-1]
        if distortion != "ReduceResolution":
            # get paramter names of distortion from the distortion class in dinv.distortions
            if distortion in ["GaussianNoise"]:
                distortion_class = getattr(mri_recon.distortions, distortion + "Distortion")
                distortion_param_names = distortion_class.__init__.__code__.co_varnames[
                    1 : distortion_class.__init__.__code__.co_argcount
                ]
            else:
                distortion_class = getattr(mri_recon.distortions, distortion)
                distortion_param_names = distortion_class.__init__.__code__.co_varnames[
                    1 : distortion_class.__init__.__code__.co_argcount
                ]
        else:
            distortion_param_names = ["factor"]
        # parse parameters_str into a list of parameters
        parameters_str = (
            metadata["distortion"].split("=")[0][-1]
            + "="
            + "=".join(metadata["distortion"].split("=")[1:])
        )
        parameter_indices = find_all(parameters_str, "=")
        # example: "e=0.05" -> ["0.05"]
        if len(parameter_indices) != len(distortion_param_names):
            print(
                f"Warning: Number of parameters in filename ({len(parameter_indices)}) does not match number of parameters in distortion class ({len(distortion_param_names)}) for distortion {distortion}."
            )
            print("Filename: ", filename)
            print("expected parameter names: ", distortion_param_names)
            parameters = {}
        else:
            parameters = {}
            for i in range(len(parameter_indices)):
                if i == len(parameter_indices) - 1:
                    parameters[distortion_param_names[i]] = literal_eval(
                        parameters_str[parameter_indices[i] + 1 :]
                    )
                else:
                    parameters[distortion_param_names[i]] = literal_eval(
                        parameters_str[parameter_indices[i] + 1 : parameter_indices[i + 1] - 1]
                    )
        metadata["distortion"] = distortion
        metadata["parameters"] = parameters

    return metadata

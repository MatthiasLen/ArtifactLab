"""Inference various reconstructors for various distortion operators.

Usage:
    python examples/fastmri_inference_plot.py --source ../ram-experiments/data/fastmri/knee/singlecoil_val
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import deepinv as dinv

from mri_recon.distortions import *
from mri_recon.reconstruction import *

REPORT_DIR = Path("reports") / "fastmri_inference_plot"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ALGORITHMS = [
    "zero-filled",
    # "conjugate-gradient",
    # "ram",
    # "dip",
    "tv-pgd",
    # "wavelet-fista",
    # "tv-fista",
    # "tv-pdhg",
]
DISTORTIONS = [
    "Translation motion",
    "Off-center anisotropic Gaussian bias field",
    "Gaussian bias field",
    "Gaussian noise",
    "Isotropic LP",
]
METRICS = [
    "PSNR",
    "NMSE",
    "SSIM",
    "HaarPSI",
    "SharpnessIndex",
    "BlurStrength",
]


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
    y_clean: torch.Tensor,
    y_distorted: torch.Tensor,
    save_fn: Path,
    distortion_name: str,
) -> None:
    images = [
        ("Original k-space", _kspace_to_log_magnitude(y_clean)),
        ("Distorted k-space", _kspace_to_log_magnitude(y_distorted)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    fig.suptitle(f"Distortion: {distortion_name}")
    for ax, (title, image) in zip(axes, images, strict=True):
        ax.imshow(image.numpy(), cmap="magma")
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(save_fn, dpi=200, bbox_inches="tight")
    plt.close(fig)


def choose_algorithm(
    name: str,
    img_size: tuple = (640, 368),
    device: torch.device = "cpu",
    verbose: bool = False,
) -> dinv.models.Reconstructor:
    match name:
        case "zero-filled":
            return ZeroFilledReconstructor()
        case "conjugate-gradient":
            return ConjugateGradientReconstructor(max_iter=20)
        case "ram":
            return RAMReconstructor(default_sigma=0.05, device=device)
        case "dip":
            return DeepImagePriorReconstructor(img_size=img_size, n_iter=100, verbose=verbose)
        case "tv-pgd":
            return TVPGDReconstructor(n_iter=100, verbose=verbose)
        case "tv-fista":
            return TVFISTAReconstructor(n_iter=200, verbose=verbose)
        case "tv-pdhg":
            return TVPDHGReconstructor(n_iter=100, verbose=verbose)
        case "wavelet-fista":
            return WaveletFISTAReconstructor(n_iter=100, device=device, verbose=verbose)
        case _:
            raise ValueError(f"Unknown algorithm {name!r}")


def choose_distortion(name: str) -> BaseDistortion:
    match name:
        case "Isotropic LP":
            return IsotropicResolutionReduction(radius_fraction=0.1)
        case "Off-center anisotropic Gaussian bias field":
            return OffCenterAnisotropicGaussianKspaceBiasField(
                width_x_fraction=0.2,
                width_y_fraction=0.35,
                center_x_fraction=0.15,
                center_y_fraction=-0.1,
                edge_gain=0.3,
            )
        case "Translation motion":
            return TranslationMotionDistortion(shift_x_pixels=60, shift_y_pixels=10)
        case "Gaussian bias field":
            return GaussianKspaceBiasField(width_fraction=0.35, edge_gain=0.4)
        case "Gaussian noise":
            return GaussianNoiseDistortion(sigma=0.00001)
        case _:
            raise ValueError(f"Unknown distortion {name!r}")


def choose_metric(name: str) -> dinv.metric.Metric:
    match name:
        case "PSNR":
            return dinv.metric.PSNR(max_pixel=None, complex_abs=True)
        case "NMSE":
            return dinv.metric.NMSE(complex_abs=True)
        case "SSIM":
            return dinv.metric.SSIM(max_pixel=None, complex_abs=True)
        case "HaarPSI":
            return dinv.metric.HaarPSI(norm_inputs="min_max", complex_abs=True)
        case "BlurStrength":
            return dinv.metric.BlurStrength(complex_abs=True)
        case "SharpnessIndex":
            return dinv.metric.SharpnessIndex(complex_abs=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=str, help="Local FastMRI directory with raw k-space .h5 files."
    )
    parser.add_argument("--distortion", type=str, default="", choices=DISTORTIONS)
    parser.add_argument(
        "--algorithm",
        type=str,
        default="",
        choices=ALGORITHMS,
        help="Reconstruction algorithm applied to undistorted and distorted k-space.",
    )
    parser.add_argument("--num_samples", type=int, default=1, help="How many samples to process.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for reconstructors that support it.",
    )
    args = parser.parse_args()

    # set up device, dataset, metrics
    device = dinv.utils.get_device()
    dataset = dinv.datasets.FastMRISliceDataset(args.source, slice_index="middle")
    metrics = [choose_metric(m) for m in METRICS]

    for i, batch in enumerate(iter(torch.utils.data.DataLoader(dataset))):
        # exit loop if we have processed the specified number of samples
        if i >= args.num_samples:
            break

        # batch is a tuple of (x, y) or (x, y, params) where x is GT (could be torch.nan),
        # y is kspace, and params is a dict containing mask (if test set)
        y = batch[1]

        for distortion_name in DISTORTIONS if args.distortion == "" else [args.distortion]:
            distortion = choose_distortion(distortion_name)

            # create physics objects for both clean and distorted k-space
            # the distortion is applied to the k-space measurements (not the image)
            # TODO: allow loading multicoil data
            physics_clean = DistortedKspaceMultiCoilMRI(
                distortion=BaseDistortion(), img_size=(1, 2, *y.shape[-2:]), device=device
            )
            physics = DistortedKspaceMultiCoilMRI(
                distortion=distortion, img_size=(1, 2, *y.shape[-2:]), device=device
            )

            y = y.to(device)
            y_distorted = distortion.A(y)

            # generate reference reconstructions (CG) for both clean and distorted k-space
            x_clean = ConjugateGradientReconstructor()(y, physics_clean)
            x_distorted = ConjugateGradientReconstructor()(y, physics)

            # plot and save the k-space magnitude for both clean and distorted k-space
            save_kspace_plot(
                y,
                y_distorted,
                REPORT_DIR / f"DISTORTION_{distortion_name}_sample_{i}.png",
                distortion_name,
            )

            # loop through algorithms to evaluate on the distorted k-space, with and without correction for the distortion
            for algo_name in ALGORITHMS if args.algorithm == "" else [args.algorithm]:
                print(f"Evaluating algo {algo_name}, distortion {distortion_name}, sample {i}...")

                algo = choose_algorithm(
                    algo_name,
                    img_size=y.shape[-2:],
                    device=device,
                    verbose=args.verbose,
                ).to(device)

                # actual reconstruction with the algo being evaluated
                x_uncorrected = algo(y_distorted, physics_clean)
                x_corrected = algo(y_distorted, physics)

                print("done!")

                dinv.utils.plot(
                    {
                        "Undistorted ksp, CG recon": x_clean,
                        "Distorted ksp, CG recon": x_distorted,
                        f"Distorted ksp, {algo_name} recon, uncorrected": x_uncorrected,
                        f"Distorted ksp, {algo_name} recon, corrected": x_corrected,
                    },
                    subtitles=[
                        "",
                        "",
                        "\n".join(
                            f"{m.__class__.__name__} {m(x_uncorrected, x_clean).item():.2f}"
                            for m in metrics
                        ),
                        "\n".join(
                            f"{m.__class__.__name__} {m(x_corrected, x_clean).item():.2f}"
                            for m in metrics
                        ),
                    ],
                    show=False,
                    close=True,
                    suptitle=f"Algo {algo_name}, distortion {distortion_name}, Sample {i}",
                    save_fn=REPORT_DIR / f"ALGO_{algo_name}_{distortion_name}_sample_{i}.png",
                    fontsize=3,
                )

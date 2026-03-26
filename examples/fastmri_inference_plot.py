"""Inference various reconstructors for various distortion operators.

Usage:
    python examples/fastmri_inference_plot.py --source ../ram-experiments/data/fastmri/knee/singlecoil_val
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import torch
import deepinv as dinv

from mri_recon.distortions import *
from mri_recon.reconstruction import *

REPORT_DIR = Path("reports") / "fastmri_inference_plot"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ALGORITHMS = [
    "zero-filled",
    "conjugate-gradient",
    "ram"
]
DISTORTIONS = [
    "Isotropic LP",
]

def choose_algorithm(name):
    match name:
        case "zero-filled":
            return ZeroFilledReconstructor()
        case "conjugate-gradient":
            return ConjugateGradientReconstructor(max_iter=20)
        case "ram":
            return RAMReconstructor(default_sigma=0.05)
        case _:
            raise ValueError(f"Unknown algorithm {name!r}")

def choose_distortion(name):
    match name:
        case "Isotropic LP":
            return IsotropicResolutionReduction(radius_fraction=0.1)
        case _:
            raise ValueError(f"Unknown distortion {name!r}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, help="Local FastMRI directory with raw k-space .h5 files.")
    parser.add_argument("--distortion", type=str, default="", choices=DISTORTIONS)
    parser.add_argument("--algorithm", type=str, default="", choices=ALGORITHMS, help="Reconstruction algorithm applied to undistorted and distorted k-space.")
    parser.add_argument("--num_samples", type=int, default=1, help="How many samples to process.")
    args = parser.parse_args()

    device = dinv.utils.get_device()
    dataset = dinv.datasets.FastMRISliceDataset(args.source, slice_index="middle")
    metric = dinv.metric.PSNR(max_pixel=None)

    for algo_name in ALGORITHMS if args.algorithm == "" else [args.algorithm]:
        for distortion_name in DISTORTIONS if args.distortion == "" else [args.distortion]:
            algo = choose_algorithm(algo_name)
            distortion = choose_distortion(distortion_name)

            for i, (_, y) in enumerate(iter(torch.utils.data.DataLoader(dataset))):
                if i > args.num_samples: continue
                print(f"Evaluating algo {algo_name}, distortion {distortion_name}, sample {i}...")

                # TODO allow loading multicoil data
                physics_clean = DistortedKspaceMultiCoilMRI(distortion=BaseDistortion(), img_size=(1, 2, *y.shape[-2:]), device=device)
                physics = DistortedKspaceMultiCoilMRI(distortion=distortion, img_size=(1, 2, *y.shape[-2:]), device=device)
                y = y.to(device)
                y_distorted = physics.distortion(y)

                x_clean = algo(y, physics_clean)
                x_uncorrected = algo(y_distorted, physics_clean)
                x_corrected = algo(y_distorted, physics)

                dinv.utils.plot(
                    {
                        "Undistorted ksp recon": x_clean,
                        "Distorted ksp, uncorrected recon": x_uncorrected,
                        "Distorted ksp, corrected recon": x_corrected,
                    },
                    subtitles=[
                        "",
                        f"{metric.__class__.__name__} {metric(x_uncorrected, x_clean).item():.1f} dB",
                        f"{metric.__class__.__name__} {metric(x_corrected, x_clean).item():.1f} dB",
                    ],
                    show=False,
                    close=True,
                    suptitle=f"Algo {algo_name}, distortion {distortion_name}, Sample {i}",
                    save_fn=REPORT_DIR / f"{algo_name}_{distortion_name}_sample_{i}.png",
                    fontsize=5,
                )
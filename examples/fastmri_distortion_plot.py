"""Plot FastMRI reconstructions for all distortion operators.

Usage:
    python examples/fastmri_distortion_plot.py --algorithm zero-filled
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from mri_recon.datasets.fastmri import FastMRIDataset
from mri_recon.distortions import (
    AliasingWrapAroundDistortion,
    AnisotropicResolutionChange,
    Apodization,
    BaseDistortion,
    CoordinateScaling,
    DirectionalSharpnessControl,
    EPINHalfGhostDistortion,
    GibbsRingingDistortion,
    HighFrequencyBoost,
    IsotropicResolutionReduction,
    LineByLineMotionGhostDistortion,
    OffResonanceDistortion,
    PhaseEncodeDecimation,
    RegularizedInverseBlur,
    UnsharpMaskKspace,
    VariableDensityBandwidthReduction,
    ZeroFillDistortion,
)
from mri_recon.reconstruction import (
    BaseReconstructor,
    ConjugateGradientReconstructor,
    FISTAL1Reconstructor,
    LandweberReconstructor,
    POCSReconstructor,
    TVPDHGReconstructor,
    TikhonovReconstructor,
    ZeroFilledReconstructor,
)


DEFAULT_SOURCE = (
    Path(r"C:\Code\mri_recon\data\fastmri")
    / "knee_singlecoil_val"
    / "singlecoil_val"
)

REPORT_DIR = Path("reports") / "fastmri_distortion_plot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Local FastMRI directory with raw k-space .h5 files.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="zero-filled",
        choices=[
            "zero-filled",
            "landweber",
            "conjugate-gradient",
            "tikhonov",
            "pocs",
            "fista-l1",
            "tv-pdhg",
        ],
        help=(
            "Reconstruction algorithm applied to undistorted "
            "and distorted k-space."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="How many samples to process.",
    )
    return parser.parse_args()


def _build_reconstructor(name: str) -> BaseReconstructor:
    if name == "zero-filled":
        return ZeroFilledReconstructor()
    if name == "landweber":
        return LandweberReconstructor(num_iterations=15, step_size=1.0)
    if name == "conjugate-gradient":
        return ConjugateGradientReconstructor(num_iterations=20)
    if name == "tikhonov":
        return TikhonovReconstructor(l2_weight=1e-3)
    if name == "pocs":
        return POCSReconstructor(num_iterations=25, l1_weight=1e-3)
    if name == "fista-l1":
        return FISTAL1Reconstructor(num_iterations=25, l1_weight=1e-3)
    if name == "tv-pdhg":
        return TVPDHGReconstructor(
            num_iterations=60,
            tv_weight=2e-4,
            tau=0.2,
            sigma=0.2,
        )
    raise ValueError(f"Unknown algorithm {name!r}")


def _distortion_suite() -> list[tuple[str, BaseDistortion]]:
    return [
        (
            "Isotropic LP",
            IsotropicResolutionReduction(radius_fraction=0.45),
        ),
        (
            "Anisotropic LP",
            AnisotropicResolutionChange(kx_fraction=0.75, ky_fraction=0.35),
        ),
        (
            "Zero-fill 2x",
            ZeroFillDistortion(pad_factor=2.0),
        ),
        (
            "PE Decimate R3",
            PhaseEncodeDecimation(factor=3),
        ),
        (
            "VD Bandwidth",
            VariableDensityBandwidthReduction(kappa=0.45),
        ),
        (
            "Coord Scale",
            CoordinateScaling(alpha_x=1.12, alpha_y=0.88),
        ),
        (
            "Apodization",
            Apodization(window="gaussian", kappa_x=0.45, kappa_y=0.45),
        ),
        (
            "Directional",
            DirectionalSharpnessControl(kappa_x=0.85, kappa_y=0.3),
        ),
        (
            "HF Boost",
            HighFrequencyBoost(beta=0.45, power=2.0),
        ),
        (
            "Unsharp",
            UnsharpMaskKspace(beta=0.4, lowpass_kappa=0.5),
        ),
        (
            "Reg Inverse",
            RegularizedInverseBlur(l2_weight=1e-3, lowpass_kappa=0.6),
        ),
        (
            "Gibbs Rect",
            GibbsRingingDistortion(kx_fraction=0.5, ky_fraction=0.5),
        ),
        (
            "Aliasing R3",
            AliasingWrapAroundDistortion(factor=3),
        ),
        (
            "EPI N/2",
            EPINHalfGhostDistortion(
                phase_offset_rad=np.deg2rad(25.0),
                delta_x_pixels=0.7,
            ),
        ),
        (
            "Motion Ghost",
            LineByLineMotionGhostDistortion(
                max_shift_x_pixels=2.8,
                max_shift_y_pixels=1.4,
                pattern="step",
            ),
        ),
        (
            "Off-resonance",
            OffResonanceDistortion(
                omega_max=2.5,
                omega_pattern="sinusoidal",
                readout_time_scale=1.6,
            ),
        ),
    ]


def _to_2d_image(image: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image)
    while image_array.ndim > 2:
        image_array = image_array[0]
    return image_array


def _kspace_log_magnitude(kspace: np.ndarray) -> np.ndarray:
    kspace_array = np.asarray(kspace)
    if kspace_array.ndim > 2:
        kspace_array = np.sqrt(np.sum(np.abs(kspace_array) ** 2, axis=0))
    return np.log1p(np.abs(kspace_array))


def _display_limits(image: np.ndarray) -> tuple[float, float]:
    finite_values = image[np.isfinite(image)]
    if finite_values.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite_values, 1.0))
    vmax = float(np.percentile(finite_values, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _distorted_sample(
    sample: dict[str, Any],
    kspace: np.ndarray,
) -> dict[str, Any]:
    distorted = dict(sample)
    distorted["kspace"] = kspace
    # Recompute support from distorted k-space when mask shapes differ.
    distorted["mask"] = None
    return distorted


def main() -> None:
    args = parse_args()
    reconstructor = _build_reconstructor(args.algorithm)
    distortions = _distortion_suite()

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: "
            "python -m pip install matplotlib"
        ) from error

    dataset = FastMRIDataset(
        root_dir=Path(".cache") / "mri_recon_distortion_example",
        split="test",
        challenge="singlecoil",
    )
    dataset.download(source=args.source)

    sample_ids = dataset.sample_ids()[: max(args.num_samples, 1)]
    if not sample_ids:
        raise RuntimeError(
            f"No FastMRI sample volumes found in {dataset.data_dir}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for sample_id in sample_ids:
        slice_indices = sorted(
            raw_sample.slice_ind
            for raw_sample in dataset.slice_dataset.raw_samples
            if raw_sample.fname.stem == sample_id
        )
        if not slice_indices:
            continue

        center_slice_index = slice_indices[len(slice_indices) // 2]
        sample = dataset.read_sample(sample_id, slice_index=center_slice_index)

        original_kspace = np.asarray(sample["kspace"])
        original_kspace_image = _kspace_log_magnitude(original_kspace)
        original_recon = _to_2d_image(
            reconstructor.apply_reconstruction(sample)
        )

        panels: list[tuple[np.ndarray, str, str]] = [
            (original_kspace_image, "K-space", "inferno"),
            (original_recon, f"Recon ({args.algorithm})", "gray"),
        ]

        for distortion_name, distortion in distortions:
            distorted_kspace = np.asarray(distortion.apply(original_kspace))
            distorted_recon = _to_2d_image(
                reconstructor.apply_reconstruction(
                    _distorted_sample(sample, distorted_kspace)
                )
            )
            panels.append(
                (
                    _kspace_log_magnitude(distorted_kspace),
                    f"{distortion_name} k-space",
                    "inferno",
                )
            )
            panels.append(
                (
                    distorted_recon,
                    f"{distortion_name} recon",
                    "gray",
                )
            )

        n_cols = 4
        n_rows = math.ceil(len(panels) / n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 4.5, n_rows * 3.5),
        )
        axes_flat = np.asarray(axes).reshape(-1)

        for axis, (image, title, cmap) in zip(axes_flat, panels):
            vmin, vmax = _display_limits(np.asarray(image))
            axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title)
            axis.axis("off")

        for axis in axes_flat[len(panels):]:
            axis.axis("off")

        fig.suptitle(
            f"{sample_id} - slice {center_slice_index} - {args.algorithm}"
        )
        fig.tight_layout()
        output_path = (
            REPORT_DIR
            / f"{sample_id}_slice_{center_slice_index}_{args.algorithm}.png"
        )
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {output_path.resolve()}")


if __name__ == "__main__":
    main()

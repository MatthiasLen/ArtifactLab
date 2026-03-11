"""Reconstruct three FastMRI images from local k-space data and plot results.

Usage:
    python examples/fastmri_reconstruction_plot.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mri_recon.datasets.fastmri import FastMRIDataset
from mri_recon.reconstruction import (
    ConjugateGradientReconstructor,
    DeepInverseRAMReconstructor,
    FISTAL1Reconstructor,
    LandweberReconstructor,
    POCSReconstructor,
    TVPDHGReconstructor,
    TikhonovReconstructor,
    ZeroFilledReconstructor,
)


DEFAULT_SOURCE = (
    Path(r"C:\Code\mri_recon\data\fastmri")
    / "knee_singlecoil_test"
    / "singlecoil_test"
)
REPORT_DIR = Path("reports") / "fastmri_reconstruction_plot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Local FastMRI directory with raw k-space .h5 files.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="How many leading volumes to reconstruct and display.",
    )
    return parser.parse_args()


def kspace_log_magnitude(kspace: np.ndarray) -> np.ndarray:
    """Return a 2D display image for k-space using log magnitude scaling."""

    kspace_array = np.asarray(kspace)
    if kspace_array.ndim > 2:
        kspace_array = np.sqrt(np.sum(np.abs(kspace_array) ** 2, axis=0))
    return np.log1p(np.abs(kspace_array))


def _kspace_display_limits(kspace_image: np.ndarray) -> tuple[float, float]:
    finite_values = kspace_image[np.isfinite(kspace_image)]
    if finite_values.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite_values, 2.0))
    vmax = float(np.percentile(finite_values, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _image_display_limits(image: np.ndarray) -> tuple[float, float]:
    finite_values = image[np.isfinite(image)]
    if finite_values.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite_values, 1.0))
    vmax = float(np.percentile(finite_values, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _to_2d_image(image: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image)
    while image_array.ndim > 2:
        image_array = image_array[0]
    return image_array


def main() -> None:
    args = parse_args()
    np.random.seed(42)

    # Load FastMRI data directly from local source into the dataset connector.
    dataset = FastMRIDataset(
        root_dir=Path(".cache") / "mri_recon_example",
        split="test",
        challenge="singlecoil",
    )
    dataset.download(source=args.source)
    all_sample_ids = dataset.sample_ids()
    sample_count = min(args.num_samples, len(all_sample_ids))
    # Deterministic sample order for reproducible comparisons.
    sample_ids = all_sample_ids[:sample_count]
    if not sample_ids:
        raise RuntimeError(
            f"No FastMRI sample volumes found in {dataset.data_dir}"
        )

    # Instantiate all reconstruction baselines once and reuse for each sample.
    zero_filled = ZeroFilledReconstructor()
    landweber = LandweberReconstructor(num_iterations=15, step_size=1.0)
    conjugate_gradient = ConjugateGradientReconstructor(num_iterations=20)
    tikhonov = TikhonovReconstructor(l2_weight=1e-3)
    pocs = POCSReconstructor(num_iterations=25, l1_weight=1e-3)
    fista_l1 = FISTAL1Reconstructor(num_iterations=25, l1_weight=1e-3)
    # Fixed TV-PDHG settings tuned to reduce oversmoothing on single-coil knee data.
    tv_pdhg = TVPDHGReconstructor(
        num_iterations=60,
        tv_weight=2e-4,
        tau=0.2,
        sigma=0.2,
    )
    deepinverse = DeepInverseRAMReconstructor()

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: "
            "python -m pip install matplotlib"
        ) from error

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each randomly selected volume at its center slice.
    for sample_id in sample_ids:
        sample_slice_indices = sorted(
            raw_sample.slice_ind
            for raw_sample in dataset.slice_dataset.raw_samples
            if raw_sample.fname.stem == sample_id
        )
        if not sample_slice_indices:
            continue

        center_slice_index = sample_slice_indices[
            len(sample_slice_indices) // 2
        ]
        sample = dataset.read_sample(sample_id, slice_index=center_slice_index)

        # Build visualizations for k-space and all reconstruction outputs.
        kspace_image = kspace_log_magnitude(sample["kspace"])
        kspace_vmin, kspace_vmax = _kspace_display_limits(kspace_image)

        zero_filled_image = _to_2d_image(
            zero_filled.apply_reconstruction(sample)
        )
        zero_vmin, zero_vmax = _image_display_limits(zero_filled_image)
        landweber_image = _to_2d_image(
            landweber.apply_reconstruction(sample)
        )
        landweber_vmin, landweber_vmax = _image_display_limits(
            landweber_image
        )
        conjugate_gradient_image = _to_2d_image(
            conjugate_gradient.apply_reconstruction(sample)
        )
        cg_vmin, cg_vmax = _image_display_limits(conjugate_gradient_image)
        tikhonov_image = _to_2d_image(
            tikhonov.apply_reconstruction(sample)
        )
        tikhonov_vmin, tikhonov_vmax = _image_display_limits(tikhonov_image)
        pocs_image = _to_2d_image(pocs.apply_reconstruction(sample))
        pocs_vmin, pocs_vmax = _image_display_limits(pocs_image)
        fista_l1_image = _to_2d_image(fista_l1.apply_reconstruction(sample))
        fista_vmin, fista_vmax = _image_display_limits(fista_l1_image)
        tv_pdhg_image = _to_2d_image(tv_pdhg.apply_reconstruction(sample))
        tv_vmin, tv_vmax = _image_display_limits(tv_pdhg_image)

        # RAM is scale-sensitive, so run it in a normalized k-space domain.
        deepinverse_sample = dict(sample)
        ram_scale = float(
            np.percentile(np.abs(np.asarray(sample["kspace"])), 99.5)
        )
        if ram_scale <= 0.0:
            ram_scale = 1.0
        deepinverse_sample["kspace"] = np.asarray(sample["kspace"]) / ram_scale

        deepinverse_image: np.ndarray | None = None
        deepinverse_vmin, deepinverse_vmax = 0.0, 1.0
        try:
            physics = DeepInverseRAMReconstructor.build_mri_physics(
                deepinverse_sample
            )
            deepinverse_raw = deepinverse.apply_reconstruction(
                deepinverse_sample,
                physics=physics,
            )
            deepinverse_image_local = _to_2d_image(
                deepinverse.to_magnitude_image(deepinverse_raw)
            )
            deepinverse_image = np.abs(deepinverse_image_local) * ram_scale
            deepinverse_vmin, deepinverse_vmax = _image_display_limits(
                deepinverse_image
            )
        except (ImportError, KeyError, TypeError, AttributeError, RuntimeError, ValueError):
            deepinverse_image = None

        # Save one wide comparison figure per sample to the reports folder.
        fig, axes = plt.subplots(1, 9, figsize=(38, 5))

        axes[0].imshow(
            kspace_image,
            cmap="inferno",
            vmin=kspace_vmin,
            vmax=kspace_vmax,
        )
        axes[0].set_title("K-space log magnitude")
        axes[0].axis("off")

        axes[1].imshow(
            zero_filled_image,
            cmap="gray",
            vmin=zero_vmin,
            vmax=zero_vmax,
        )
        axes[1].set_title("Zero-filled")
        axes[1].axis("off")

        axes[2].imshow(
            landweber_image,
            cmap="gray",
            vmin=landweber_vmin,
            vmax=landweber_vmax,
        )
        axes[2].set_title("Landweber")
        axes[2].axis("off")

        axes[3].imshow(
            conjugate_gradient_image,
            cmap="gray",
            vmin=cg_vmin,
            vmax=cg_vmax,
        )
        axes[3].set_title("Conjugate Gradient")
        axes[3].axis("off")

        axes[4].imshow(
            tikhonov_image,
            cmap="gray",
            vmin=tikhonov_vmin,
            vmax=tikhonov_vmax,
        )
        axes[4].set_title("Tikhonov")
        axes[4].axis("off")

        axes[5].imshow(
            pocs_image,
            cmap="gray",
            vmin=pocs_vmin,
            vmax=pocs_vmax,
        )
        axes[5].set_title("POCS")
        axes[5].axis("off")

        axes[6].imshow(
            fista_l1_image,
            cmap="gray",
            vmin=fista_vmin,
            vmax=fista_vmax,
        )
        axes[6].set_title("FISTA-L1")
        axes[6].axis("off")

        axes[7].imshow(
            tv_pdhg_image,
            cmap="gray",
            vmin=tv_vmin,
            vmax=tv_vmax,
        )
        axes[7].set_title("TV-PDHG")
        axes[7].axis("off")

        if deepinverse_image is None:
            axes[8].text(
                0.5,
                0.5,
                "DeepInverse unavailable",
                ha="center",
                va="center",
                fontsize=10,
            )
            axes[8].set_facecolor("black")
        else:
            axes[8].imshow(
                deepinverse_image,
                cmap="gray",
                vmin=deepinverse_vmin,
                vmax=deepinverse_vmax,
            )
        axes[8].set_title("DeepInverse RAM")
        axes[8].axis("off")

        fig.suptitle(
            f"{sample_id} - center slice {center_slice_index}"
        )
        fig.tight_layout()
        output_path = (
            REPORT_DIR / f"{sample_id}_slice_{center_slice_index}.png"
        )
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved reconstruction reports to {REPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()

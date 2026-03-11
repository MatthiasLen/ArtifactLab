"""Download a FastMRI sample, reconstruct from k-space, and plot the result.

Usage:
    python examples/fastmri_reconstruction_plot.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mri_recon.datasets.fastmri import FastMRIDataset
from mri_recon.reconstruction import ZeroFilledReconstructor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(".cache") / "mri_recon_example",
        help="Directory where the FastMRI sample is stored.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Dataset split folder name.",
    )
    parser.add_argument(
        "--sample-id",
        default="fastmri_sample_singlecoil",
        help="Sample identifier without .h5 extension.",
    )
    parser.add_argument(
        "--slice-index",
        type=int,
        default=0,
        help="Slice index to read from the selected volume.",
    )
    return parser.parse_args()


def kspace_log_magnitude(kspace: np.ndarray) -> np.ndarray:
    """Return a 2D display image for k-space using log magnitude scaling."""

    kspace_array = np.asarray(kspace)
    if kspace_array.ndim > 2:
        kspace_array = np.sqrt(np.sum(np.abs(kspace_array) ** 2, axis=0))
    return np.log1p(np.abs(kspace_array))


def main() -> None:
    args = parse_args()

    dataset = FastMRIDataset(
        root_dir=args.root_dir,
        split=args.split,
        challenge="singlecoil",
    )
    dataset.download()
    sample = dataset.read_sample(args.sample_id, slice_index=args.slice_index)

    target_shape = np.asarray(sample["target"]).shape
    if target_shape[-2:] == (2, 2):
        raise RuntimeError(
            "The default packaged fixture is a tiny 2x2 test sample, not a "
            "representative MRI case. Set "
            "MRI_RECON_FASTMRI_SAMPLE_URL to a real "
            "FastMRI .h5 source and rerun. The script already uses "
            "FastMRIDataset.download()."
        )

    reconstructor = ZeroFilledReconstructor()
    reconstructed = reconstructor.apply_reconstruction(sample)

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: "
            "python -m pip install matplotlib"
        ) from error

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].imshow(kspace_log_magnitude(sample["kspace"]), cmap="gray")
    axes[0].set_title("Input k-space (log magnitude)")
    axes[0].axis("off")

    axes[1].imshow(np.asarray(reconstructed), cmap="gray")
    axes[1].set_title("Zero-filled reconstruction")
    axes[1].axis("off")

    fig.suptitle(f"Sample: {args.sample_id}, slice {args.slice_index}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

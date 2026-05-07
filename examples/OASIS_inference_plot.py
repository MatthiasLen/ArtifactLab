"""Inference OASIS reconstructors for k-space distortion operators.

Usage:
    python examples/OASIS_inference_plot.py --source /path/to/oasis_cross_sectional_data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch
import deepinv as dinv
from torch.utils.data import DataLoader, Dataset

try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError(
        "The OASIS example requires nibabel. Install the project dependencies "
        "or add nibabel to your environment before running this script."
    ) from exc

from mri_recon.distortions import (
    AnisotropicResolutionReduction,
    BaseDistortion,
    CartesianUndersampling,
    DistortedKspaceMultiCoilMRI,
    GaussianKspaceBiasField,
    GaussianNoiseDistortion,
    HannTaperResolutionReduction,
    IsotropicResolutionReduction,
    KaiserTaperResolutionReduction,
    OffCenterAnisotropicGaussianKspaceBiasField,
    PhaseEncodeGhostingDistortion,
    RadialHighPassEmphasisDistortion,
    RotationalMotionDistortion,
    SegmentedTranslationMotionDistortion,
    TranslationMotionDistortion,
)
from mri_recon.reconstruction import (
    ConjugateGradientReconstructor,
    DeepImagePriorReconstructor,
    OASISSinglecoilUnetReconstructor,
    RAMReconstructor,
    TVFISTAReconstructor,
    TVPDHGReconstructor,
    TVPGDReconstructor,
    WaveletFISTAReconstructor,
    ZeroFilledReconstructor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path("reports") / "oasis_inference_plot"
DEFAULT_SPLIT_CSV = REPO_ROOT / "reconstruction_only" / "splits" / "oasis_balanced_test.csv"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "reconstruction_only" / "checkpoints" / "manifest.json"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

ALGORITHMS = [
    # "zero-filled",
    # "conjugate-gradient",
    # "ram",
    # "dip",
    "tv-pgd",
    # "wavelet-fista",
    # "tv-fista",
    # "tv-pdhg",
    "oasis-unet",
    "unet",
]
DISTORTIONS = [
    "None",
    "Cartesian undersampling (variable density)",
    "Cartesian undersampling (uniform random)",
    "Cartesian undersampling (uniform random, zero ACS)",
    "Cartesian undersampling (equispaced)",
    "Cartesian undersampling (equispaced, zero ACS)",
    "Phase-encode ghosting",
    "Segmented translation motion",
    "Translation motion",
    "Rotational motion",
    "Off-center anisotropic Gaussian bias field",
    "Gaussian bias field",
    "Anisotropic LP",
    "Hann taper LP",
    "Kaiser taper LP",
    "Radial high-pass emphasis",
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


class OasisSliceDataset(Dataset):
    """Load 2D OASIS slices from Analyze/NIfTI volumes.

    Parameters
    ----------
    split_csv : Path
        CSV file listing OASIS subjects and slice counts.
    data_path : Path
        Root directory containing OASIS subject folders.
    sample_rate : float, optional
        Fraction of slices to include from each volume.
    cache_size : int, optional
        Number of loaded volumes to keep in memory.
    """

    def __init__(
        self,
        split_csv: Path,
        data_path: Path,
        sample_rate: float = 1.0,
        cache_size: int = 2,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.data_path = Path(data_path)
        if not 0 < sample_rate <= 1.0:
            raise ValueError("sample_rate must be in the range (0, 1].")
        self.sample_rate = sample_rate
        self.cache_size = max(0, cache_size)
        self._volume_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.raw_samples = self._create_sample_list()

    def __len__(self) -> int:
        """Return the number of available slices."""

        return len(self.raw_samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one complex-valued OASIS slice in repo tensor convention."""

        subject_id, slice_num = self.raw_samples[index]
        target_np = self._read_raw_slice(subject_id, slice_num)
        real = torch.from_numpy(target_np)
        x = torch.stack([real, torch.zeros_like(real)], dim=0)
        return {"x": x.float(), "subject_id": subject_id, "slice_num": slice_num}

    def _create_sample_list(self) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        with self.split_csv.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = [item.strip() for item in line.split(",")]
                if not row or not row[0]:
                    continue
                try:
                    total_slices = int(row[-1])
                except ValueError:
                    continue

                subject_id = row[0]
                if self.sample_rate >= 1.0:
                    start = 0
                    stop = total_slices
                else:
                    mid = round(total_slices / 2)
                    half_span = round(total_slices * self.sample_rate / 2)
                    start = max(0, mid - half_span)
                    stop = min(total_slices, mid + half_span)

                for slice_num in range(start, stop):
                    samples.append((subject_id, slice_num))
        return samples

    def _read_raw_slice(self, subject_id: str, slice_num: int) -> np.ndarray:
        volume = self._get_volume(subject_id)
        return np.ascontiguousarray(volume[slice_num], dtype=np.float32)

    def _get_volume(self, subject_id: str) -> np.ndarray:
        if self.cache_size > 0 and subject_id in self._volume_cache:
            self._volume_cache.move_to_end(subject_id)
            return self._volume_cache[subject_id]

        image_glob = self.data_path / subject_id / "PROCESSED" / "MPRAGE" / "T88_111"
        matches = sorted(image_glob.glob("*t88_gfc.img"))
        if not matches:
            raise FileNotFoundError(
                f"Could not find OASIS image for subject {subject_id!r} under {image_glob}."
            )

        image_data = nib.load(str(matches[0])).get_fdata(dtype=np.float32)
        volume = np.ascontiguousarray(
            np.transpose(np.squeeze(image_data), (1, 0, 2)),
            dtype=np.float32,
        )

        if self.cache_size > 0:
            self._volume_cache[subject_id] = volume
            if len(self._volume_cache) > self.cache_size:
                self._volume_cache.popitem(last=False)

        return volume


def _kspace_to_log_magnitude(kspace: torch.Tensor) -> torch.Tensor:
    """Convert k-space tensor to log-magnitude image for visualization."""

    kspace = kspace.detach().cpu()
    if kspace.ndim == 5:
        kspace = kspace[0]
    if kspace.ndim == 4:
        if kspace.shape[0] == 1 and kspace.shape[1] == 2:
            kspace = kspace[0]
        elif kspace.shape[0] != 2:
            raise ValueError(
                "Expected k-space with shape (2, H, W), (1, 2, H, W), "
                f"or (1, 2, N, H, W), got {tuple(kspace.shape)}"
            )
    if kspace.ndim != 3 and kspace.ndim != 4:
        raise ValueError(
            "Expected k-space with shape (2, H, W), (1, 2, H, W), "
            f"or (1, 2, N, H, W), got {tuple(kspace.shape)}"
        )
    if kspace.shape[0] != 2:
        raise ValueError(f"Expected real/imaginary channel first, got {tuple(kspace.shape)}")

    kspace_complex = torch.view_as_complex(torch.movedim(kspace, 0, -1).contiguous())
    magnitude = torch.abs(kspace_complex)
    if magnitude.ndim == 3:
        magnitude = torch.sqrt(torch.sum(magnitude.square(), dim=0))
    magnitude = torch.log1p(magnitude)

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
    """Save clean and distorted k-space magnitude plots."""

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


def image_to_kspace(x: torch.Tensor) -> torch.Tensor:
    """Convert channel-first complex images to centered k-space."""

    x_complex = torch.view_as_complex(x.movedim(1, -1).contiguous())
    y_complex = torch.fft.fftshift(
        torch.fft.fft2(x_complex, dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )
    return torch.view_as_real(y_complex).movedim(-1, 1).contiguous()


def kspace_to_image(y: torch.Tensor) -> torch.Tensor:
    """Convert centered channel-first k-space to complex images."""

    y_complex = torch.view_as_complex(y.movedim(1, -1).contiguous())
    x_complex = torch.fft.ifft2(
        torch.fft.ifftshift(y_complex, dim=(-2, -1)),
        dim=(-2, -1),
        norm="ortho",
    )
    return torch.view_as_real(x_complex).movedim(-1, 1).contiguous()


class OasisCenteredFFTPhysics:
    """Physics adapter matching the OASIS U-Net FFT convention.

    Parameters
    ----------
    distortion : BaseDistortion
        K-space distortion applied after the centered FFT.
    """

    def __init__(self, distortion: BaseDistortion) -> None:
        self.distortion = distortion

    def A(self, x: torch.Tensor) -> torch.Tensor:
        """Apply centered FFT and k-space distortion.

        Parameters
        ----------
        x : torch.Tensor
            Complex image tensor with shape ``(B, 2, H, W)``.

        Returns
        -------
        torch.Tensor
            Distorted centered k-space tensor.
        """

        return self.distortion.A(image_to_kspace(x))

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint distortion and centered inverse FFT.

        Parameters
        ----------
        y : torch.Tensor
            Distorted centered k-space tensor.

        Returns
        -------
        torch.Tensor
            Complex image tensor with shape ``(B, 2, H, W)``.
        """

        return kspace_to_image(self.distortion.A_adjoint(y))


def resolve_oasis_checkpoint(
    checkpoint: Optional[Path],
    acceleration: int,
    manifest_path: Path,
) -> Path:
    """Resolve an explicit or packaged OASIS checkpoint path.

    Parameters
    ----------
    checkpoint : Path or None
        User-provided checkpoint path.
    acceleration : int
        Acceleration key used when loading from the manifest.
    manifest_path : Path
        JSON manifest with packaged checkpoint metadata.

    Returns
    -------
    Path
        Resolved checkpoint path.
    """

    if checkpoint is not None:
        return checkpoint.expanduser().resolve()

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    checkpoints = manifest.get("checkpoints", {})
    key = str(acceleration)
    if key not in checkpoints:
        available = ", ".join(sorted(checkpoints))
        raise ValueError(
            f"No packaged checkpoint for acceleration {acceleration}. Available: {available}."
        )

    filename = Path(checkpoints[key]["filename"])
    if filename.is_absolute():
        return filename
    return (manifest_path.parent.parent / filename).resolve()


def choose_algorithm(
    name: str,
    checkpoint_file: Path,
    img_size: tuple = (640, 368),
    device: torch.device = "cpu",
    verbose: bool = False,
) -> dinv.models.Reconstructor:
    """Construct a reconstructor by selector name."""

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
        case "oasis-unet" | "unet":
            return OASISSinglecoilUnetReconstructor(
                checkpoint_file=str(checkpoint_file),
                device=device,
            )
        case _:
            raise ValueError(f"Unknown algorithm {name!r}")


def choose_distortion(
    name: str,
    acceleration: int,
    center_fraction: float,
) -> BaseDistortion:
    """Construct a k-space distortion by display name."""

    match name:
        case "None":
            return BaseDistortion()
        case "Cartesian undersampling (variable density)":
            return CartesianUndersampling(
                keep_fraction=1.0 / acceleration,
                center_fraction=center_fraction,
                pattern="variable_density_random",
                axis=-1,
                seed=42,
            )
        case "Cartesian undersampling (uniform random)":
            return CartesianUndersampling(
                keep_fraction=1.0 / acceleration,
                center_fraction=center_fraction,
                pattern="uniform_random",
                axis=-1,
                seed=42,
            )
        case "Cartesian undersampling (uniform random, zero ACS)":
            return CartesianUndersampling(
                keep_fraction=1.0 / acceleration,
                center_fraction=0.0,
                pattern="uniform_random",
                axis=-1,
                seed=42,
            )
        case "Cartesian undersampling (equispaced)":
            return CartesianUndersampling(
                keep_fraction=1.0 / acceleration,
                center_fraction=center_fraction,
                pattern="equispaced",
                axis=-1,
                seed=42,
            )
        case "Cartesian undersampling (equispaced, zero ACS)":
            return CartesianUndersampling(
                keep_fraction=1.0 / acceleration,
                center_fraction=0.0,
                pattern="equispaced",
                axis=-1,
                seed=42,
            )
        case "Phase-encode ghosting":
            return PhaseEncodeGhostingDistortion(
                line_period=2,
                line_offset=1,
                phase_error_radians=torch.pi / 2,
                corrupted_line_scale=1.0,
            )
        case "Anisotropic LP":
            return AnisotropicResolutionReduction(
                kx_radius_fraction=1.0,
                ky_radius_fraction=0.25,
            )
        case "Hann taper LP":
            return HannTaperResolutionReduction(
                radius_fraction=0.35,
                transition_fraction=0.4,
            )
        case "Kaiser taper LP":
            return KaiserTaperResolutionReduction(
                radius_fraction=0.35,
                transition_fraction=0.4,
                beta=8.6,
            )
        case "Radial high-pass emphasis":
            return RadialHighPassEmphasisDistortion(alpha=0.4)
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
        case "Rotational motion":
            return RotationalMotionDistortion(angle_radians=torch.pi / 6)
        case "Segmented translation motion":
            return SegmentedTranslationMotionDistortion(
                shift_x_pixels=(0.0, 20.0, 50.0, -50.0),
                shift_y_pixels=(0.0, 10.0, -20.0, 20.0),
            )
        case "Gaussian bias field":
            return GaussianKspaceBiasField(width_fraction=0.35, edge_gain=0.4)
        case "Gaussian noise":
            return GaussianNoiseDistortion(sigma=0.00001)
        case _:
            raise ValueError(f"Unknown distortion {name!r}")


def choose_metric(name: str) -> dinv.metric.Metric:
    """Construct a DeepInverse metric by selector name."""

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
        case _:
            raise ValueError(f"Unknown metric {name!r}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="OASIS root directory containing subject folders.",
    )
    parser.add_argument(
        "--split_csv",
        type=Path,
        default=DEFAULT_SPLIT_CSV,
        help="CSV listing OASIS subjects and slice counts.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Checkpoint manifest JSON.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit OASIS U-Net checkpoint. Overrides --acceleration.",
    )
    parser.add_argument(
        "--acceleration",
        type=int,
        default=4,
        help="Packaged OASIS checkpoint acceleration factor.",
    )
    parser.add_argument(
        "--center_fraction",
        type=float,
        default=0.08,
        help="Center fraction used by the Cartesian undersampling distortion.",
    )
    parser.add_argument(
        "--distortion",
        type=str,
        default="Cartesian undersampling (uniform random)",
        choices=DISTORTIONS,
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="unet",
        choices=ALGORITHMS,
        help="Reconstruction algorithm applied to distorted OASIS k-space.",
    )
    parser.add_argument("--num_samples", type=int, default=1, help="How many slices to process.")
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.6,
        help="Fraction of slices per volume to include from the split CSV.",
    )
    parser.add_argument("--volume_cache_size", type=int, default=2)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for reconstructors that support it.",
    )
    return parser


def main() -> None:
    """Run OASIS inference plots."""

    args = build_parser().parse_args()
    args.source = args.source.expanduser().resolve()
    args.split_csv = args.split_csv.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    checkpoint_file = resolve_oasis_checkpoint(args.checkpoint, args.acceleration, args.manifest)

    device = dinv.utils.get_device()
    dataset = OasisSliceDataset(
        split_csv=args.split_csv,
        data_path=args.source,
        sample_rate=args.sample_rate,
        cache_size=args.volume_cache_size,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    metrics = [choose_metric(m) for m in METRICS]

    for i, batch in enumerate(iter(dataloader)):
        if i >= args.num_samples:
            break

        x = batch["x"].to(device)
        subject_id = batch["subject_id"][0]
        slice_num = int(batch["slice_num"][0])

        for distortion_name in [args.distortion]:
            distortion = choose_distortion(
                distortion_name,
                acceleration=args.acceleration,
                center_fraction=args.center_fraction,
            )

            physics_clean = DistortedKspaceMultiCoilMRI(
                distortion=BaseDistortion(),
                img_size=(1, 2, *x.shape[-2:]),
                device=device,
            )
            physics = DistortedKspaceMultiCoilMRI(
                distortion=distortion,
                img_size=(1, 2, *x.shape[-2:]),
                device=device,
            )
            oasis_physics_clean = OasisCenteredFFTPhysics(BaseDistortion())
            oasis_physics = OasisCenteredFFTPhysics(distortion)

            y = image_to_kspace(x)
            y_distorted = distortion.A(y)
            y_physics_distorted = physics(x)
            x_distorted = kspace_to_image(y_distorted)

            save_kspace_plot(
                y,
                y_distorted,
                REPORT_DIR / f"DISTORTION_{distortion_name}_sample_{i}.png",
                distortion_name,
            )

            for algo_name in ALGORITHMS if args.algorithm == "" else [args.algorithm]:
                print(
                    f"Evaluating algo {algo_name}, distortion {distortion_name}, "
                    f"subject {subject_id}, slice {slice_num}..."
                )

                algo = choose_algorithm(
                    algo_name,
                    checkpoint_file=checkpoint_file,
                    img_size=x.shape[-2:],
                    device=device,
                    verbose=args.verbose,
                ).to(device)

                if algo_name in {"oasis-unet", "unet"}:
                    y_eval = y_distorted
                    eval_physics_clean = oasis_physics_clean
                    eval_physics = oasis_physics
                else:
                    y_eval = y_physics_distorted
                    eval_physics_clean = physics_clean
                    eval_physics = physics

                x_uncorrected = algo(y_eval, eval_physics_clean)
                x_corrected = algo(y_eval, eval_physics)
                uncorrected_scores = [
                    f"{m.__class__.__name__} {m(x_uncorrected, x).item():.2f}" for m in metrics
                ]
                corrected_scores = [
                    f"{m.__class__.__name__} {m(x_corrected, x).item():.2f}" for m in metrics
                ]
                print(f"  uncorrected: {', '.join(uncorrected_scores)}")
                print(f"  corrected: {', '.join(corrected_scores)}")

                dinv.utils.plot(
                    {
                        "Ground truth OASIS slice": x,
                        "Distorted ksp, zero-filled": x_distorted,
                        f"Distorted ksp, {algo_name} recon, uncorrected": x_uncorrected,
                        f"Distorted ksp, {algo_name} recon, corrected": x_corrected,
                    },
                    subtitles=[
                        "",
                        "",
                        "\n".join(uncorrected_scores),
                        "\n".join(corrected_scores),
                    ],
                    show=False,
                    close=True,
                    suptitle=(
                        f"Algo {algo_name}, distortion {distortion_name}, "
                        f"subject {subject_id}, slice {slice_num}"
                    ),
                    save_fn=REPORT_DIR / f"ALGO_{algo_name}_{distortion_name}_sample_{i}.png",
                    fontsize=3,
                )

                print("done!")


if __name__ == "__main__":
    main()

"""Inference various reconstructors for various distortion operators.

Usage:
    python examples/fastmri_inference_plot.py --source ../ram-experiments/data/fastmri/knee/singlecoil_val
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import deepinv as dinv
import torch

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
    PartialFourierDistortion,
    PhaseEncodeGhostingDistortion,
    RadialHighPassEmphasisDistortion,
    RotationalMotionDistortion,
    SegmentedRotationalMotionDistortion,
    SegmentedTranslationMotionDistortion,
    TranslationMotionDistortion,
)
from mri_recon.reconstruction import (
    ConjugateGradientReconstructor,
    EXPLICIT_UNET_ALGORITHMS,
    OASISSinglecoilUnetReconstructor,
    choose_reconstructor,
    uses_oasis_centered_path,
    validate_algorithm_dataset_compatibility,
)
from mri_recon.utils import (
    OasisCenteredFFTPhysics,
    OasisSliceDataset,
    fastmri_measurement_to_image,
    fastmri_measurement_to_oasis_kspace,
    image_to_kspace,
    kspace_to_image,
    save_kspace_plot,
)

FASTMRI_REPORT_DIR = Path("reports") / "fastmri_inference_plot"
OASIS_REPORT_DIR = Path("reports") / "oasis_inference_plot"
FASTMRI_REPORT_DIR.mkdir(parents=True, exist_ok=True)
OASIS_REPORT_DIR.mkdir(parents=True, exist_ok=True)
ALGORITHMS = [
    "zero-filled",
    # "conjugate-gradient",
    # "ram",
    # "dip",
    "tv-pgd",
    # "wavelet-fista",
    "tv-fista",
    # "tv-pdhg",
    *list(EXPLICIT_UNET_ALGORITHMS),
]

DISTORTIONS = [
    "Cartesian undersampling (variable density)",
    "Cartesian undersampling (uniform random)",
    "Cartesian undersampling (uniform random, zero ACS)",
    "Cartesian undersampling (equispaced)",
    "Cartesian undersampling (equispaced, zero ACS)",
    "Partial Fourier",
    "Phase-encode ghosting",
    "Segmented translation motion",
    "Segmented rotational motion",
    "Translation motion",
    "Rotational motion",
    "Off-center anisotropic Gaussian bias field",
    "Gaussian bias field",
    "Anisotropic LP",
    "Hann taper LP",
    "Kaiser taper LP",
    "Gaussian noise",
    "Isotropic LP",
    "Radial high-pass emphasis",
]
METRICS = [
    "PSNR",
    "NMSE",
    "SSIM",
    "HaarPSI",
    "SharpnessIndex",
    "BlurStrength",
]


def choose_distortion(
    name: str,
    keep_fraction: float = 0.25,
    center_fraction: float = 0.125,
    cartesian_axis: int = -2,
) -> BaseDistortion:
    """Build one distortion operator for the inference comparison script.

    The ``cartesian_axis`` is supplied by the active measurement convention:
    FastMRI-native runs use the repository's existing axis, while OASIS-native
    and FastMRI-to-OASIS runs use the centered OASIS axis.
    """

    match name:
        case "Phase-encode ghosting":
            return PhaseEncodeGhostingDistortion(
                line_period=2,
                line_offset=1,
                phase_error_radians=torch.pi / 2,
                corrupted_line_scale=1.0,
            )
        case "Cartesian undersampling (variable density)":
            return CartesianUndersampling(
                keep_fraction=keep_fraction,
                center_fraction=center_fraction,
                pattern="variable_density_random",
                axis=cartesian_axis,
                seed=42,
            )
        case "Cartesian undersampling (uniform random)":
            return CartesianUndersampling(
                keep_fraction=keep_fraction,
                center_fraction=center_fraction,
                pattern="uniform_random",
                axis=cartesian_axis,
                seed=42,
            )
        case "Cartesian undersampling (uniform random, zero ACS)":
            return CartesianUndersampling(
                keep_fraction=keep_fraction,
                center_fraction=0.0,
                pattern="uniform_random",
                axis=cartesian_axis,
                seed=42,
            )
        case "Cartesian undersampling (equispaced)":
            return CartesianUndersampling(
                keep_fraction=keep_fraction,
                center_fraction=center_fraction,
                pattern="equispaced",
                axis=cartesian_axis,
                seed=42,
            )
        case "Cartesian undersampling (equispaced, zero ACS)":
            return CartesianUndersampling(
                keep_fraction=keep_fraction,
                center_fraction=0.0,
                pattern="equispaced",
                axis=cartesian_axis,
                seed=42,
            )
        case "Partial Fourier":
            return PartialFourierDistortion(
                partial_fraction=0.7,
                center_fraction=center_fraction,
                axis=cartesian_axis,
                side="high",
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
        case "Segmented rotational motion":
            return SegmentedRotationalMotionDistortion(
                angle_radians=(0.0, torch.pi / 20, -torch.pi / 24, torch.pi / 16),
            )
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
    """Build one evaluation metric used in the saved comparison plots."""

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


def prepare_measurement_sample(
    sample_batch: object,
    dataset_name: str,
    use_oasis_fft_path: bool,
    run_device: torch.device | str,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Prepare one input measurement and its clean image reference.

    FastMRI samples are loaded as native measurements. When the OASIS U-Net is
    selected on FastMRI data, the helper converts those measurements into the
    centered OASIS k-space convention while preserving the native adjoint image
    as the clean reference.
    """

    if dataset_name == "oasis":
        reference_image = sample_batch["x"].to(run_device)
        return reference_image, image_to_kspace(reference_image)

    # FastMRI batches are tuples such as (x, y) or (x, y, params).
    y_fastmri = sample_batch[1].to(run_device)
    if use_oasis_fft_path:
        reference_image = fastmri_measurement_to_image(y_fastmri, device=run_device)
        return reference_image, fastmri_measurement_to_oasis_kspace(y_fastmri, device=run_device)

    return None, y_fastmri


def build_physics_pair(
    image_shape: tuple[int, int],
    distortion_operator: BaseDistortion,
    run_device: torch.device | str,
    use_oasis_fft_path: bool,
) -> tuple[object, object]:
    """Build clean and distorted physics operators for the active path."""

    if use_oasis_fft_path:
        return OasisCenteredFFTPhysics(BaseDistortion()), OasisCenteredFFTPhysics(
            distortion_operator
        )

    clean_physics = DistortedKspaceMultiCoilMRI(
        distortion=BaseDistortion(),
        img_size=(1, 2, *image_shape),
        device=run_device,
    )
    distorted_physics = DistortedKspaceMultiCoilMRI(
        distortion=distortion_operator,
        img_size=(1, 2, *image_shape),
        device=run_device,
    )
    return clean_physics, distorted_physics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    # data related arguments
    parser.add_argument(
        "--source",
        type=Path,
        help="Local FastMRI directory with raw k-space .h5 files or OASIS root directory.",
    )
    parser.add_argument("--dataset", choices=("fastmri", "oasis"), default="fastmri")

    parser.add_argument("--distortion", type=str, default="", choices=DISTORTIONS)
    parser.add_argument(
        "--keep_fraction",
        type=float,
        default=0.25,
        help="Fraction of k-space lines to keep for undersampling distortions.",
    )
    parser.add_argument(
        "--center_fraction",
        type=float,
        default=0.125,
        help="Fraction of low-frequency k-space lines to keep fully for undersampling distortions.",
    )

    # algo related arguments
    parser.add_argument(
        "--algorithm",
        type=str,
        default="",
        choices=ALGORITHMS,
        help="Reconstruction algorithm applied to undistorted and distorted k-space.",
    )
    # inference related arguments
    parser.add_argument("--num_samples", type=int, default=1, help="How many samples to process.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for reconstructors that support it.",
    )
    args = parser.parse_args()

    selected_algorithms = ALGORITHMS if args.algorithm == "" else [args.algorithm]
    selected_distortions = DISTORTIONS if args.distortion == "" else [args.distortion]
    for algo_name in selected_algorithms:
        validate_algorithm_dataset_compatibility(args.dataset, algo_name)

    # set up report dir
    REPORT_DIR = OASIS_REPORT_DIR if args.dataset == "oasis" else FASTMRI_REPORT_DIR

    # set up device, dataset, metrics
    device = dinv.utils.get_device()
    if args.dataset == "oasis":
        split_csv = OASISSinglecoilUnetReconstructor.resolve_default_split_csv()
        dataset = OasisSliceDataset(
            data_path=args.source,
            split_csv=split_csv,
            sample_rate=0.6,
        )
    else:
        dataset = dinv.datasets.FastMRISliceDataset(str(args.source), slice_index="middle")
    metrics = [choose_metric(m) for m in METRICS]

    for i, batch in enumerate(iter(torch.utils.data.DataLoader(dataset))):
        # exit loop if we have processed the specified number of samples
        if i >= args.num_samples:
            break

        for algo_name in selected_algorithms:
            use_oasis_path = uses_oasis_centered_path(args.dataset, algo_name)
            x_reference, y = prepare_measurement_sample(
                sample_batch=batch,
                dataset_name=args.dataset,
                use_oasis_fft_path=use_oasis_path,
                run_device=device,
            )
            algo = choose_reconstructor(
                algo_name,
                img_size=y.shape[-2:],
                device=device,
                verbose=args.verbose,
                dataset=args.dataset,
            ).to(device)

            for distortion_name in selected_distortions:
                distortion = choose_distortion(
                    distortion_name,
                    keep_fraction=args.keep_fraction,
                    center_fraction=args.center_fraction,
                    cartesian_axis=-1 if use_oasis_path else -2,
                )

                physics_clean, physics = build_physics_pair(
                    image_shape=y.shape[-2:],
                    distortion_operator=distortion,
                    run_device=device,
                    use_oasis_fft_path=use_oasis_path,
                )
                y_distorted = distortion.A(y)

                # generate reference reconstructions (CG) for both clean and distorted k-space
                # without correction for the distortion, i.e. using physics_clean in both cases
                if use_oasis_path:
                    x_clean = x_reference
                    x_distorted = kspace_to_image(y_distorted)
                else:
                    x_clean = ConjugateGradientReconstructor()(y, physics_clean)
                    x_distorted = ConjugateGradientReconstructor()(y_distorted, physics_clean)

                save_kspace_plot(
                    y,
                    y_distorted,
                    REPORT_DIR / f"DISTORTION_{algo_name}_{distortion_name}_sample_{i}.png",
                    distortion_name,
                )

                print(f"Evaluating algo {algo_name}, distortion {distortion_name}, sample {i}...")

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

"""Inference various reconstructors for various distortion operators.

Usage:
    python examples/fastmri_inference_plot.py --source ../ram-experiments/data/fastmri/knee/singlecoil_val
"""

import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import deepinv as dinv
import torch
from tifffile import imwrite

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
    choose_reconstructor,
    uses_oasis_centered_path,
    validate_algorithm_dataset_compatibility,
    EXPLICIT_UNET_ALGORITHMS,
)
from mri_recon.utils import (
    OasisCenteredFFTPhysics,
    OasisCenterSliceFolderDataset,
    FastMRIProstateDataset,
    fastmri_measurement_to_image,
    fastmri_measurement_to_oasis_kspace,
    oasis_kspace_to_fastmri_measurement,
    image_to_kspace,
    _kspace_to_log_magnitude,
)

EXPERIMENTS_DIR = Path("reports") / "experiments"

ALGORITHMS = [
    # "zero-filled",
    "conjugate-gradient",
    # "ram",
    # "dip",
    # "tv-pgd",
    # "wavelet-fista",
    # "tv-fista",
    # "tv-pdhg",
    *list(EXPLICIT_UNET_ALGORITHMS),
]

DISTORTIONS = [
    "no distortion",
    # "Cartesian undersampling (variable density)",
    # "Cartesian undersampling (uniform random)",
    # "Cartesian undersampling (uniform random, zero ACS)",
    # "Cartesian undersampling (equispaced)",
    "Cartesian undersampling (equispaced, zero ACS)",
    # "Partial Fourier",
    # "Phase-encode ghosting",
    # "Segmented translation motion",
    # "Segmented rotational motion",
    # "Translation motion",
    # "Rotational motion",
    # "Off-center anisotropic Gaussian bias field",
    # "Gaussian bias field",
    # "Anisotropic LP",
    # "Hann taper LP",
    # "Kaiser taper LP",
    # "Gaussian noise",
    # "Isotropic LP",
    # "Radial high-pass emphasis",
]
METRICS = [
    "PSNR",
    # "NMSE",
    # "SSIM",
    # "HaarPSI",
    # "SharpnessIndex",
    # "BlurStrength",
]

DATASETS = {
    # "fastmri": "/home/melanie.dohmen/mri_recon/data/fastmri/singlecoil_val",
    "oasis": "/home/melanie.dohmen/mri_recon/data/oasis",
    # "fastmri_multicoil": "/home/melanie.dohmen/mri_recon/data/fastmri/multicoil_train",
    "cmrxrecon": "/home/melanie.dohmen/mri_recon/data/CMRxRecon/CMRxRecon/",  # SingleCoil/Cine/TrainingSet/FullSample",
    "prostate": "/home/melanie.dohmen/mri_recon/data/fastmri/fastMRI_prostate_T2_IDS_001_020",
}


def convert_image_for_save(im: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor image complex tensor to a real-valued NumPy array suitable
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
        case "no distortion":
            return BaseDistortion()
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


# def prepare_measurement_sample(
#     sample_batch: object,
#     dataset_name: str,
#     use_oasis_fft_path: bool,
#     run_device: torch.device | str,
# ) -> tuple[torch.Tensor | None, torch.Tensor]:
#     """Prepare one input measurement and its clean image reference.

#     FastMRI samples are loaded as native measurements. When the OASIS U-Net is
#     selected on FastMRI data, the helper converts those measurements into the
#     centered OASIS k-space convention while preserving the native adjoint image
#     as the clean reference.
#     """

#     if dataset_name == "oasis":
#         x = sample_batch["x"].to(run_device)
#         y = image_to_kspace(x)
#         coil_maps = None
#     elif dataset_name in ("fastmri",) and use_oasis_fft_path:
#         y = sample_batch[1].to(run_device)
#         x = fastmri_measurement_to_image(y)
#         y = fastmri_measurement_to_oasis_kspace(y, device=run_device)
#         coil_maps = None
#     elif dataset_name == "fastmri_multicoil" and use_oasis_fft_path:
#         y = sample_batch[1].to(run_device)

#         coil_maps = (
#             sample_batch[2]["coil_maps"].to(run_device)
#             if isinstance(sample_batch, (tuple, list))
#             and len(sample_batch) == 3
#             and "coil_maps" in sample_batch[2]
#             else None
#         )
#         x = fastmri_measurement_to_image(y, coil_maps=coil_maps)
#         y = fastmri_measurement_to_oasis_kspace(y, coil_maps=coil_maps, device=run_device)
#     elif dataset_name in ("fastmri", "fastmri_multicoil"):
#         x = None
#         y = sample_batch[1].to(run_device)
#         coil_maps = (
#             sample_batch[2]["coil_maps"].to(run_device)
#             if isinstance(sample_batch, (tuple, list))
#             and len(sample_batch) == 3
#             and "coil_maps" in sample_batch[2]
#             else None
#         )
#     elif dataset_name in ("cmrxrecon"):
#         x = sample_batch[0].to(run_device)
#         y = sample_batch[1].to(run_device)
#         coil_maps = (
#             sample_batch[2]["coil_maps"].to(run_device)
#             if isinstance(sample_batch, (tuple, list))
#             and len(sample_batch) == 3
#             and "coil_maps" in sample_batch[2]
#             else None
#         )
#     elif dataset_name in ("prostate"):
#         x = sample_batch[0].to(run_device)
#         y = sample_batch[1].to(run_device)
#         coil_maps = None

#     print(f"\t[Prepared measurement] k-space shape {y.shape} and reference image shape: {x.shape if x is not None else None}")

#     return x, y, coil_maps


def get_measurement_sample(
    sample_batch: object,
    dataset_name: str,
    run_device: torch.device | str,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Prepare one input measurement and its clean image reference.

    FastMRI samples are loaded as native measurements. When the OASIS U-Net is
    selected on FastMRI data, the helper converts those measurements into the
    centered OASIS k-space convention while preserving the native adjoint image
    as the clean reference.
    """
    coil_maps = None
    if dataset_name == "oasis":
        x = sample_batch["x"].to(run_device)
        print(f"\t[Debug] Reference image shape: {x.shape}, dtype: {x.dtype}")
        y_centered = image_to_kspace(x)
        print(f"\t[Debug] Centered k-space shape: {y_centered.shape}, dtype: {y_centered.dtype}")
        y = oasis_kspace_to_fastmri_measurement(y_centered, device=run_device)
    elif dataset_name in ("fastmri"):
        y = sample_batch[1].to(run_device)
        y_centered = fastmri_measurement_to_oasis_kspace(y, device=run_device)
        x = fastmri_measurement_to_image(y, rss=True)
    elif dataset_name in ("fastmri_multicoil"):
        y = sample_batch[1].to(run_device)
        coil_maps = (
            sample_batch[2]["coil_maps"].to(run_device)
            if isinstance(sample_batch, (tuple, list))
            and len(sample_batch) == 3
            and "coil_maps" in sample_batch[2]
            else None
        )
        x = fastmri_measurement_to_image(y, coil_maps=coil_maps, rss=True)
        y_centered = fastmri_measurement_to_oasis_kspace(y, coil_maps=coil_maps, device=run_device)
    elif dataset_name in ("cmrxrecon"):
        # ignore multi-coil reference image data
        x = sample_batch[0].to(run_device)
        print(f"\t[Debug] Reference image shape: {x.shape}, dtype: {x.dtype}")

        y = sample_batch[1].to(run_device)

        coil_maps = (
            sample_batch[2]["coil_maps"].to(run_device)
            if isinstance(sample_batch, (tuple, list))
            and len(sample_batch) == 3
            and "coil_maps" in sample_batch[2]
            else None
        )
        y_centered = fastmri_measurement_to_oasis_kspace(y, coil_maps=coil_maps, device=run_device)
        # reconstruct coil-combined image reference from multi-coil k-space data using
        # integrated espirit sensitivity map estimation, RSS coil combination
        x = fastmri_measurement_to_image(y, coil_maps=coil_maps, rss=True)

    elif dataset_name in ("prostate"):
        # has shape: (B, num_averages, coils, H, W) with dtype= complex128-> take first average and convert to image space reference
        x = sample_batch[0].to(run_device)
        print(f"\t[Debug] Reference image shape: {x.shape}, type: {x.dtype}")
        # take mean of average images:
        x = x.mean(dim=1)
        print(f"\t[Debug] Mean of averages image shape: {x.shape}, type: {x.dtype}")

        # convert to channel representation of complex numbers
        # (B, H, W) with complex dtype -> (B, H, W, 2) with real dtype
        x = torch.view_as_real(x) if torch.is_complex(x) else x
        print(f"\t[Debug] after view_as_real (if complex) shape: {x.shape}, type: {x.dtype}")

        # move channel with real and imaginary parts to channel dimension
        # (B, H, W, 2) -> (B, 2, H, W)
        x = x.moveaxis(-1, 1)
        print(f"\t[Debug] after moving channels: {x.shape}, type: {x.dtype}")

        # ignore k-space data:
        y = sample_batch[1].to(run_device)
        print(f"\t[Debug] Original k-space shape: {y.shape}, type: {y.dtype}")
        # y_centered = fastmri_measurement_to_oasis_kspace(y, coil_maps=coil_maps, device=run_device)
        # create oasis-like k-space data from image:
        y_centered = image_to_kspace(x)
        print(f"\t[Debug] Centered k-space shape: {y_centered.shape}, type: {y_centered.dtype}")
        y = oasis_kspace_to_fastmri_measurement(y_centered, device=run_device)

    print(f"\tk-space shape {y.shape} and reference image shape: {x.shape}")

    return x, y, y_centered, coil_maps


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description=__doc__)

    # # data related arguments
    # parser.add_argument(
    #     "--source",
    #     type=Path,
    #     help="Local FastMRI directory with raw k-space .h5 files or OASIS root directory.",
    # )
    # parser.add_argument(
    #     "--dataset",
    #     choices=("fastmri", "oasis", "fastmri_multicoil", "cmrxrecon"),
    #     default="fastmri",
    # )

    # parser.add_argument("--distortion", type=str, default="", choices=DISTORTIONS)
    # parser.add_argument(
    #     "--keep_fraction",
    #     type=float,
    #     default=0.25,
    #     help="Fraction of k-space lines to keep for undersampling distortions.",
    # )
    # parser.add_argument(
    #     "--center_fraction",
    #     type=float,
    #     default=0.125,
    #     help="Fraction of low-frequency k-space lines to keep fully for undersampling distortions.",
    # )

    # # algo related arguments
    # parser.add_argument(
    #     "--algorithm",
    #     type=str,
    #     default="",
    #     choices=ALGORITHMS,
    #     help="Reconstruction algorithm applied to undistorted and distorted k-space.",
    # )
    # # inference related arguments
    # parser.add_argument("--num_samples", type=int, default=1, help="How many samples to process.")
    # parser.add_argument(
    #     "--verbose",
    #     action="store_true",
    #     help="Enable verbose output for reconstructors that support it.",
    # )
    # args = parser.parse_args()
    num_samples = 1
    keep_fraction = 0.25
    center_fraction = 0.125
    verbose = True

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    # set up device, dataset, metrics
    device = dinv.utils.get_device()

    for dataset_name, dataset_rootdir in DATASETS.items():
        print(f"=== {dataset_name} ===")

        selected_algorithms = ALGORITHMS
        selected_distortions = DISTORTIONS

        if dataset_name == "oasis":
            # split_csv = OASISSinglecoilUnetReconstructor.resolve_default_split_csv()
            dataset = OasisCenterSliceFolderDataset(
                data_path=dataset_rootdir,
            )
        elif dataset_name == "fastmri":
            dataset = dinv.datasets.FastMRISliceDataset(str(dataset_rootdir), slice_index="middle")
        elif dataset_name == "fastmri_multicoil":
            dataset = dinv.datasets.FastMRISliceDataset(
                str(dataset_rootdir),
                slice_index="middle",
                transform=dinv.datasets.MRISliceTransform(
                    estimate_coil_maps=True,
                    acs=15,
                ),
            )
        elif dataset_name == "cmrxrecon":
            dataset = dinv.datasets.CMRxReconSliceDataset(
                str(dataset_rootdir),
                data_dir="SingleCoil/Cine/TrainingSet/FullSample",
                apply_mask=False,
            )
        elif dataset_name == "prostate":
            dataset = FastMRIProstateDataset(data_path=dataset_rootdir, num_samples=num_samples)
        else:
            raise NotImplementedError(f"Invalid dataset: {dataset_name}")
        metrics = [choose_metric(m) for m in METRICS]

        for i, batch in enumerate(iter(torch.utils.data.DataLoader(dataset))):
            # exit loop if we have processed the specified number of samples
            if i >= num_samples:
                break

            print(f"{dataset_name} sample {i}...")
            x_reference, y, y_centered, coil_maps = get_measurement_sample(
                sample_batch=batch,
                dataset_name=dataset_name,
                run_device=device,
            )

            physics_clean_oasis_fft_path = OasisCenteredFFTPhysics(BaseDistortion())
            physics_clean_fastmri_path = DistortedKspaceMultiCoilMRI(
                BaseDistortion(), img_size=x_reference.shape, coil_maps=coil_maps, device=device
            )

            x_clean_oasis_fft_path = ConjugateGradientReconstructor()(
                y, physics_clean_oasis_fft_path
            )
            x_clean_fastmri_path = ConjugateGradientReconstructor()(y, physics_clean_fastmri_path)

            # reference reconstructions:
            imwrite(
                os.path.join(
                    EXPERIMENTS_DIR, f"image_{dataset_name}_sample_{i}_CG_oasis_fft_path.tiff"
                ),
                convert_image_for_save(x_clean_oasis_fft_path),
            )
            imwrite(
                os.path.join(
                    EXPERIMENTS_DIR, f"image_{dataset_name}_sample_{i}_CG_fastmri_path.tiff"
                ),
                convert_image_for_save(x_clean_fastmri_path),
            )
            imwrite(
                os.path.join(EXPERIMENTS_DIR, f"image_{dataset_name}_sample_{i}_reference.tiff"),
                convert_image_for_save(x_reference),
            )

            for distortion_name in selected_distortions:
                print(f"\t{distortion_name} ...")
                distortion_oasis_fft_path = choose_distortion(
                    distortion_name,
                    keep_fraction=keep_fraction,
                    center_fraction=center_fraction,
                    cartesian_axis=-1,
                )

                distortion_fastmri_path = choose_distortion(
                    distortion_name,
                    keep_fraction=keep_fraction,
                    center_fraction=center_fraction,
                    cartesian_axis=-2,
                )

                y_distorted_oasis_fft_path = distortion_oasis_fft_path.A(y_centered)
                y_distorted_fastmri_path = distortion_fastmri_path.A(y)

                physics_distorted_oasis_fft_path = OasisCenteredFFTPhysics(
                    distortion_oasis_fft_path
                )
                physics_distorted_fastmri_path = DistortedKspaceMultiCoilMRI(
                    distortion_fastmri_path,
                    img_size=x_reference.shape,
                    coil_maps=coil_maps,
                    device=device,
                )

                for algo_name in selected_algorithms:
                    print(f"\t\t{algo_name} ...")
                    try:
                        validate_algorithm_dataset_compatibility(dataset_name, algo_name)

                        use_oasis_path = uses_oasis_centered_path(dataset_name, algo_name)
                        if use_oasis_path:
                            y_distorted = y_distorted_oasis_fft_path
                            physics_clean = physics_clean_oasis_fft_path
                            physics_distorted = physics_distorted_oasis_fft_path
                            x_clean = x_reference
                        else:
                            y_distorted = y_distorted_fastmri_path
                            physics_clean = physics_clean_fastmri_path
                            physics_distorted = physics_distorted_fastmri_path

                        algo = choose_reconstructor(
                            algo_name,
                            img_size=y_distorted.shape[-2:],
                            device=device,
                            verbose=verbose,
                            dataset=dataset_name,
                        ).to(device)

                        # save reference and distorted k-space for debugging and visualization purposes
                        imwrite(
                            os.path.join(
                                EXPERIMENTS_DIR, f"kspace_{dataset_name}_sample_{i}_reference.tiff"
                            ),
                            _kspace_to_log_magnitude(y).numpy(),
                        )
                        imwrite(
                            os.path.join(
                                EXPERIMENTS_DIR,
                                f"kspace_{dataset_name}_sample_{i}_{distortion_name}.tiff",
                            ),
                            _kspace_to_log_magnitude(y_distorted).numpy(),
                        )

                        # actual reconstruction with the algo being evaluated
                        try:
                            if dataset_name == "prostate":
                                # prostate dataset has multiple k-space averages,
                                # so we reconstruct each average separately and then average in the image domain
                                x_corrected_averages = []
                                x_uncorrected_averages = []
                                for average in range(y_distorted.shape[0]):
                                    x_uncorrected_averages.append(
                                        algo(y_distorted[average], physics_clean)
                                    )
                                    x_corrected_averages.append(
                                        algo(y_distorted[average], physics_distorted)
                                    )

                                x_uncorrected = torch.stack(x_uncorrected_averages, dim=0).mean(
                                    dim=0
                                )
                                x_corrected = torch.stack(x_corrected_averages, dim=0).mean(dim=0)

                            else:
                                x_uncorrected = algo(y_distorted, physics_clean)
                                x_corrected = algo(y_distorted, physics_distorted)

                            # performed reconstruction images
                            imwrite(
                                os.path.join(
                                    EXPERIMENTS_DIR,
                                    f"image_{dataset_name}_sample_{i}_{distortion_name}_{algo_name}_uncorrected.tiff",
                                ),
                                convert_image_for_save(x_uncorrected),
                            )
                            imwrite(
                                os.path.join(
                                    EXPERIMENTS_DIR,
                                    f"image_{dataset_name}_sample_{i}_{distortion_name}_{algo_name}_corrected.tiff",
                                ),
                                convert_image_for_save(x_corrected),
                            )

                        except Exception as e:
                            print(
                                f"Error reconstructing algo {algo_name} with distortion {distortion_name} on sample {i}: {e}"
                            )

                        # dinv.utils.plot(
                        #     {
                        #         "Undistorted ksp, CG recon": x_clean,
                        #         "Distorted ksp, CG recon": x_distorted,
                        #         f"Distorted ksp, {algo_name} recon, uncorrected": x_uncorrected,
                        #         f"Distorted ksp, {algo_name} recon, corrected": x_corrected,
                        #     },
                        #     subtitles=[
                        #         "",
                        #         "",
                        #         "\n".join(
                        #             f"{m.__class__.__name__} {m(x_uncorrected, x_clean).item():.2f}"
                        #             for m in metrics
                        #         ),
                        #         "\n".join(
                        #             f"{m.__class__.__name__} {m(x_corrected, x_clean).item():.2f}"
                        #             for m in metrics
                        #         ),
                        #     ],
                        #     show=False,
                        #     close=True,
                        #     suptitle=f"Algo {algo_name}, distortion {distortion_name}, Sample {i}",
                        #     save_fn=REPORT_DIR / f"ALGO_{algo_name}_{distortion_name}_sample_{i}.png",
                        #     fontsize=3,
                        # )
                    except Exception as e:
                        print(
                            f"\t\tError processing algo {algo_name}, distortion {distortion_name}, sample {i}: {e}"
                        )

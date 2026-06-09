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
    choose_distortion_with_params,
    BaseDistortion,
    DistortedKspaceMultiCoilMRI,
    image_to_shifted_kspace,
)
from mri_recon.reconstruction import (
    ConjugateGradientReconstructor,
    OASISSinglecoilUnetReconstructor,
    choose_reconstructor,
    uses_oasis_centered_path,
    compatible_dataset_with_reconstructor,
    EXPLICIT_UNET_ALGORITHMS,
)
from mri_recon.utils import (
    OasisCenteredFFTPhysics,
    OasisSliceDataset,
    fastmri_measurement_to_oasis_kspace,
    kspace_to_image,
    save_kspace_plot,
)

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
    {"BaseDistortion": {}},
    {
        "CartesianUndersamplingVariableDensity": {
            "keep_fraction": 0.25,
            "center_fraction": 0.125,
        }
    },
    # {"CartesianUndersamplingUniformRandom": {
    #     "keep_fraction": 0.25,
    #     "center_fraction": 0.125,
    # }},
    # {"CartesianUndersamplingUniformRandomZeroACS": {
    #     "keep_fraction": 0.25,
    # }},
    # {"CartesianUndersamplingEquispaced": {
    #     "keep_fraction": 0.25,
    #     "center_fraction": 0.125,
    # }},
    # {"CartesianUndersamplingEquispacedZeroACS": {
    #     "keep_fraction": 0.25,
    # }},
    # {"PartialFourier": {
    #     "side": "high",
    # }},
    # {"PhaseEncodeGhosting":  {
    #     "line_period": 2,
    #     "line_offset": 1,
    #     "phase_error_degrees":  90.0,
    #     "corrupted_line_scale": 1.0,
    # }},
    # {"SegmentedTranslationMotion": {
    #     "shift_x_pixels": [0.0, 20.0, 50.0, -50.0],
    #     "shift_y_pixels": [0.0, 10.0, -20.0, 20.0],
    # }},
    # {"SegmentedRotationalMotion": {
    #     #"angle_radians": [0.0, torch.pi / 20, -torch.pi / 24, torch.pi / 16]
    #     "angle_degrees": [0.0, 18.0, -15.0, 22.5],
    # }},
    # {"TranslationMotion": {
    #     "shift_x_pixels": 60,
    #     "shift_y_pixels": 10,
    # }},
    # {"RotationalMotion": {
    #     # angle_radians=torch.pi / 6
    #     "angle_degrees": 60.0,
    # }},
    {
        "OffCenterAnisotropicGaussianBiasField": {
            "width_x_fraction": 0.2,
            "width_y_fraction": 0.35,
            "center_x_fraction": 0.15,
            "center_y_fraction": -0.1,
            "edge_gain": 0.3,
        }
    },
    {
        "GaussianBiasField": {
            "width_fraction": 0.35,
            "edge_gain": 0.4,
        }
    },
    # {"AnisotropicLP": {
    #     "kx_radius_fraction": 1.0,
    #     "ky_radius_fraction": 0.25,
    # }},
    # {"HannTaperLP": {
    #     "radius_fraction": 0.35,
    #     "transition_fraction": 0.4,
    # }},
    # {"KaiserTaperLP": {
    #     "radius_fraction": 0.35,
    #     "transition_fraction": 0.4,
    #     "beta": 8.6,
    # }},
    # {"GaussianNoise": {
    #     "sigma": 0.00001,
    # }},
    # {"IsotropicLP": {
    #     "radius_fraction": 0.1,
    # }},
    # {"RadialHighPassEmphasis": {
    #     "alpha": 0.4,
    # }}
]

METRICS = [
    "PSNR",
    # "NMSE",
    # "SSIM",
    # "HaarPSI",
    # "SharpnessIndex",
    # "BlurStrength",
]


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
        x = sample_batch["x"].to(run_device)
        y = image_to_shifted_kspace(x)
        coil_maps = None

    elif dataset_name in ("fastmri", "fastmri_multicoil"):
        x = sample_batch[0].to(run_device)
        y = sample_batch[1].to(run_device)
        coil_maps = (
            sample_batch[2]["coil_maps"].to(run_device)
            if isinstance(sample_batch, (tuple, list))
            and len(sample_batch) == 3
            and "coil_maps" in sample_batch[2]
            else None
        )
    elif dataset_name in ("cmrxrecon"):
        x = sample_batch[0].to(run_device)
        y = sample_batch[1].to(run_device)
        coil_maps = (
            sample_batch[2]["coil_maps"].to(run_device)
            if isinstance(sample_batch, (tuple, list))
            and len(sample_batch) == 3
            and "coil_maps" in sample_batch[2]
            else None
        )

        if use_oasis_fft_path:
            y = fastmri_measurement_to_oasis_kspace(y, coil_maps=coil_maps, device=run_device)

    return x, y, coil_maps


def build_physics_pair(
    image_shape: tuple[int, int],
    distortion_operator: BaseDistortion,
    run_device: torch.device | str,
    use_oasis_fft_path: bool,
    coil_maps: torch.Tensor | None = None,
) -> tuple[object, object]:
    """Build clean and distorted physics operators for the active path."""

    if use_oasis_fft_path:
        return OasisCenteredFFTPhysics(BaseDistortion()), OasisCenteredFFTPhysics(
            distortion_operator
        )

    clean_physics = DistortedKspaceMultiCoilMRI(
        distortion=BaseDistortion(),
        img_size=(1, 2, *image_shape),
        coil_maps=coil_maps,
        device=run_device,
    )
    distorted_physics = DistortedKspaceMultiCoilMRI(
        distortion=distortion_operator,
        img_size=(1, 2, *image_shape),
        coil_maps=coil_maps,
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
    parser.add_argument(
        "--dataset",
        choices=("fastmri", "oasis", "fastmri_multicoil", "cmrxrecon"),
        default="fastmri",
    )

    parser.add_argument("--distortion", type=str, default="", choices=DISTORTIONS)

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

    REPORT_DIR = Path("reports") / Path(args.dataset + "_inference_plot")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # skip non-compatible algorithm-dataset pairs
    selected_algorithms = [
        algo_name
        for algo_name in selected_algorithms
        if compatible_dataset_with_reconstructor(args.dataset, algo_name)
    ]

    # set up report dir
    if args.dataset not in ["fastmri", "oasis", "fastmri_multicoil", "cmrxrecon"]:
        raise NotImplementedError(f"Invalid dataset: {args.dataset}")

    # set up device, dataset, metrics
    device = dinv.utils.get_device()
    if args.dataset == "oasis":
        split_csv = OASISSinglecoilUnetReconstructor.resolve_default_split_csv()
        dataset = OasisSliceDataset(
            data_path=args.source,
            split_csv=split_csv,
            sample_rate=0.6,
        )
    elif args.dataset == "fastmri":
        dataset = dinv.datasets.FastMRISliceDataset(str(args.source), slice_index="middle")
    elif args.dataset == "fastmri_multicoil":
        dataset = dinv.datasets.FastMRISliceDataset(
            str(args.source),
            slice_index="middle",
            transform=dinv.datasets.MRISliceTransform(
                estimate_coil_maps=True,
                acs=15,
            ),
        )
    elif args.dataset == "cmrxrecon":
        dataset = dinv.datasets.CMRxReconSliceDataset(
            str(args.source), data_dir="SingleCoil/Cine/TrainingSet/FullSample", apply_mask=False
        )
    else:
        raise NotImplementedError(f"Invalid dataset: {args.dataset}")
    metrics = [choose_metric(m) for m in METRICS]

    for i, batch in enumerate(iter(torch.utils.data.DataLoader(dataset))):
        # exit loop if we have processed the specified number of samples
        if i >= args.num_samples:
            break

        for algo_name in selected_algorithms:
            try:
                use_oasis_path = uses_oasis_centered_path(algo_name)
                x_reference, y, coil_maps = prepare_measurement_sample(
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

                for selected_distortion in selected_distortions:
                    for distortion_name, distortion_params in selected_distortion.items():
                        distortion = choose_distortion_with_params(
                            distortion_name,
                            **distortion_params,
                            cartesian_axis=-1 if use_oasis_path else -2,
                        )

                        physics_clean, physics = build_physics_pair(
                            image_shape=y.shape[-2:],
                            distortion_operator=distortion,
                            run_device=device,
                            use_oasis_fft_path=use_oasis_path,
                            coil_maps=coil_maps,
                        )
                        y_distorted = distortion.A(y)

                        # generate reference reconstructions (CG) for both clean and distorted k-space
                        # without correction for the distortion, i.e. using physics_clean in both cases
                        if use_oasis_path:
                            x_clean = x_reference
                            x_distorted = kspace_to_image(y_distorted)
                        else:
                            x_clean = ConjugateGradientReconstructor()(y, physics_clean)
                            x_distorted = ConjugateGradientReconstructor()(
                                y_distorted, physics_clean
                            )

                        if x_clean.shape[-2:] != x_reference.shape[-2:]:
                            x_clean = physics_clean.crop(x_clean, shape=x_reference.shape[-2:])

                        if x_distorted.shape[-2:] != x_reference.shape[-2:]:
                            x_distorted = physics.crop(x_distorted, shape=x_reference.shape[-2:])

                        save_kspace_plot(
                            y,
                            y_distorted,
                            REPORT_DIR / f"DISTORTION_{algo_name}_{distortion_name}_sample_{i}.png",
                            distortion_name,
                        )

                        print(
                            f"Evaluating algo {algo_name}, distortion {distortion_name}, sample {i}..."
                        )

                        # actual reconstruction with the algo being evaluated
                        x_uncorrected = algo(y_distorted, physics_clean)

                        # crop recostructed image to reference image size:
                        if x_uncorrected.shape[-2:] != x_reference.shape[-2:]:
                            x_uncorrected = physics_clean.crop(
                                x_uncorrected, shape=x_reference.shape[-2:]
                            )

                        x_corrected = algo(y_distorted, physics)
                        if x_corrected.shape[-2:] != x_reference.shape[-2:]:
                            x_corrected = physics.crop(x_corrected, shape=x_reference.shape[-2:])

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
                            save_fn=REPORT_DIR
                            / f"ALGO_{algo_name}_{distortion_name}_sample_{i}.png",
                            fontsize=3,
                        )
            except Exception as e:
                print(f"Error processing algo {algo_name}, sample {i}: {e}")

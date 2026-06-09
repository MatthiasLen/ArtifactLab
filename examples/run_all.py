"""Inference various reconstructors for various distortion operators.

Usage:
    python examples/run_all.py config.yaml
"""

import os
import sys
import glob
import SimpleITK as sitk
import numpy as np
import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import deepinv as dinv
import torch
import yaml
from tifffile import imwrite, imread

from mri_recon.distortions import (
    BaseDistortion,
    DistortedKspaceMultiCoilMRI,
    choose_distortion_with_params,
    image_to_shifted_kspace,
)
from mri_recon.reconstruction import (
    choose_reconstructor,
    uses_oasis_centered_path,
    compatible_dataset_with_reconstructor,
)
from mri_recon.utils import (
    OasisCenteredFFTPhysics,
    OasisCenterSliceFolderDataset,
    FastMRIProstateDataset,
    fastmri_measurement_to_oasis_kspace,
    image_to_kspace,
    _kspace_to_log_magnitude,
    convert_image_for_save,
)


def get_measurement_sample(
    sample_batch: object,
    dataset_name: str,
    run_device: torch.device | str,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Prepare one input measurement and its clean image reference.

    Always prepare a (fast-MRI-like) non-centered k-space measurement
    as well as a (oasis-like) centered k-space version of the measurement
    and a reference reconstruction in the image domain.
    """
    coil_maps = None
    if dataset_name == "oasis":
        # reference image, shape: (B, 2, H, W) dtype: float32
        x = sample_batch["x"].to(run_device)
        # centered k-space data, shape: (B, 2, H, W) dtype: float32
        y_centered = image_to_kspace(x)
        # k-space data, shape: (B, 2, H, W) dtype: float32
        # y = oasis_kspace_to_fastmri_measurement(y_centered)
        y = image_to_shifted_kspace(x)
    elif dataset_name == "fastmri_knee":
        # reference image, shape: (B, 1, H/2, H/2) dtype: float32
        x = sample_batch[0].to(run_device)
        # kspace data, shape: (B, 2, H, W) dtype: float32
        y = sample_batch[1].to(run_device)
        # centered k-space data, shape: (B, 2, H, W) dtype: float32
        y_centered = fastmri_measurement_to_oasis_kspace(y, device=run_device)
        # reconstructed reference image:
        # shape: (B, 1, H, W) dtype: float32
    elif dataset_name == "fastmri_brain":
        # reference image, shape: (B, 1, H/2, H/2) dtype: float32
        x = sample_batch[0].to(run_device)
        # kspace data, shape: (B, 2, num_coils, H, W) dtype: float32
        y = sample_batch[1].to(run_device)
        # coil maps, shape: (B, num_coils, H, W) dtype: complex64
        coil_maps = (
            sample_batch[2]["coil_maps"].to(run_device)
            if isinstance(sample_batch, (tuple, list))
            and len(sample_batch) == 3
            and "coil_maps" in sample_batch[2]
            else None
        )
        # centered k-space data, shape: (B, 2, H, W) dtype: float32
        y_centered = fastmri_measurement_to_oasis_kspace(y, coil_maps=coil_maps, device=run_device)

    elif dataset_name == "cmrxrecon":
        # reference image, shape: (B, 2, n_timepoints, (n_coils), H, W)
        x = sample_batch[0].to(run_device)
        # k-space data, shape: (B, 2, n_timepoints, (n_coils), H, W) dtype: float32
        y = sample_batch[1].to(run_device)

        # maybe not needed, as there are no coil maps in the current
        # cmrxrecon sample
        coil_maps = (
            sample_batch[2]["coil_maps"].to(run_device)
            if isinstance(sample_batch, (tuple, list))
            and len(sample_batch) == 3
            and "coil_maps" in sample_batch[2]
            else None
        )
        # centered k-space data, shape: (B, 2, n_timepoints, (n_coils), H, W) dtype: float32
        y_centered = fastmri_measurement_to_oasis_kspace(y, coil_maps=coil_maps, device=run_device)

        # select the center timepoint in order to simplify the evaluation of the reconstruction algorithms
        center_time_point = y.shape[2] // 2
        y = y[:, :, center_time_point, ...]
        y_centered = y_centered[:, :, center_time_point, ...]
        x = x[:, :, center_time_point, ...]

    elif dataset_name == "fastmri_prostate":
        # reference image, shape: (B, W, H): dtype float32
        x = sample_batch[0].to(run_device)

        # add zero imaginary channel:
        # (B, H, W) -> (B, 2, H, W)
        x = torch.stack([x, torch.zeros_like(x)], dim=1)

        # (B, 2, H, W)
        y_centered = image_to_kspace(x)
        # y = oasis_kspace_to_fastmri_measurement(y_centered)
        y = image_to_shifted_kspace(x)

    print("Debug shapes:")
    print("x: ", x.shape)
    print("y: ", y.shape)
    print("y_centered: ", y_centered.shape)
    print("coil_maps: ", coil_maps.shape if coil_maps is not None else "None")

    return x, y, y_centered, coil_maps


def run_all(config) -> None:
    os.makedirs(config["results_dir"], exist_ok=True)

    # set up device
    device = dinv.utils.get_device()

    for dataset_name, dataset_rootdir in config["data"].items():
        print(f"=== {dataset_name} ===")

        # initialize dataset
        if dataset_name == "oasis":
            dataset = OasisCenterSliceFolderDataset(
                data_path=dataset_rootdir,
            )
        elif dataset_name == "fastmri_knee":
            dataset = dinv.datasets.FastMRISliceDataset(str(dataset_rootdir), slice_index="middle")
        elif dataset_name == "fastmri_brain":
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
        elif dataset_name == "fastmri_prostate":
            dataset = FastMRIProstateDataset(
                data_path=dataset_rootdir, num_samples=config["num_samples"], slice_index="middle"
            )
        else:
            raise NotImplementedError(f"Invalid dataset: {dataset_name}")

        # loop through samples of dataset
        for i, batch in enumerate(iter(torch.utils.data.DataLoader(dataset))):
            # exit loop if we have processed the specified number of samples
            if (config["num_samples"] is not None) and (i >= config["num_samples"]):
                break

            print(f"{dataset_name} sample {i}...")
            x_reference, y, y_centered, coil_maps = get_measurement_sample(
                sample_batch=batch,
                dataset_name=dataset_name,
                run_device=device,
            )

            # save reference image
            imwrite(
                os.path.join(
                    config["results_dir"], f"image_{dataset_name}_sample_{i}_reference.tiff"
                ),
                convert_image_for_save(x_reference),
            )

            # use fast-mri type samples first, later proceed with oasis-centered fft path
            physics_clean = DistortedKspaceMultiCoilMRI(
                BaseDistortion(), img_size=y.shape[-2:], coil_maps=coil_maps, device=device
            )

            for selected_distortion in config["distortions"]:
                for distortion_name, distortion_params in selected_distortion.items():
                    print(f"\t{distortion_name} ...")
                    distortion_name_with_params = distortion_name + "".join(
                        [p[0] + "=" + str(v) for p, v in distortion_params.items()]
                    )

                    distortion = choose_distortion_with_params(
                        distortion_name,
                        **distortion_params,
                        # keep_fraction=config["keep_fraction"],
                        # center_fraction=config["center_fraction"],
                        cartesian_axis=-2,
                    )

                    y_distorted = distortion.A(y)

                    physics_distorted = DistortedKspaceMultiCoilMRI(
                        distortion,
                        img_size=y.shape[-2:],
                        coil_maps=coil_maps,
                        device=device,
                    )

                    for reconstructor_name in config["reconstruction_algorithms"]:
                        # only run on reconstructors, that use the fastmri-like k-space
                        if not uses_oasis_centered_path(reconstructor_name):
                            print(f"\t\t{reconstructor_name} ...")
                            start = datetime.now()
                            if compatible_dataset_with_reconstructor(
                                dataset_name, reconstructor_name
                            ):
                                reconstructor = choose_reconstructor(
                                    reconstructor_name,
                                    img_size=y_distorted.shape[-2:],
                                    device=device,
                                    verbose=config["verbose"],
                                ).to(device)

                                # save reference and distorted k-space for debugging purposes
                                imwrite(
                                    os.path.join(
                                        config["results_dir"],
                                        f"kspace_{dataset_name}_sample_{i}_reference.tiff",
                                    ),
                                    _kspace_to_log_magnitude(y).numpy(),
                                )
                                imwrite(
                                    os.path.join(
                                        config["results_dir"],
                                        f"kspace_{dataset_name}_sample_{i}_{distortion_name_with_params}.tiff",
                                    ),
                                    _kspace_to_log_magnitude(y_distorted).numpy(),
                                )

                                # actual reconstruction with the selected reconstructor
                                try:
                                    x_uncorrected = reconstructor(y_distorted, physics_clean)
                                    x_corrected = reconstructor(y_distorted, physics_distorted)

                                    # crop recostructed image to reference image size:
                                    if x_uncorrected.shape[-2:] != x_reference.shape[-2:]:
                                        x_uncorrected = physics_clean.crop(
                                            x_uncorrected, shape=x_reference.shape[-2:]
                                        )

                                    if x_corrected.shape[-2:] != x_reference.shape[-2:]:
                                        x_corrected = physics_distorted.crop(
                                            x_corrected, shape=x_reference.shape[-2:]
                                        )

                                    # save reconstructed images
                                    imwrite(
                                        os.path.join(
                                            config["results_dir"],
                                            f"image_{dataset_name}_sample_{i}_{distortion_name_with_params}_{reconstructor_name}_uncorrected.tiff",
                                        ),
                                        convert_image_for_save(x_uncorrected),
                                    )
                                    imwrite(
                                        os.path.join(
                                            config["results_dir"],
                                            f"image_{dataset_name}_sample_{i}_{distortion_name_with_params}_{reconstructor_name}_corrected.tiff",
                                        ),
                                        convert_image_for_save(x_corrected),
                                    )
                                    print(f"\t\t... done in {datetime.now() - start}")

                                except Exception as e:
                                    print(f"Error using {reconstructor_name}: {e}")

                            else:
                                print(f"\t\t ... not compatible with {dataset_name}")

            # now proceed with oasis-centered fft path
            physics_clean = OasisCenteredFFTPhysics(BaseDistortion())

            for selected_distortion in config["distortions"]:
                for distortion_name, distortion_params in selected_distortion.items():
                    print(f"\t{distortion_name} ...")
                    distortion_name_with_params = distortion_name + "".join(
                        [p[0] + "=" + str(v) for p, v in distortion_params.items()]
                    )

                    distortion = choose_distortion_with_params(
                        distortion_name,
                        **distortion_params,
                        cartesian_axis=-1,
                    )

                    y_distorted = torch.fft.fftshift(
                        distortion.A(torch.fft.fftshift(y_centered, dim=(-1, -2))), dim=(-2, -1)
                    )

                    physics_distorted = OasisCenteredFFTPhysics(distortion)

                    for reconstructor_name in config["reconstruction_algorithms"]:
                        # skip all reconstructors, that don't use the oasis-centered path
                        if uses_oasis_centered_path(reconstructor_name):
                            print(f"\t\t{reconstructor_name} ...")
                            start = datetime.now()
                            if compatible_dataset_with_reconstructor(
                                dataset_name, reconstructor_name
                            ):
                                reconstructor = choose_reconstructor(
                                    reconstructor_name,
                                    img_size=y_distorted.shape[-2:],
                                    device=device,
                                    verbose=config["verbose"],
                                ).to(device)

                                # save reference and distorted k-space for debugging purposes
                                imwrite(
                                    os.path.join(
                                        config["results_dir"],
                                        f"kspace_centered_{dataset_name}_sample_{i}_reference.tiff",
                                    ),
                                    _kspace_to_log_magnitude(y_centered).numpy(),
                                )
                                imwrite(
                                    os.path.join(
                                        config["results_dir"],
                                        f"kspace_centered_{dataset_name}_sample_{i}_{distortion_name_with_params}.tiff",
                                    ),
                                    _kspace_to_log_magnitude(y_distorted).numpy(),
                                )

                                # actual reconstruction with the algo being evaluated
                                try:
                                    x_uncorrected = reconstructor(y_distorted, physics_clean)
                                    x_corrected = reconstructor(y_distorted, physics_distorted)

                                    if x_uncorrected.shape[-2:] != x_reference.shape[-2:]:
                                        x_uncorrected = physics_clean.crop(
                                            x_uncorrected, shape=x_reference.shape[-2:]
                                        )

                                    if x_corrected.shape[-2:] != x_reference.shape[-2:]:
                                        x_corrected = physics_distorted.crop(
                                            x_corrected, shape=x_reference.shape[-2:]
                                        )

                                    # save reconstructed images
                                    imwrite(
                                        os.path.join(
                                            config["results_dir"],
                                            f"image_{dataset_name}_sample_{i}_{distortion_name_with_params}_{reconstructor_name}_uncorrected.tiff",
                                        ),
                                        convert_image_for_save(x_uncorrected),
                                    )
                                    imwrite(
                                        os.path.join(
                                            config["results_dir"],
                                            f"image_{dataset_name}_sample_{i}_{distortion_name_with_params}_{reconstructor_name}_corrected.tiff",
                                        ),
                                        convert_image_for_save(x_corrected),
                                    )
                                    print(f"\t\t... done in {datetime.now() - start}")

                                except Exception as e:
                                    print(
                                        f"\t\tError using {reconstructor_name} with distortion {distortion_name_with_params} on sample {i}: {e}"
                                    )

                            else:
                                print(f"\t\t ... not compatible with {dataset_name}")

            if config["add_N4Correction"]:
                reconstructed_bias_field_images = glob.glob(
                    os.path.join(config["results_dir"], "*BiasField*corrected.tiff")
                )
                reference_images = glob.glob(
                    os.path.join(config["results_dir"], "image*reference.tiff")
                )
                images_for_n4_correction = reconstructed_bias_field_images + reference_images
                with tqdm.tqdm(
                    total=len(images_for_n4_correction), desc="Applying N4 Bias Field Correction"
                ) as pbar:
                    for reconstructed_image_filename in images_for_n4_correction:
                        print(reconstructed_image_filename)
                        reconstructed_image = imread(reconstructed_image_filename).squeeze()
                        if len(reconstructed_image.shape) == 2:
                            sitk_img = sitk.GetImageFromArray(reconstructed_image.T)
                            # sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8).T)

                            corrector = sitk.N4BiasFieldCorrectionImageFilter()
                            corrector.SetMaximumNumberOfIterations([50] * 4)
                            corrector.SetConvergenceThreshold(0.001)

                            # sitk.N4BiasFieldCorrectionImageFilter.Execute(corrector, sitk_img, sitk_mask)
                            sitk.N4BiasFieldCorrectionImageFilter.Execute(corrector, sitk_img)

                            log_bias_sitk = corrector.GetLogBiasFieldAsImage(sitk_img)
                            bias_n4 = np.exp(sitk.GetArrayFromImage(log_bias_sitk).T)
                            reconstructed_image_n4 = reconstructed_image / np.where(
                                bias_n4 > 0, bias_n4, 1.0
                            )
                            imwrite(
                                reconstructed_image_filename.replace(".tiff", "_N4.tiff"),
                                reconstructed_image_n4,
                            )

                        else:
                            print("Skipping N4 Bias Field correction for image")
                            print(os.path.basename(reconstructed_image_filename))
                            print("which as shape ", reconstructed_image.shape)

                        pbar.update(1)


if __name__ == "__main__":
    # read config file in yaml format as first argument from commmand line
    if len(sys.argv) < 2:
        print("Usage: python examples/run_all.py <config_file.yaml>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    run_all(config)

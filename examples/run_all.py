"""Inference various reconstructors for various distortion operators.

Usage:
    python examples/fastmri_inference_plot.py --source ../ram-experiments/data/fastmri/knee/singlecoil_val
"""

import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import deepinv as dinv
import torch
import yaml
from tifffile import imwrite

from mri_recon.distortions import (
    BaseDistortion,
    DistortedKspaceMultiCoilMRI,
    choose_distortion,
)
from mri_recon.reconstruction import (
    ConjugateGradientReconstructor,
    choose_reconstructor,
    uses_oasis_centered_path,
    compatible_dataset_with_reconstructor,
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
        y = oasis_kspace_to_fastmri_measurement(y_centered, device=run_device)
    elif dataset_name == "fastmri_knee":
        # reference image, shape: (B, 1, H/2, H/2) dtype: float32
        x = sample_batch[0].to(run_device)
        # kspace data, shape: (B, 2, H, W) dtype: float32
        y = sample_batch[1].to(run_device)
        # centered k-space data, shape: (B, 2, H, W) dtype: float32
        y_centered = fastmri_measurement_to_oasis_kspace(y, device=run_device)
        # reconstructed reference image:
        # shape: (B, 1, H, W) dtype: float32
        print("stop for testing")
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
        print(f"\t[Debug] Reference image shape: {x.shape}, dtype: {x.dtype}")
        # k-space data, shape: (B, 2, n_timepoints, (n_coils), H, W) dtype: float32
        y = sample_batch[1].to(run_device)
        print(f"\t[Debug] k-space shape: {y.shape}, dtype: {y.dtype}")

        # not available for all samples, either None or 
        # shape (1, num_coils, H, W)
        coil_maps = (
            sample_batch[2]["coil_maps"].to(run_device)
            if isinstance(sample_batch, (tuple, list))
            and len(sample_batch) == 3
            and "coil_maps" in sample_batch[2]
            else None
        )
        print(f"\t[Debug] Coil maps shape: {coil_maps.shape if coil_maps is not None else None}, dtype: {coil_maps.dtype if coil_maps is not None else None}")
        # centered k-space data, shape: (B, 2, H, W) dtype: float32
        y_centered = fastmri_measurement_to_oasis_kspace(y, coil_maps=coil_maps, device=run_device)
        print(f"\t[Debug] Centered k-space shape: {y_centered.shape}, dtype: {y_centered.dtype}")
        # reconstruct coil-combined image reference from multi-coil k-space data using
        # integrated espirit sensitivity map estimation, RSS coil combination
        
        #x = fastmri_measurement_to_image(y, coil_maps=coil_maps, rss=True)
        #print(f"\t[Debug] Reference image shape: {x.shape}, dtype: {x.dtype}")
    elif dataset_name == "fastmri_prostate":
        # reference image, shape: (slices, W, H): dtype float32
        x = sample_batch[0].to(run_device)
        print(f"\t[Debug] Reference image shape: {x.shape}, type: {x.dtype}")

        # add zero imaginary channel:
        # (B, slices, H, W) -> (B, 2, slices, H, W)
        x = torch.stack([x, torch.zeros_like(x)], dim=1)
        

        
        # y_centered = fastmri_measurement_to_oasis_kspace(y, coil_maps=coil_maps, device=run_device)
        # create oasis-like k-space data from image:
        y_centered = image_to_kspace(x)
        print(f"\t[Debug] Centered k-space shape: {y_centered.shape}, type: {y_centered.dtype}")
        y = oasis_kspace_to_fastmri_measurement(y_centered, device=run_device)

    print(f"\tk-space shape {y.shape}[{y.dtype}] and reference image shape: {x.shape}[{x.dtype}]")
    if coil_maps is not None:
        print(f"\tcoil maps shape: {coil_maps.shape}[{coil_maps.dtype}]")

    return x, y, y_centered, coil_maps


if __name__ == "__main__":
    
    # read config file in yaml format as first argument from commmand line
    if len(sys.argv) < 2:
        print("Usage: python examples/run_all.py <config_file.yaml>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

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
            dataset = FastMRIProstateDataset(data_path=dataset_rootdir, num_samples=config["num_samples"], slice_index="middle")
        else:
            raise NotImplementedError(f"Invalid dataset: {dataset_name}")

        # loop through samples of dataset
        for i, batch in enumerate(iter(torch.utils.data.DataLoader(dataset))):

            # exit loop if we have processed the specified number of samples
            if i >= config["num_samples"]:
                break

            print(f"{dataset_name} sample {i}...")
            x_reference, y, y_centered, coil_maps = get_measurement_sample(
                sample_batch=batch,
                dataset_name=dataset_name,
                run_device=device,
            )

            # save reference image
            imwrite(
                os.path.join(config["results_dir"], f"image_{dataset_name}_sample_{i}_reference.tiff"),
                convert_image_for_save(x_reference),
            )

            # use fast-mri type samples first, later proceed with oasis-centered fft path
            physics_clean = DistortedKspaceMultiCoilMRI(
                BaseDistortion(), img_size=y.shape[-2:], coil_maps=coil_maps, device=device
            )
            
            # reference from dataset:
            for distortion_name in config["distortions"]:
                print(f"\t{distortion_name} ...")

                distortion = choose_distortion(
                    distortion_name,
                    keep_fraction=config["keep_fraction"],
                    center_fraction=config["center_fraction"],
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
                    print(f"\t\t{reconstructor_name} ...")
                    if compatible_dataset_with_reconstructor(dataset_name, reconstructor_name):

                        # only run on reconstructors, that use the fastmri-like k-space
                        if not uses_oasis_centered_path(dataset_name, reconstructor_name):

                            reconstructor = choose_reconstructor(
                                reconstructor_name,
                                img_size=y_distorted.shape[-2:],
                                device=device,
                                verbose=config["verbose"],
                            ).to(device)

                            # save reference and distorted k-space for debugging purposes
                            imwrite(
                                os.path.join(
                                    config["results_dir"], f"kspace_{dataset_name}_sample_{i}_reference.tiff"
                                ),
                                _kspace_to_log_magnitude(y).numpy(),
                            )
                            imwrite(
                                os.path.join(
                                    config["results_dir"],
                                    f"kspace_{dataset_name}_sample_{i}_{distortion_name}.tiff",
                                ),
                                _kspace_to_log_magnitude(y_distorted).numpy(),
                            )

                            # actual reconstruction with the selected reconstructor
                            try:
                                
                                x_uncorrected = reconstructor(y_distorted, physics_clean)
                                x_corrected = reconstructor(y_distorted, physics_distorted)


                                # crop recostructed image to reference image size:
                                if x_uncorrected.shape[-2:] != x_reference.shape[-2:]:
                                    x_uncorrected = physics_clean.crop(x_uncorrected, shape=x_reference.shape[-2:])

                                if x_corrected.shape[-2:] != x_reference.shape[-2:]:
                                    x_corrected_clean = physics_distorted.crop(x_corrected, shape=x_reference.shape[-2:])

                                # save reconstructed images
                                imwrite(
                                    os.path.join(
                                        config["results_dir"],
                                        f"image_{dataset_name}_sample_{i}_{distortion_name}_{reconstructor_name}_uncorrected.tiff",
                                    ),
                                    convert_image_for_save(x_uncorrected),
                                )
                                imwrite(
                                    os.path.join(
                                        config["results_dir"],
                                        f"image_{dataset_name}_sample_{i}_{distortion_name}_{reconstructor_name}_corrected.tiff",
                                    ),
                                    convert_image_for_save(x_corrected),
                                )

                            except Exception as e:
                                print(
                                    f"Error using {reconstructor_name}: {e}"
                                )

                            
                    else:
                        print(f"\t\t ... not compatible with {dataset_name}")


            # now proceed with oasis-centered fft path
            physics_clean = OasisCenteredFFTPhysics(BaseDistortion())

            for distortion_name in config["distortions"]:
                print(f"\t{distortion_name} ...")
                distortion = choose_distortion(
                    distortion_name,
                    keep_fraction=config["keep_fraction"],
                    center_fraction=config["center_fraction"],
                    cartesian_axis=-1,
                )


                y_distorted = torch.fft.fftshift(distortion.A(torch.fft.fftshift(y_centered, dim=(-1, -2))), dim=(-2, -1))

                physics_distorted = OasisCenteredFFTPhysics(
                    distortion
                )

                for reconstructor_name in config["reconstruction_algorithms"]:
                    print(f"\t\t{reconstructor_name} ...")
                    if compatible_dataset_with_reconstructor(dataset_name, reconstructor_name):

                        # skip all reconstructors, that don't use the oasis-centered path
                        if uses_oasis_centered_path(dataset_name, reconstructor_name):
                            
                            reconstructor = choose_reconstructor(
                                reconstructor_name,
                                img_size=y_distorted.shape[-2:],
                                device=device,
                                verbose=config["verbose"],
                            ).to(device)

                            # save reference and distorted k-space for debugging purposes
                            imwrite(
                                os.path.join(
                                    config["results_dir"], f"kspace_centered_{dataset_name}_sample_{i}_reference.tiff"
                                ),
                                _kspace_to_log_magnitude(y_centered).numpy(),
                            )
                            imwrite(
                                os.path.join(
                                    config["results_dir"],
                                    f"kspace_centered_{dataset_name}_sample_{i}_{distortion_name}.tiff",
                                ),
                                _kspace_to_log_magnitude(y_distorted).numpy(),
                            )

                            # actual reconstruction with the algo being evaluated
                            try:
                                
                                x_uncorrected = reconstructor(y_distorted, physics_clean)
                                x_corrected = reconstructor(y_distorted, physics_distorted)

                                if x_uncorrected.shape[-2:] != x_reference.shape[-2:]:
                                    x_uncorrected = physics_clean.crop(x_uncorrected, shape=x_reference.shape[-2:])

                                if x_corrected.shape[-2:] != x_reference.shape[-2:]:
                                    x_corrected_clean = physics_distorted.crop(x_corrected, shape=x_reference.shape[-2:])


                                # save reconstructed images
                                imwrite(
                                    os.path.join(
                                        config["results_dir"],
                                        f"image_{dataset_name}_sample_{i}_{distortion_name}_{reconstructor_name}_uncorrected.tiff",
                                    ),
                                    convert_image_for_save(x_uncorrected),
                                )
                                imwrite(
                                    os.path.join(
                                        config["results_dir"],
                                        f"image_{dataset_name}_sample_{i}_{distortion_name}_{reconstructor_name}_corrected.tiff",
                                    ),
                                    convert_image_for_save(x_corrected),
                                )

                            except Exception as e:
                                print(
                                    f"\t\tError using {reconstructor_name} with distortion {distortion_name} on sample {i}: {e}"
                                )

                    else:
                        print(f"\t\t ... not compatible with {dataset_name}")


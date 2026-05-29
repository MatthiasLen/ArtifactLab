from datetime import datetime
import os
import sys
import h5py
from tifffile import imwrite
import torch
import deepinv as dinv
import pandas as pd


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mri_recon.distortions import DistortedKspaceMultiCoilMRI, BaseDistortion, choose_distortion
from mri_recon.reconstruction import choose_reconstructor
from mri_recon.utils.oasis_adapter import (
    DistortedOasisMeasurement,
    fastmri_measurement_to_oasis_kspace,
    kspace_to_image,
)


sensitivity_map_estimation_algorithm = [
    "espirit",
    # "unity",
    # "birdcage",
]


DISTORTIONS = [
    "BaseDistortion",
    # "PhaseEncodeGhosting",
    # "CartesianUndersamplingVariableDensity",
    # "CartesianUndersamplingUniformRandom",
    # "HannTaperLP",
    # "KaiserTaperLP",
    # "RadialHighPassEmphasis",
    # "IsotropicLP",
    # "OffCenterAnisotropicGaussianKspaceBiasField",
    # "TranslationMotion",
    # "RotationalMotion",
    # "SegmentedRotationalMotion",
    # "SegmentedTranslationMotion",
    # "GaussianKspaceBiasField",
    # "GaussianNoise",
]

RECONSTRUCTORS = [
    # "zero-filled",
    # "conjugate-gradient",
    # "ram",
    # "dip",
    # "tv-pgd",
    # "wavelet-fista",
    # "tv-fista",
    # "tv-pdhg",
    # "unet",  # will trigger download of pretrained weights if not already present
    # *list(EXPLICIT_UNET_ALGORITHMS)
    "unet-fastmri",
    "unet-oasis-acceleration4",
    # "unet-oasis-acceleration8",
    # "unet-oasis-acceleration10",
]

reference_reconstructor = "conjugate-gradient"
reference_map_estimation = "espirit"

filenames = [
    "/home/melanie.dohmen/mri_recon/data/fastmri/multicoil_brain_test/file_brain_AXFLAIR_200_6002452.h5"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
result_path = "/home/melanie.dohmen/mri_recon/reports/experiments_fastmri_brain/"

os.makedirs(result_path, exist_ok=True)

distortion_times = {}
reconstruction_times = {}

for f_idx, filename in enumerate(filenames):
    with h5py.File(filename, "r") as hf:
        kspace_data = hf["kspace"][:]
        reconstruction_rss = hf["reconstruction_rss"][:]
        # hdr = hf["ismrmrd_header"][()]
        # print(hf.keys())

    x = torch.from_numpy(reconstruction_rss).unsqueeze(0).unsqueeze(0)
    y = torch.view_as_real(torch.from_numpy(kspace_data)).unsqueeze(0).moveaxis(-1, 1)
    # image shape: (1, channels, slices, H, W)
    print("image x.shape:", x.shape)
    # k-space shape: (1, channels, slices, coils, H, W)
    print("k-space y.shape:", y.shape)
    x = x.to(device)
    y = y.to(device)

    # select middle slice:
    x = x[:, :, x.shape[2] // 2, ...]
    y = y[:, :, y.shape[2] // 2, ...]

    # read size of dimensions:
    batch_size, channels, n_coils, ksp_w, ksp_h = y.shape

    for map_estimation in sensitivity_map_estimation_algorithm:
        print(f"Estimating coil maps with {map_estimation}...")
        if map_estimation == "espirit":
            estimate_coil_maps = dinv.datasets.MRISliceTransform(
                estimate_coil_maps=True,
                acs=15,  # Num. low frequency, fix to 15
            )
            _, _, params = estimate_coil_maps(target=x[0], kspace=y[0])

            coil_maps = params["coil_maps"]
            print("estimated coil maps shape: ", coil_maps.shape)

            coil_maps_result_path = os.path.join(
                result_path, f"brain_sample_{f_idx}_coil_maps_{map_estimation}"
            )
            os.makedirs(coil_maps_result_path, exist_ok=True)
            for c_idx in range(coil_maps.shape[0]):
                imwrite(
                    os.path.join(
                        coil_maps_result_path,
                        f"brain_sample_{f_idx}_{map_estimation}_map_{c_idx}.tiff",
                    ),
                    coil_maps[c_idx].abs().numpy(),
                )

        elif map_estimation == "unity":
            coil_maps = torch.ones((n_coils, ksp_w, ksp_h), dtype=torch.complex64, device=device)
            print("estimated coil maps shape: ", coil_maps.shape)
            coil_maps_result_path = os.path.join(
                result_path, f"brain_sample_{f_idx}_coil_maps_{map_estimation}"
            )
            os.makedirs(coil_maps_result_path, exist_ok=True)
            for c_idx in range(coil_maps.shape[0]):
                imwrite(
                    os.path.join(
                        coil_maps_result_path,
                        f"brain_sample_{f_idx}_{map_estimation}_map_{c_idx}.tiff",
                    ),
                    coil_maps[c_idx].abs().numpy(),
                )

        elif map_estimation == "birdcage":
            coil_maps = n_coils

        if map_estimation == reference_map_estimation:
            physics_clean = dinv.physics.MultiCoilMRI(
                img_size=(ksp_w, ksp_h),
                mask=None,
                coil_maps=coil_maps,
                device=device,
            )

            y_distorted = BaseDistortion()(y)
            print("base distortion does not change y:", torch.all(y_distorted == y))
            print("kspace distorted.shape: ", y_distorted.shape)

            x_recon_reference = choose_reconstructor(reference_reconstructor)(
                y_distorted, physics_clean
            )
            print("reconstructed image shape: ", x_recon_reference.shape)

            x_recon_reference_cropped = physics_clean.crop(x_recon_reference, shape=x.shape)
            print("cropped reconstructed image shape: ", x_recon_reference_cropped.shape)

            x_as_inversed_y = physics_clean.A_adjoint(y_distorted, rss=True)
            print("inversed y shape: ", x_as_inversed_y.shape)
            x_as_inversed_y_cropped = physics_clean.crop(x_as_inversed_y, shape=x.shape)
            print("cropped inversed y shape: ", x_as_inversed_y_cropped.shape)

            imwrite(
                os.path.join(
                    result_path,
                    f"brain_sample_{f_idx}_{map_estimation}_reconstructed_reference.tiff",
                ),
                x_recon_reference_cropped[0, 0].abs().numpy(),
            )
            imwrite(
                os.path.join(result_path, f"brain_sample_{f_idx}_{map_estimation}_inversed_y.tiff"),
                x_as_inversed_y_cropped[0, 0].abs().numpy(),
            )

        if map_estimation == "birdcage":
            coil_maps = physics_clean.coil_maps
            print("estimated coil maps shape: ", coil_maps.shape)
            coil_maps_result_path = os.path.join(
                result_path, f"brain_sample_{f_idx}_coil_maps_{map_estimation}"
            )
            os.makedirs(coil_maps_result_path, exist_ok=True)
            for c_idx in range(coil_maps.shape[0]):
                imwrite(
                    os.path.join(
                        coil_maps_result_path,
                        f"brain_sample_{f_idx}_{map_estimation}_map_{c_idx}.tiff",
                    ),
                    coil_maps[0, c_idx].abs().numpy(),
                )

        for distortion_name in DISTORTIONS:
            print("Distortion: ", distortion_name)

            start = datetime.now()

            distortion = choose_distortion(distortion_name)
            physics_distorted = DistortedKspaceMultiCoilMRI(
                distortion=distortion,
                img_size=(ksp_w, ksp_h),
                mask=None,
                coil_maps=coil_maps,
                device=device,
            )

            print("inserting k-space into distortion with shape: ", y.shape)
            y_distorted = distortion(y)

            x_distorted_as_inversed_y = physics_distorted.A_adjoint(y_distorted, rss=True)
            x_distorted_as_inversed_y_cropped = physics_distorted.crop(
                x_distorted_as_inversed_y, shape=x.shape
            )
            imwrite(
                os.path.join(
                    result_path,
                    f"brain_sample_{f_idx}_{map_estimation}_{distortion_name}_inversed_y.tiff",
                ),
                x_distorted_as_inversed_y_cropped[0, 0].abs().numpy(),
            )

            for reconstructor_name in RECONSTRUCTORS:
                print("Reconstructor: ", reconstructor_name)

                start_recon = datetime.now()

                if reconstructor_name in [
                    "unet-oasis-acceleration4",
                    "unet-oasis-acceleration8",
                    "unet-oasis-acceleration10",
                ]:
                    physics_distorted = DistortedOasisMeasurement(
                        distortion=distortion,
                        img_size=(ksp_w, ksp_h),
                        mask=None,
                        coil_maps=coil_maps,
                        device=device,
                    )
                    y_distorted_for_recon = kspace_to_image(
                        fastmri_measurement_to_oasis_kspace(y_distorted)
                    )
                else:
                    y_distorted_for_recon = y_distorted

                reconstructor = choose_reconstructor(
                    reconstructor_name,
                    img_size=y.shape[-2:],
                    device=device,
                    verbose=True,
                ).to(device)

                try:
                    x_distorted = reconstructor(y_distorted_for_recon, physics_distorted)

                    x_distorted_cropped = (
                        physics_distorted.crop(x_distorted, shape=x.shape).detach().cpu()
                    )

                    imwrite(
                        os.path.join(
                            result_path,
                            f"brain_sample_{f_idx}_{map_estimation}_{distortion_name}_{reconstructor_name}.tiff",
                        ),
                        physics_distorted.coil_maps[:, 0].abs().numpy(),
                    )
                    imwrite(
                        os.path.join(
                            result_path,
                            f"brain_sample_{f_idx}_{map_estimation}_{distortion_name}_{reconstructor_name}_reconstructed.tiff",
                        ),
                        x_distorted_cropped[0, 0].abs().numpy(),
                    )
                    imwrite(
                        os.path.join(
                            result_path,
                            f"brain_sample_{f_idx}_{map_estimation}_{distortion_name}_{reconstructor_name}_reconstructed.tiff",
                        ),
                        x_distorted_cropped[0, 0].abs().numpy(),
                    )
                except Exception as e:
                    print(f"Reconstruction with {reconstructor_name} failed due to error: {e}")
                    continue

                end_recon = datetime.now()
                if reconstructor_name not in reconstruction_times:
                    reconstruction_times[reconstructor_name] = [
                        (end_recon - start_recon).total_seconds()
                    ]
                else:
                    reconstruction_times[reconstructor_name].append(
                        (end_recon - start_recon).total_seconds()
                    )

            end = datetime.now()
            if distortion_name not in distortion_times:
                distortion_times[distortion_name] = [(end - start).total_seconds()]
            else:
                distortion_times[distortion_name].append((end - start).total_seconds())


distortion_times_df = pd.DataFrame(distortion_times, index=[sensitivity_map_estimation_algorithm])
distortion_times_df.to_csv(os.path.join(result_path, "distortion_times.csv"), index=False)
reconstruction_times_df = pd.DataFrame(reconstruction_times)
reconstruction_times_df.to_csv(os.path.join(result_path, "reconstruction_times.csv"), index=False)

print(distortion_times_df.mean())
print(reconstruction_times_df.mean())

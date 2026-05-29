import os
import sys
import h5py
from tifffile import imwrite
import torch
import deepinv as dinv

import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mri_recon.distortions import DistortedKspaceMultiCoilMRI, BaseDistortion
from mri_recon.reconstruction import choose_reconstructor
from mri_recon.utils import Grappa


sensitivity_map_estimation_algorithm = [
    # "espirit",
    "unity",
    # "birdcage",
]


DISTORTIONS = [
    # "PhaseEncodeGhosting",
    # "CartesianUndersamplingVariableDensityRandom",
    "CartesianUndersamplingUniformRandom",
    # "HannTaperLP",
    # "KaiserTaperLP",
    # "RadialHighPassEmphasis",
    # "IsotropicLP",
    # "OffCenterAnisotropicGaussianBiasField",
    # "TranslationMotion",
    # "RotationalMotion",
    # "SegmentedRotationalMotion",
    # "SegmentedTranslationMotion",
    "GaussianKspaceBiasField",
    # "GaussianNoise",
]

RECONSTRUCTORS = [
    "zero-filled",
    "conjugate-gradient",
    # "ram",
    # "dip",
    # "tv-pgd",
    # "wavelet-fista",
    # "tv-fista",
    # "tv-pdhg",
    # "unet",  # will trigger download of pretrained weights if not already present
    # *list(EXPLICIT_UNET_ALGORITHMS)
    #'unet-fastmri',
    #'unet-oasis-acceleration4',
    #'unet-oasis-acceleration8',
    #'unet-oasis-acceleration10',
]

reference_reconstructor = "conjugate-gradient"
reference_map_estimation = "unity"

filenames = [
    "/home/melanie.dohmen/mri_recon/data/fastmri/fastMRI_prostate_T2_IDS_001_020/file_prostate_AXT2_001.h5"
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
result_path = "/home/melanie.dohmen/mri_recon/reports/experiments_fastmri_prostateT2/"

os.makedirs(result_path, exist_ok=True)


def correct_kspace_data_with_calibration(
    y: torch.Tensor, calibration_data: torch.Tensor
) -> torch.Tensor:
    n_avg, n_slices, n_coils, ksp_w, ksp_h = kspace_data.shape

    # Calib_data shape: num_slices, num_coils, num_pe_cal
    grappa_weight_dict = {}
    grappa_weight_dict_2 = {}

    kspace_slice_regridded = kspace_data[0, 0, ...]
    grappa_obj = Grappa(
        np.transpose(kspace_slice_regridded, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1
    )

    kspace_slice_regridded_2 = kspace_data[1, 0, ...]
    grappa_obj_2 = Grappa(
        np.transpose(kspace_slice_regridded_2, (2, 0, 1)), kernel_size=(5, 5), coil_axis=1
    )

    # calculate GRAPPA weights
    for slice_num in range(n_slices):
        calibration_regridded = calibration_data[slice_num, ...]
        grappa_weight_dict[slice_num] = grappa_obj.compute_weights(
            np.transpose(calibration_regridded, (2, 0, 1))
        )
        grappa_weight_dict_2[slice_num] = grappa_obj_2.compute_weights(
            np.transpose(calibration_regridded, (2, 0, 1))
        )

    # apply GRAPPA weights
    kspace_post_grappa_all = np.zeros(shape=kspace_data.shape, dtype=complex)

    for average, grappa_obj, grappa_weight_dict in zip(
        [0, 1, 2],
        [grappa_obj, grappa_obj_2, grappa_obj],
        [grappa_weight_dict, grappa_weight_dict_2, grappa_weight_dict],
    ):
        for slice_num in range(n_slices):
            kspace_slice_regridded = kspace_data[average, slice_num, ...]
            kspace_post_grappa = grappa_obj.apply_weights(
                np.transpose(kspace_slice_regridded, (2, 0, 1)), grappa_weight_dict[slice_num]
            )
            kspace_post_grappa_all[average, slice_num, ...] = np.moveaxis(
                np.moveaxis(kspace_post_grappa, 0, 1), 1, 2
            )

    return kspace_post_grappa_all


for f_idx, filename in enumerate(filenames):
    with h5py.File(filename, "r") as hf:
        kspace_data = hf["kspace"][:]
        calibration_data = hf["calibration_data"][:]
        hdr = hf["ismrmrd_header"][()]
        reconstruction_rss = hf["reconstruction_rss"][:]
        atts = dict()
        atts["max"] = hf.attrs["max"]
        atts["norm"] = hf.attrs["norm"]
        atts["patient_id"] = hf.attrs["patient_id"]
        atts["acquisition"] = hf.attrs["acquisition"]

    n_avg, n_slices, n_coils, ksp_w, ksp_h = kspace_data.shape

    # correct k-space data with calibration data:

    kspace_data = correct_kspace_data_with_calibration(kspace_data, calibration_data)

    x = torch.from_numpy(reconstruction_rss).unsqueeze(0).unsqueeze(0)
    y = torch.view_as_real(torch.from_numpy(kspace_data)).unsqueeze(0).moveaxis(-1, 1)
    # image shape: (1, slices, H, W)
    print("image x.shape:", x.shape)
    # k-space shape: (1, slices, coils, H, W)
    print("k-space y.shape:", y.shape)
    x = x.to(device)
    y = y.to(device)

    # select middle slice:
    x = x[:, :, x.shape[2] // 2, ...]
    y = y[:, :, y.shape[2] // 2, ...]

    recon_1 = []

    for ave in range(n_avg):
        # estimate coil maps:
        for map_estimation in sensitivity_map_estimation_algorithm:
            if map_estimation == "espirit":
                estimate_coil_maps = dinv.datasets.MRISliceTransform(
                    estimate_coil_maps=True,
                    acs=15,  # Num. low frequency, fix to 15
                )
                _, _, params = estimate_coil_maps(target=x[0], kspace=y[0])

                coil_maps = params["coil_maps"]
                print("estimated coil maps shape: ", coil_maps.shape)
                imwrite(
                    os.path.join(
                        result_path, f"prostate_sample_{f_idx}_{map_estimation}_maps.tiff"
                    ),
                    coil_maps[:, 0].abs().numpy(),
                )

            elif map_estimation == "unity":
                coil_maps = torch.ones(
                    (n_coils, ksp_w, ksp_h), dtype=torch.complex64, device=device
                )
                print("estimated coil maps shape: ", coil_maps.shape)
                imwrite(
                    os.path.join(
                        result_path, f"prostate_sample_{f_idx}_{map_estimation}_maps.tiff"
                    ),
                    coil_maps[:, 0].abs().numpy(),
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

                recon_1.append(x_recon_reference_cropped[0, 0].abs().numpy())

                x_as_inversed_y = physics_clean.A_adjoint(y_distorted, rss=True)
                print("inversed y shape: ", x_as_inversed_y.shape)
                x_as_inversed_y_cropped = physics_clean.crop(x_as_inversed_y, shape=x.shape)
                print("cropped inversed y shape: ", x_as_inversed_y_cropped.shape)

                imwrite(
                    os.path.join(
                        result_path,
                        f"prostate_sample_{f_idx}_{map_estimation}_reconstructed_reference.tiff",
                    ),
                    x_recon_reference_cropped[0, 0].abs().numpy(),
                )
                imwrite(
                    os.path.join(
                        result_path, f"prostate_sample_{f_idx}_{map_estimation}_inversed_y.tiff"
                    ),
                    x_as_inversed_y_cropped[0, 0].abs().numpy(),
                )

            if map_estimation == "birdcage":
                coil_maps = physics_clean.coil_maps
                print("estimated coil maps shape: ", coil_maps.shape)
                imwrite(
                    os.path.join(
                        result_path, f"prostate_sample_{f_idx}_{map_estimation}_maps.tiff"
                    ),
                    coil_maps[:, 0].abs().numpy(),
                )

            for distortion_name in DISTORTIONS:
                physics_distorted = DistortedKspaceMultiCoilMRI(
                    distortion=BaseDistortion(),
                    img_size=(ksp_w, ksp_h),
                    mask=None,
                    coil_maps=coil_maps,
                    device=device,
                )

                x_distorted_as_inversed_y = physics_distorted.A_adjoint(y_distorted, rss=True)
                x_distorted_as_inversed_y_cropped = physics_distorted.crop(
                    x_distorted_as_inversed_y, shape=x.shape
                )
                imwrite(
                    os.path.join(
                        result_path, f"prostate_sample_{f_idx}_{distortion_name}_inversed_y.tiff"
                    ),
                    x_distorted_as_inversed_y_cropped[0, 0].abs().numpy(),
                )

                for reconstructor_name in RECONSTRUCTORS:
                    reconstructor = choose_reconstructor(reconstructor_name)

                    x_distorted = reconstructor(y_distorted, physics_distorted)

                    x_distorted_cropped = physics_distorted.crop(x_distorted, shape=x.shape)

                    imwrite(
                        os.path.join(
                            result_path,
                            f"prostate_sample_{f_idx}_{distortion_name}_{reconstructor_name}.tiff",
                        ),
                        physics_distorted.coil_maps[:, 0].abs().numpy(),
                    )
                    imwrite(
                        os.path.join(
                            result_path,
                            f"prostate_sample_{f_idx}_{distortion_name}_{reconstructor_name}_reconstructed.tiff",
                        ),
                        x_distorted_cropped[0, 0].abs().numpy(),
                    )

    recon_1_mean = np.mean(recon_1, axis=0)
    imwrite(
        os.path.join(
            result_path,
            f"prostate_sample_{f_idx}_{reference_map_estimation}_reconstructed_reference_mean.tiff",
        ),
        recon_1_mean,
    )

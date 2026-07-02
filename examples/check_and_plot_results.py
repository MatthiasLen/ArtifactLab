import glob
import os

import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt

result_folder = "/home/melanie.dohmen/ArtifactLab/reports/experiments_run1_20260616"

result_file_names = glob.glob(os.path.join(result_folder, "*.tiff"))


distortion_names = [
    "BaseDistortion",
    "CartesianUndersamplingVariableDensity",
    "CartesianUndersamplingUniformRandom",
    "CartesianUndersamplingUniformRandomZeroACS",
    "CartesianUndersamplingEquispaced",
    "CartesianUndersamplingEquispacedZeroACS",
    "PartialFourier",
    "PhaseEncodeGhosting",
    "SegmentedTranslationMotion",
    "SegmentedRotationalMotion",
    "TranslationMotion",
    "RotationalMotion",
    "OffCenterAnisotropicGaussianBiasField",
    "GaussianBiasField",
    "AnisotropicLP",
    "HannTaperLP",
    "KaiserTaperLP",
    "GaussianNoise",
    "IsotropicLP",
    "RadialHighPassEmphasis",
]

reconstruction_names = [
    "zero-filled",
    "conjugate-gradient",
    "ram",
    "dip",
    "tv-pgd",
    "wavelet-fista",
    "tv-fista",
    "tv-pdhg",
    "unet-fastmri",
    "unet-oasis-acceleration4",
    "unet-oasis-acceleration8",
    "unet-oasis-acceleration10",
]

samples = {
    "fastmri_knee": ["1000000"],
    "oasis": ["OAS1-0088-MR1"],
    "fastmri_brain": ["AXFLAIR-200-6002452"],
    "cmrxrecon": ["P001-cine-lax"],
    "fastmri_prostate": ["AXT2-007"],
}


#####
# Check results
#####
check_results = True
if check_results:
    missing_files = []
    for dataset_name, sample_names in samples.items():
        for sample_name in sample_names:
            for distortion_name in distortion_names:
                for reconstruction_name in reconstruction_names:
                    result_file_names_corr = glob.glob(
                        os.path.join(
                            result_folder,
                            f"image_{dataset_name}_{sample_name}_{distortion_name}*_{reconstruction_name}_corrected.tiff",
                        )
                    )
                    if len(result_file_names_corr) == 0:
                        missing_files.append(
                            f"image_{dataset_name}_{sample_name}_{distortion_name}_{reconstruction_name}_corrected.tiff"
                        )
                # else:
                #    result_file_names.remove(result_file_names_corr[0])
                result_file_names_uncorr = glob.glob(
                    os.path.join(
                        result_folder,
                        f"image_{dataset_name}_{sample_name}_{distortion_name}*_{reconstruction_name}_uncorrected.tiff",
                    )
                )
                if len(result_file_names_uncorr) == 0:
                    missing_files.append(
                        f"image_{dataset_name}_{sample_name}_{distortion_name}_{reconstruction_name}_uncorrected.tiff"
                    )
                # else:
                #    result_file_names.remove(result_file_names_uncorr[0])
        reference_file_name = os.path.join(
            result_folder, f"image_{dataset_name}_{sample_name}_reference.tiff"
        )
        if not os.path.exists(reference_file_name):
            missing_files.append(reference_file_name)
        kspace_ref_filename = os.path.join(
            result_folder, f"kspace_{dataset_name}_{sample_name}_kspace_reference.tiff"
        )
        if not os.path.exists(kspace_ref_filename):
            missing_files.append(kspace_ref_filename)
            distorted_kspace_file_name = os.path.join(
                result_folder,
                f"kspace_{dataset_name}_{sample_name}_{distortion_name}_distorted.tiff",
            )
            if not os.path.exists(distorted_kspace_file_name):
                missing_files.append(distorted_kspace_file_name)

    # found for prostate:

    found_prostate_files = sorted(glob.glob(os.path.join(result_folder, "*prostate*.tiff")))
    print(f"Found prostate files: {len(found_prostate_files)}")

    for file_name in found_prostate_files:
        print(f"Found file: {file_name}")

    found_cmrxrecon_files = sorted(glob.glob(os.path.join(result_folder, "*cmrxrecon*.tiff")))
    print(f"Found cmrxrecon files: {len(found_cmrxrecon_files)}")

    for file_name in found_cmrxrecon_files:
        print(f"Found file: {file_name}")

    for file_name in missing_files:
        print("missing: ", file_name)


######
# Create Plots for certain groups
######


for dataset_name, sample_names in samples.items():
    for sample_name in sample_names:
        print("looping over ", dataset_name, " ", sample_name)

        for part in range(2):
            print("part ", part + 1)

            # always use CG in first column:
            reconstruction_names = [
                recon for recon in reconstruction_names if recon != "conjugate-gradient"
            ]
            reconstruction_names_part = reconstruction_names[
                part * len(reconstruction_names) // 2 : (part + 1) * len(reconstruction_names) // 2
            ]
            reconstruction_names_part = ["conjugate-gradient"] + reconstruction_names_part

            nr_rows = len(distortion_names)
            nr_cols = len(reconstruction_names_part) * 2

            fig, axes = plt.subplots(
                int(nr_rows), int(nr_cols), figsize=(3 * nr_cols, 3 * nr_rows), squeeze=False
            )
            fig.suptitle(f"{dataset_name} {sample_name} - part {part + 1}")

            # first row contains reference, BaseDistortion, without corrections

            axes[0, 0].set_title(f"{dataset_name} reference")
            reference_file_name = os.path.join(
                result_folder, f"image_{dataset_name}_{sample_name}_reference.tiff"
            )
            if os.path.exists(reference_file_name):
                img = imread(reference_file_name).squeeze()
                if len(img.shape) == 3:
                    print(f"Warning: image has 3 dimensions: {img.shape}")
                    print(reference_file_name)
                    img = img[0, ...]
                axes[0, 0].imshow(img, cmap="gray")
                axes[0, 0].set_title(f"{dataset_name} reference")
                axes[0, 0].xaxis.set_visible(False)
                axes[0, 0].set_yticks([])
                axes[0, 0].set_ylabel("BaseDistortion", fontsize=12)

            else:
                axes[0, 0].text(
                    0.5,
                    0.5,
                    "MISSING",
                    transform=axes[0, 0].transAxes,
                    fontsize=12,
                    color="red",
                    ha="center",
                )
                axes[0, 0].axis("off")

            print("\tBaseDistortion")
            for r_idx, reconstruction in enumerate(reconstruction_names_part):
                print("\t\t", reconstruction)
                if r_idx != 0:
                    result_file_names_uncorr = glob.glob(
                        os.path.join(
                            result_folder,
                            f"image_{dataset_name}_{sample_name}_BaseDistortion*_{reconstruction}_uncorrected.tiff",
                        )
                    )
                    if len(result_file_names_uncorr) > 0:
                        img = imread(result_file_names_uncorr[0]).squeeze()
                        if len(img.shape) == 3:
                            print(f"Warning: image has 3 dimensions: {img.shape}")
                            print(result_file_names_uncorr[0])
                            img = img[0, ...]
                        axes[0, 2 * r_idx].imshow(img, cmap="gray")
                        axes[0, 2 * r_idx].set_title(f"{reconstruction} (u)")
                        axes[0, 2 * r_idx].axis("off")
                    else:
                        axes[0, 2 * r_idx].set_title(f"{reconstruction} (u)")
                        axes[0, 2 * r_idx].text(
                            0.5,
                            0.5,
                            "MISSING",
                            transform=axes[0, 2 * r_idx].transAxes,
                            fontsize=12,
                            color="red",
                            ha="center",
                        )
                        axes[0, 2 * r_idx].axis("off")

                else:
                    axes[0, r_idx].axis("on")
                    axes[0, r_idx].xaxis.set_visible(False)
                    axes[0, r_idx].set_yticks([])
                    axes[0, r_idx].set_ylabel("BaseDistortion", fontsize=12)

                result_file_names_corr = glob.glob(
                    os.path.join(
                        result_folder,
                        f"image_{dataset_name}_{sample_name}_BaseDistortion*_{reconstruction}_corrected.tiff",
                    )
                )
                if len(result_file_names_corr) > 0:
                    img = imread(result_file_names_corr[0]).squeeze()
                    if len(img.shape) == 3:
                        print(f"Warning: image has 3 dimensions: {img.shape}")
                        print(result_file_names_corr[0])
                        img = img[0, ...]
                    axes[0, 2 * r_idx + 1].imshow(img, cmap="gray")
                    axes[0, 2 * r_idx + 1].set_title(f"{reconstruction} (c)")
                    axes[0, 2 * r_idx + 1].axis("off")
                else:
                    axes[0, 2 * r_idx + 1].set_title(f"{reconstruction} (c)")
                    axes[0, 2 * r_idx + 1].text(
                        0.5,
                        0.5,
                        "MISSING",
                        transform=axes[0, 2 * r_idx + 1].transAxes,
                        fontsize=12,
                        color="red",
                        ha="center",
                    )
                    axes[0, 2 * r_idx + 1].axis("off")

            # reconstruction methods in columns, distortions in rows
            for d_idx, distortion in enumerate(
                [dist for dist in distortion_names if dist != "BaseDistortion"]
            ):
                print("\t", distortion)
                for r_idx, reconstruction in enumerate(reconstruction_names_part):
                    print("\t\t", reconstruction)
                    result_file_names_corr = glob.glob(
                        os.path.join(
                            result_folder,
                            f"image_{dataset_name}_{sample_name}_{distortion}*_{reconstruction}_corrected.tiff",
                        )
                    )
                    if len(result_file_names_corr) > 0:
                        img = imread(result_file_names_corr[0]).squeeze()
                        if len(img.shape) == 3:
                            print("Warning: image has 3 dimensions: [img.shape]")
                            print(result_file_names_corr[0])
                            img = img[0, ...]
                        axes[d_idx + 1, 2 * r_idx + 1].imshow(img, cmap="gray")
                        axes[d_idx + 1, 2 * r_idx + 1].set_title(f"{reconstruction} (c)")
                        axes[d_idx + 1, 2 * r_idx + 1].axis("off")
                    else:
                        axes[d_idx + 1, 2 * r_idx + 1].set_title(f"{reconstruction} (c)")
                        axes[d_idx + 1, 2 * r_idx + 1].text(
                            0.5,
                            0.5,
                            "MISSING",
                            transform=axes[d_idx + 1, 2 * r_idx].transAxes,
                            fontsize=12,
                            color="red",
                            ha="center",
                        )
                        axes[d_idx + 1, 2 * r_idx + 1].axis("off")

                    result_file_names_uncorr = glob.glob(
                        os.path.join(
                            result_folder,
                            f"image_{dataset_name}_{sample_name}_{distortion}*_{reconstruction}_uncorrected.tiff",
                        )
                    )
                    if len(result_file_names_uncorr) > 0:
                        img_u = imread(result_file_names_uncorr[0]).squeeze()
                        if len(result_file_names_corr) > 0:
                            if (img == img_u).all():
                                print(
                                    f"Warning: corrected and uncorrected images are the same for {dataset_name}, {distortion}, {reconstruction}"
                                )
                        if len(img_u.shape) == 3:
                            print(f"Warning: image has 3 dimensions: {img_u.shape}")
                            print(result_file_names_uncorr[0])
                            img_u = img_u[0, ...]
                        axes[d_idx + 1, 2 * r_idx].imshow(img_u, cmap="gray")
                        axes[d_idx + 1, 2 * r_idx].set_title(f"{reconstruction} (u)")
                        axes[d_idx + 1, 2 * r_idx].axis("off")
                    else:
                        axes[d_idx + 1, 2 * r_idx].set_title(f"{reconstruction} (u)")
                        axes[d_idx + 1, 2 * r_idx].text(
                            0.5,
                            0.5,
                            "MISSING",
                            transform=axes[d_idx + 1, 2 * r_idx + 1].transAxes,
                            fontsize=12,
                            color="red",
                            ha="center",
                        )
                        axes[d_idx + 1, 2 * r_idx].axis("off")

                    if r_idx == 0:
                        axes[d_idx + 1, 0].axis("on")
                        axes[d_idx + 1, 0].xaxis.set_visible(False)
                        axes[d_idx + 1, 0].set_yticks([])
                        if len(distortion) > 30:
                            # find the latest capital letter between second and 30th character and split there
                            for c in distortion[1:30]:
                                if c.isupper():
                                    cap_letter_idx = distortion[1:30].find(c)
                            if cap_letter_idx != -1:
                                str_distortion = (
                                    distortion[: cap_letter_idx + 1]
                                    + "\n"
                                    + distortion[cap_letter_idx + 1 :]
                                )
                            else:
                                str_distortion = distortion[:30] + "\n" + distortion[30:50]
                        else:
                            str_distortion = distortion
                        axes[d_idx + 1, 0].set_ylabel(str_distortion, fontsize=12)

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    "/home/melanie.dohmen/ArtifactLab/reports",
                    f"summary_20260616_{dataset_name}_{sample_name}_part_{part + 1}.png",
                )
            )


######
# Create Plots for each distortion and sample
######

for dataset_name, sample_names in samples.items():
    for sample_name in sample_names:
        print("looping over ", dataset_name, " ", sample_name)

        for d_idx, distortion in enumerate(
            [dist for dist in distortion_names if dist != "BaseDistortion"]
        ):
            print("\t", distortion)

            results_for_distortion = glob.glob(
                os.path.join(
                    result_folder, f"image_{dataset_name}_{sample_name}_{distortion}*.tiff"
                )
            )
            nr_rows = np.ceil(np.sqrt(len(results_for_distortion)))
            nr_cols = np.ceil(len(results_for_distortion) / nr_rows)

            fig, axes = plt.subplots(
                int(nr_rows), int(nr_cols), figsize=(3 * nr_cols, 3 * nr_rows), squeeze=False
            )
            fig.suptitle(f"{dataset_name} {sample_name}")

            # sort results:
            results_for_distortion_sorted = sorted(
                results_for_distortion, key=lambda x: (x.split("_")[-2], x.split("_")[-1])
            )

            # set BaseDistortion and CG first:
            results_for_distortion_sorted = sorted(
                results_for_distortion_sorted,
                key=lambda x: (
                    x.split("_")[-2] != "BaseDistortion",
                    x.split("_")[-2] != "conjugate-gradient",
                ),
            )

            for r_idx, result_file_name in enumerate(results_for_distortion_sorted):
                # split filename to get reconstruction name and corrected/uncorrected
                reconstruction = result_file_name.split("_")[-2]
                corrected = result_file_name.split("_")[-1].split(".")[0]

                img = imread(result_file_name).squeeze()
                if len(img.shape) == 3:
                    print(f"Warning: image has 3 dimensions: {img.shape}")
                    print(result_file_name)
                    img = img[0, ...]
                axes[r_idx // nr_cols, r_idx % nr_cols].imshow(img, cmap="gray")
                axes[r_idx // nr_cols, r_idx % nr_cols].set_title(f"{reconstruction} (c)")
                axes[r_idx // nr_cols, r_idx % nr_cols].axis("off")

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    "/home/melanie.dohmen/ArtifactLab/reports",
                    f"summary_20260616_{dataset_name}_{sample_name}_{distortion}.png",
                )
            )

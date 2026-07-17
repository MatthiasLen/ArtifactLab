import glob
import os
import sys

import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from mri_recon.utils.plot import get_metadata_from_filename

# result_folder = "/home/melanie.dohmen/ArtifactLab/reports/experiments_run1"
result_folder = "/home/melanie.dohmen/ArtifactLab/reports/test_params_GaussianNoise"
#result_folder = "/home/melanie.dohmen/ArtifactLab/reports/test_params_RadialHighPassEmphasis"
#result_folder = "/home/melanie.dohmen/ArtifactLab/reports/experiments_brain_slice0"
# result_file_names = glob.glob(os.path.join(result_folder, "*.tiff"))


distortion_names = [
    # "BaseDistortion",
    # "CartesianUndersamplingVariableDensity",
    # "CartesianUndersamplingUniformRandom",
    # "CartesianUndersamplingUniformRandomZeroACS",
    # "CartesianUndersamplingEquispaced",
    # "CartesianUndersamplingEquispacedZeroACS",
    # "PartialFourier",
    # "PhaseEncodeGhosting",
    # "SegmentedTranslationMotion",
    # "SegmentedRotationalMotion",
    # "TranslationMotion",
    # "RotationalMotion",
    #"OffCenterAnisotropicGaussianBiasField",
    #"GaussianBiasField",
    # "AnisotropicLP",
    # "HannTaperLP",
    # "KaiserTaperLP",
    "GaussianNoise",
    # "IsotropicLP",
    #"RadialHighPassEmphasis",
    #"ReduceResolution",
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
datasets = [
    #"fastmri_knee",
    #"oasis",
    "fastmri_brain",
    #"cmrxrecon",
    #"fastmri_prostate",
]

samples = {
    #"fastmri_knee": ["1000000", "1000007", "1000017"],
    #"oasis": ["OAS1-0088-MR1"],
    "fastmri_brain": [ "AXFLAIR-200-6002467"], #, "AXFLAIR-200-6002452","AXFLAIR-200-6002467", "AXFLAIR-200-6002512"],
    #"cmrxrecon": ["P001-cine-lax"],
    #"fastmri_prostate": ["AXT2-013", "AXT2-007"],
}


#####
# Check results
#####
check_results = False
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
# Create Plots for each sample
######

create_sample_summary = False

if create_sample_summary:
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
                    part * len(reconstruction_names) // 2 : (part + 1)
                    * len(reconstruction_names)
                    // 2
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
                        result_folder,
                        f"summary_20260616_{dataset_name}_{sample_name}_part_{part + 1}.png",
                    )
                )


######
# Create Plots for each distortion and sample
######
def find_all(string, substring):
    indices = []
    start = 0
    while True:
        start = string.find(substring, start)
        if start == -1:
            break
        indices.append(start)
        start += len(substring)  # Move past the last found substring
    return indices


create_distortion_summary = True

if create_distortion_summary:
    for dataset_name, sample_names in samples.items():
        for sample_name in sample_names:
            print("looping over ", dataset_name, " ", sample_name)

            for d_idx, distortion in enumerate(
                [dist for dist in distortion_names if dist != "BaseDistortion"]
            ):
                results_for_distortion = glob.glob(
                    os.path.join(
                        result_folder, f"image_{dataset_name}_{sample_name}_{distortion}*.tiff"
                    )
                )

                # find reference:
                reference = glob.glob(
                    os.path.join(
                        result_folder, f"image_{dataset_name}_{sample_name}_reference.tiff"
                    )
                )

                nr_rows = int(np.ceil(np.sqrt(len(results_for_distortion)))) + len(reference)
                if nr_rows > 1:
                    print(
                        f"For {distortion} found {len(results_for_distortion)} results and {len(reference)} reference images"
                    )
                    nr_cols = int(np.ceil((len(results_for_distortion) + len(reference)) / nr_rows))

                    fig, axes = plt.subplots(
                        int(nr_rows),
                        int(nr_cols),
                        figsize=(3 * nr_cols, 3 * nr_rows),
                        squeeze=False,
                    )

                    # sort results:
                    results_for_distortion_sorted = reference + sorted(
                        results_for_distortion, key=lambda x: (x.split("_")[-2], x.split("_")[-1])
                    )

                    for r_idx, result_file_name in enumerate(results_for_distortion_sorted):
                        # split filename to get reconstruction name and corrected/uncorrected and parameters
                        # to add details to each result
                        metadata = get_metadata_from_filename(result_file_name)
                        img = imread(result_file_name).squeeze()
                        
                        if len(img.shape) == 3:
                            print(f"Warning: image has 3 dimensions: {img.shape}")
                            print(result_file_name)
                            img = img[0, ...]
                        if len(img.shape) != 2:
                            print(f"Warning: image has {len(img.shape)} dimensions: {img.shape}")
                            axes[r_idx // nr_cols, r_idx % nr_cols].text(
                                0.5,
                                0.5,
                                "INVALID SHAPE: " + str(img.shape),
                                transform=axes[r_idx // nr_cols, r_idx % nr_cols].transAxes,
                                fontsize=12,
                                color="red",
                                ha="center",
                            )
                            if r_idx < len(reference):
                                axes[r_idx // nr_cols, r_idx % nr_cols].set_title(
                                    f"{distortion}\n{dataset_name}{sample_name}\n(reference)"
                                )
                            else:
                                axes[r_idx // nr_cols, r_idx % nr_cols].set_title(
                                    f"{metadata['reconstruction_method']} (+{metadata['add']})\n{list(metadata['parameters'].values())}"
                                )
                        else:
                            min_value = np.min(img)
                            max_value = np.max(img)
                            mean_value = np.mean(img)
                           
                            axes[r_idx // nr_cols, r_idx % nr_cols].imshow(img, cmap="gray")
                            if r_idx < len(reference):
                                ref_mean = np.mean(img)
                                ref_std = np.std(img)
                                value_str = f"[{(min_value-ref_mean)/ref_std:.2f}-{(max_value-ref_mean)/ref_std:.2f}]({ref_mean:.2f}/{ref_std:.2f})"
                                axes[r_idx // nr_cols, r_idx % nr_cols].set_title(
                                    f"{metadata['dataset']} {metadata['sample_name']}\n(reference)\n{value_str}"
                                )
                            else:
                                # if references are available, normalize to reference mean and std
                                if len(reference) > 0:
                                    value_str = f"[{(min_value-ref_mean)/ref_std:.2f}-{(max_value-ref_mean)/ref_std:.2f}]({mean_value:.2f})"
                                else:
                                    value_str = f"[{min_value:.2f}-{max_value:.2f}]({mean_value:.2f})"
                                axes[r_idx // nr_cols, r_idx % nr_cols].set_title(
                                    f"{metadata['reconstruction_method']} (+{metadata['add']})\n{list(metadata['parameters'].values())}\n{value_str}"
                                )
                        axes[r_idx // nr_cols, r_idx % nr_cols].axis("off")

                    # remove axis for empty subplots
                    for r_idx in range(len(results_for_distortion_sorted), nr_rows * nr_cols):
                        axes[r_idx // nr_cols, r_idx % nr_cols].axis("off")

                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            result_folder,
                            f"summary_{dataset_name}_{sample_name}_{distortion}.png",
                        )
                    )

                else:
                    print("No results for ", distortion)


overview_cases = False

if overview_cases:
    print("Creating overview of all reference images for each dataset")
    for dataset_name in datasets:
        reference_images = sorted(glob.glob(os.path.join(result_folder, f"image_{dataset_name}_*_reference.tiff")))

        # Split reference images into partitions of < 101 samples
        partitions = [reference_images[i:i+100] for i in range(0, len(reference_images), 100)]
        
        for part_idx, partition_images in enumerate(partitions):
            nr_rows = int(np.ceil(np.sqrt(len(partition_images))))        
            nr_cols = int(np.ceil(len(partition_images) / nr_rows))

            print("Processing partition ", part_idx + 1, "/", len(partitions), " for dataset ", dataset_name)
            fig, axes = plt.subplots(
                int(nr_rows),
                int(nr_cols),
                figsize=(3 * nr_cols, 3 * nr_rows),
                squeeze=False,
            )

            for r_idx, result_file_name in enumerate(partition_images):
                # split filename to get reconstruction name and corrected/uncorrected and parameters
                # to add details to each result
                metadata = get_metadata_from_filename(result_file_name)
                img = imread(result_file_name).squeeze()
                
                if len(img.shape) == 3:
                    print(f"Warning: image has 3 dimensions: {img.shape}")
                    print(result_file_name)
                    img = img[0, ...]
                if len(img.shape) != 2:
                    print(f"Warning: image has {len(img.shape)} dimensions: {img.shape}")
                    axes[r_idx // nr_cols, r_idx % nr_cols].text(
                        0.5,
                        0.5,
                        "INVALID SHAPE: " + str(img.shape),
                        transform=axes[r_idx // nr_cols, r_idx % nr_cols].transAxes,
                        fontsize=12,
                        color="red",
                        ha="center",
                    )
                    axes[r_idx // nr_cols, r_idx % nr_cols].set_title(
                        f"{metadata['dataset']}\n{metadata['sample_name']}"
                    )
                else:
                
                    axes[r_idx // nr_cols, r_idx % nr_cols].imshow(img, cmap="gray")
                    axes[r_idx // nr_cols, r_idx % nr_cols].set_title(
                        f"{metadata['dataset']}\n{metadata['sample_name']}"
                    )

                axes[r_idx // nr_cols, r_idx % nr_cols].axis("off")

            # Remove axes for empty subplots
            for r_idx in range(len(partition_images), nr_rows * nr_cols):
                axes[r_idx // nr_cols, r_idx % nr_cols].axis("off")

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    result_folder,
                    f"overview_samples_{dataset_name}_part{part_idx+1}.png",
                )
            )
            plt.close(fig)

                
import os
import pandas as pd

from tifffile import imread, imwrite

# Structure for exporting examples:


# |-- property_01_pixel_resolution/
# | |-- example_001
# | |-- example_002/
# | | |-- reference.npy
# | | |-- reference.png
# | | |-- degraded_1.npy
# | | |-- degraded_1.png
# | | |-- degraded_2.npy
# | | |-- degraded_2.png
# | | |-- degraded_3.npy
# | | |-- degraded_3.png
# | | `-- metadata.json <- Information about image sources and degration levels applied
# | |-- example_003
# | |-- ...
# | `-- example_014
# |-- property_02_texture_structure
# |-- property_03_image_contrast
# |-- ...
# |-- property_07_morphological_correctness `-- metadata.csv <- Spreadsheet with metadata from all examples from all properties
# ├── property_01_pixel_resolution
# │ ├── example_001
# │ ├── example_002
# │ │ ├── reference.npy
# │ │ ├── reference.png
# │ │ ├── degraded_1.npy
# │ │ ├── degraded_1.png
# │ │ ├── degraded_2.npy
# │ │ ├── degraded_2.png
# │ │ ├── degraded_3.npy
# │ │ ├── degraded_3.png
# │ │ └── metadata.json <- Information about image sources and degration levels applied
# │ ├── example_003
# │ ├── ...
# │ └── example_014
# ├── property_02_texture_structure
# ├── property_03_image_contrast
# ├── ...
# ├── property_07_morphological_correctness
# └── metadata.csv <- Spreadsheet with metadata from all examples from all properties
# .


# selected examples for each property:
properties = {
    "property_01_pixel_resolution": {
        "example_001": {
            "reference": "image_fastmri_knee_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
        "example_002": {
            "reference": "image_fastmri_brain_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
    },
    "property_02_sharpness": {
        "example_001": {
            "reference": "image_fastmri_knee_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
        "example_002": {
            "reference": "image_fastmri_brain_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
    },
    "property_03_intensity_uniformity": {
        "example_001": {
            "reference": "image_fastmri_knee_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
        "example_002": {
            "reference": "image_fastmri_brain_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
    },
    "property_04_noise_level": {
        "example_001": {
            "reference": "image_fastmri_knee_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
        "example_002": {
            "reference": "image_fastmri_brain_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
    },
    "property_05_roi_homogeneity": {
        "example_001": {
            "reference": "image_fastmri_knee_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
        "example_002": {
            "reference": "image_fastmri_brain_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
    },
    "property_06_local_signal_preservation": {
        "example_001": {
            "reference": "image_fastmri_knee_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
        "example_002": {
            "reference": "image_fastmri_brain_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
    },
    "property_07_edges": {
        "example_001": {
            "reference": "image_fastmri_knee_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
        "example_002": {
            "reference": "image_fastmri_brain_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
    },
    "property_08_contrast_preservation_of_anatomical_structures": {
        "example_001": {
            "reference": "image_fastmri_knee_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_knee_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
        "example_002": {
            "reference": "image_fastmri_brain_sample_0_reference.tiff",
            "degraded": [
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_uncorrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_zero-filled_corrected.tiff",
                "image_fastmri_brain_sample_0_PartialFouriers=high_tv-pgd_corrected.tiff",
            ],
        },
    },
}

SOURCE_INFO = {
    "fastmri_knee": {
        "source_name": "fastMRI Knee",
        "source_url": "https://fastmri.med.nyu.edu/",
        "source_license": "internal research and educational purposes only",
    },
    "fastmri_brain": {
        "source_name": "fastMRI Brain",
        "source_url": "https://fastmri.med.nyu.edu/",
        "source_license": "internal research and educational purposes only",
    },
    "fastmri_prostate": {
        "source_name": "fastMRI Prostate",
        "source_url": "https://fastmri.med.nyu.edu/",
        "source_license": "internal research and educational purposes only",
    },
    "oasis": {
        "source_name": "OASIS",
        "source_url": "https://sites.wustl.edu/oasisbrains/",
        "source_license": "academic research purposes only",
    },
    "cmrxrecon": {
        "source_name": "CMRxRecon",
        "source_url": "https://www.cmrxrecon.org/",
        "source_license": "CC BY-NC-SA 4.0",
    },
}


def get_metadata(example_path: str) -> dict:
    # remove parent directories from filename
    example_path = os.path.basename(example_path)
    example_path = example_path.replace(".tiff", "")
    if "_N4" in example_path:
        example_path = example_path.replace("_N4", "")
    if "uncorrected" in example_path:
        correction = "uncorrected"
        example_path = example_path.replace("_uncorrected", "")
    elif "corrected" in example_path:
        correction = "corrected"
        example_path = example_path.replace("_corrected", "")
    else:
        correction = "unknown"
    # if no sample name is given, remove extra underscore
    if "sample_0_" in example_path:
        example_path = example_path.replace("sample_0_", "sample0_")

    filename_parts = example_path.split("_")
    # [image_or_kspace, dataset_part1, (dataset_part2,) sample_name, distortion_or_reference, (reconstruction)]
    if "reference" in filename_parts:
        if len(filename_parts) == 4:
            return {
                **SOURCE_INFO[filename_parts[1]],  # dataset_part1
                "sample_name": filename_parts[2],  # sample_name
                "distortion_type": "reference",
                "correction": correction,
                "reconstruction_method": "reference",
            }

        elif len(filename_parts) == 5:
            return {
                **SOURCE_INFO[
                    f"{filename_parts[1]}_{filename_parts[2]}"
                ],  # dataset_part1_dataset_part2
                "sample_name": filename_parts[3],  # sample_name
                "distortion_type": "reference",
                "correction": correction,
                "reconstruction_method": "reference",
            }
        else:
            print(f"Warning: Unexpected filename format for reference example path: {example_path}")
    elif len(filename_parts) == 5:
        return {
            **SOURCE_INFO[filename_parts[1]],  # dataset_part1
            "sample_name": filename_parts[2],  # sample_name
            "distortion_type": filename_parts[3].split("=")[0],  # distortion without parameters
            "correction": correction,
            "reconstruction_method": filename_parts[4],  # reconstruction
        }
    elif len(filename_parts) == 6:
        return {
            **SOURCE_INFO[
                f"{filename_parts[1]}_{filename_parts[2]}"
            ],  # dataset_part1_dataset_part2
            "sample_name": filename_parts[3],  # sample_name
            "distortion_type": filename_parts[4].split("=")[0],  # distortion without parameters
            "correction": correction,
            "reconstruction_method": filename_parts[5],  # reconstruction
        }
    else:
        print(f"Warning: Unexpected filename format for example path: {example_path}")

    return {
        "source": "unknown",
        "sample_name": "unknown",
        "distortion_type": "unknown",
        "correction": correction,
        "reconstruction_method": "unknown",
    }


result_path = "/home/melanie.dohmen/ArtifactLab/reports/experiments_run1_20260616/"
export_path = "/home/melanie.dohmen/ArtifactLab/reports/exported_examples/"
os.makedirs(result_path, exist_ok=True)


metadata_df_list = []

for property_name, property_examples in properties.items():
    property_path = os.path.join(export_path, property_name)
    os.makedirs(property_path, exist_ok=True)

    for example_name, example_data in property_examples.items():
        example_path = os.path.join(property_path, example_name)
        os.makedirs(example_path, exist_ok=True)

        # Save reference image
        old_reference_image_path = os.path.join(result_path, example_data["reference"])
        new_reference_image_path = os.path.join(example_path, "reference.tiff")

        imwrite(new_reference_image_path, imread(old_reference_image_path))
        # Code to save the reference image using example_data["reference"]

        # Save degraded images
        for i, degraded_image in enumerate(example_data["degraded"]):
            old_degraded_image_path = os.path.join(result_path, degraded_image)
            new_degraded_image_path = os.path.join(example_path, f"degraded_{i + 1}.tiff")
            imwrite(new_degraded_image_path, imread(old_degraded_image_path))

            metadata = {
                "property": property_name,
                "example": example_name,
                "degraded_image_index": i + 1,
                "relative_path": os.path.relpath(new_degraded_image_path, result_path),
                **get_metadata(
                    old_degraded_image_path
                ),  # Function to extract metadata info from filename
            }

            metadata_df_list.append(metadata)

metadata_df = pd.DataFrame(metadata_df_list)
metadata_csv_path = os.path.join(export_path, "metadata.csv")
metadata_df.to_csv(metadata_csv_path, index=False)

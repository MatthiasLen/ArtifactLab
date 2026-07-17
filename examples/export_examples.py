import glob
import os
import pandas as pd
import sys

from ast import literal_eval
from tifffile import imread, imwrite
from matplotlib import pyplot as plt

from run_all import run_all
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from mri_recon.utils.plot import get_metadata_from_filename

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
            "dataset": "fastmri_knee",
            "sample_name": "1000000",
            "degraded": [
                "ReduceResolutionf=2_wavelet-fista_corrected.tiff",
                "ReduceResolutionf=3_conjugate-gradient_corrected.tiff",
                "ReduceResolutionf=4_zero-filled_corrected.tiff",
            ],
        },
        "example_002": {
            "dataset": "fastmri_knee",
            "sample_name": "1001122",
            "degraded": [
                "ReduceResolutionf=2_ram_corrected.tiff",
                "ReduceResolutionf=3_tv-fista_corrected.tiff",
                "ReduceResolutionf=4_tv-pgd_corrected.tiff",
            ],
        },
        "example_003": {
            "dataset": "fastmri_knee",
            "sample_name": "1000926",
            "degraded": [
                "ReduceResolutionf=2_wavelet-fista_corrected.tiff",
                "ReduceResolutionf=3_tv-pgd_corrected.tiff",
                "ReduceResolutionf=4_ram_corrected.tiff",
            ],
        },
        "example_004": {
            "dataset": "fastmri_brain",
            "sample_name": "AXFLAIR-201-6002940",
            "degraded": [
                "ReduceResolutionf=2_tv-fista_corrected.tiff",
                "ReduceResolutionf=3_ram_corrected.tiff",
                "ReduceResolutionf=4_conjugate-gradient_corrected.tiff",
            ],
        },
        "example_005": {
            "dataset": "fastmri_brain",
            "sample_name": "AXFLAIR-203-6000890",
            "degraded": [
                "ReduceResolutionf=2_zero-filled_corrected.tiff",
                "ReduceResolutionf=3_conjugate-gradient_corrected.tiff",
                "ReduceResolutionf=4_wavelet-fista_corrected.tiff",
            ],
        },
        "example_006": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT1-202-2020109",
            "degraded": [
                "ReduceResolutionf=2_wavelet-fista_corrected.tiff",
                "ReduceResolutionf=3_conjugate-gradient_corrected.tiff",
                "ReduceResolutionf=4_zero-filled_corrected.tiff",
            ],
        },
        "example_007": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT1POST-205-6000023",
            "degraded": [
                "ReduceResolutionf=2_ram_corrected.tiff",
                "ReduceResolutionf=3_tv-fista_corrected.tiff",
                "ReduceResolutionf=4_tv-pgd_corrected.tiff",
            ],
        },
        "example_008": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT1POST-207-2070143",
            "degraded": [
                "ReduceResolutionf=2_wavelet-fista_corrected.tiff",
                "ReduceResolutionf=3_tv-pgd_corrected.tiff",
                "ReduceResolutionf=4_ram_corrected.tiff",
            ],
        },
        "example_009": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT2-200-2000290",
            "degraded": [
                "ReduceResolutionf=2_tv-fista_corrected.tiff",
                "ReduceResolutionf=3_ram_corrected.tiff",
                "ReduceResolutionf=4_conjugate-gradient_corrected.tiff",
            ],
        },
        "example_010": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT2-201-2010519",
            "degraded": [
                "ReduceResolutionf=2_zero-filled_corrected.tiff",
                "ReduceResolutionf=3_conjugate-gradient_corrected.tiff",
                "ReduceResolutionf=4_wavelet-fista_corrected.tiff",
            ],
        },
        "example_011": {
            "dataset": "cmrxrecon",
            "sample_name": "P001-cine-lax",
            "degraded": [
                "ReduceResolutionf=2_zero-filled_corrected.tiff",
                "ReduceResolutionf=3_conjugate-gradient_corrected.tiff",
                "ReduceResolutionf=4_ram_corrected.tiff",
            ],
        },
        "example_012": {
            "dataset": "fastmri_prostate",
            "sample_name": "AXT2-001",
            "degraded": [
                "ReduceResolutionf=2_zero-filled_corrected.tiff",
                "ReduceResolutionf=3_conjugate-gradient_corrected.tiff",
                "ReduceResolutionf=4_wavelet-fista_corrected.tiff",
            ],
        },
        "example_013": {
            "dataset": "oasis",
            "sample_name": "OAS1-0088-MR1",
            "degraded": [
                "ReduceResolutionf=2_zero-filled_corrected.tiff",
                "ReduceResolutionf=3_conjugate-gradient_corrected.tiff",
                "ReduceResolutionf=4_wavelet-fista_corrected.tiff",
            ],
        },
    },
    # "property_02_sharpness": {
    #     "example_001": {
    #         "reference": "image_fastmri_knee_1000000_reference.tiff",
    #         "degraded": [
    #             "image_fastmri_knee_1000000_PartialFouriers=high_zero-filled_uncorrected.tiff",
    #             "image_fastmri_knee_1000000_PartialFouriers=high_zero-filled_corrected.tiff",
    #             "image_fastmri_knee_1000000_PartialFouriers=high_tv-pgd_corrected.tiff",
    #         ],
    #     },
    #     "example_002": {
    #         "reference": "image_fastmri_brain_AXFLAIR-200-6002452_reference.tiff",
    #         "degraded": [
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_zero-filled_uncorrected.tiff",
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_zero-filled_corrected.tiff",
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_tv-pgd_corrected.tiff",
    #         ],
    #     },
    # },
    "property_03_intensity_uniformity": {
        "example_001": {
            "dataset": "fastmri_knee",
            "sample_name": "1000000",
            "degraded": [
                "GaussianBiasFieldw=0.15e=0.05_conjugate-gradient_uncorrected_N4.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_ram_uncorrected_N4.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_zero-filled_uncorrected.tiff",
            ],
        },
        "example_002": {
            "dataset": "fastmri_knee",
            "sample_name": "1001122",
            "degraded": [
                "OffCenterAnisotropicGaussianBiasFieldw=0.3w=0.2c=0.15c=-0.1e=0.5_conjugate-gradient_uncorrected.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.1c=0.15c=-0.1e=0.1_ram_uncorrected.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.1c=0.15c=-0.1e=0.1_zero-filled_uncorrected_N4.tiff",
            ],
        },
        "example_003": {
            "dataset": "fastmri_knee",
            "sample_name": "1000926",
            "degraded": [
                "GaussianBiasFieldw=0.15e=0.05_conjugate-gradient_uncorrected_N4.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_ram_uncorrected_N4.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_zero-filled_uncorrected.tiff",
            ],
        },
        "example_004": {
            "dataset": "fastmri_brain",
            "sample_name": "AXFLAIR-201-6002940",
            "degraded": [
                "OffCenterAnisotropicGaussianBiasFieldw=0.3w=0.2c=0.15c=-0.1e=0.5_conjugate-gradient_uncorrected.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.1c=0.15c=-0.1e=0.1_ram_uncorrected.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.1c=0.15c=-0.1e=0.1_zero-filled_uncorrected_N4.tiff",
            ],
        },
        "example_005": {
            "dataset": "fastmri_brain",
            "sample_name": "AXFLAIR-203-6000890",
            "degraded": [
                "GaussianBiasFieldw=0.15e=0.05_conjugate-gradient_uncorrected_N4.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_ram_uncorrected_N4.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_zero-filled_uncorrected.tiff",
            ],
        },
        "example_006": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT1-202-2020109",
            "degraded": [
                "OffCenterAnisotropicGaussianBiasFieldw=0.3w=0.2c=0.15c=-0.1e=0.5_conjugate-gradient_uncorrected.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.1c=0.15c=-0.1e=0.1_ram_uncorrected.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.1c=0.15c=-0.1e=0.1_zero-filled_uncorrected_N4.tiff",
            ],
        },
        "example_007": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT1POST-205-6000023",
            "degraded": [
                "GaussianBiasFieldw=0.15e=0.05_conjugate-gradient_uncorrected_N4.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_ram_uncorrected_N4.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_zero-filled_uncorrected.tiff",
            ],
        },
        "example_008": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT1POST-207-2070143",
            "degraded": [
                "OffCenterAnisotropicGaussianBiasFieldw=0.3w=0.2c=0.15c=-0.1e=0.5_conjugate-gradient_uncorrected.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.1c=0.15c=-0.1e=0.1_ram_uncorrected.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.1c=0.15c=-0.1e=0.1_zero-filled_uncorrected_N4.tiff",
            ],
        },
        "example_009": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT2-200-2000290",
            "degraded": [
                "GaussianBiasFieldw=0.15e=0.05_conjugate-gradient_uncorrected_N4.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_ram_uncorrected_N4.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_zero-filled_uncorrected.tiff",
            ],
        },
        "example_010": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT2-201-2010519",
            "degraded": [
                "OffCenterAnisotropicGaussianBiasFieldw=0.3w=0.2c=0.15c=-0.1e=0.5_conjugate-gradient_uncorrected.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.1c=0.15c=-0.1e=0.1_ram_uncorrected.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.1c=0.15c=-0.1e=0.1_zero-filled_uncorrected_N4.tiff",
            ],
        },
        "example_011": {
            "dataset": "cmrxrecon",
            "sample_name": "P001-cine-lax",
            "degraded": [
                "GaussianBiasFieldw=0.15e=0.05_conjugate-gradient_uncorrected_N4.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_ram_uncorrected_N4.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_zero-filled_uncorrected.tiff",
            ],
        },
        "example_012": {
            "dataset": "fastmri_prostate",
            "sample_name": "AXT2-001",
            "degraded": [
                "OffCenterAnisotropicGaussianBiasFieldw=0.3w=0.2c=0.15c=-0.1e=0.5_conjugate-gradient_uncorrected.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.1c=0.15c=-0.1e=0.1_ram_uncorrected.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.1c=0.15c=-0.1e=0.1_zero-filled_uncorrected_N4.tiff",
            ],
        },
        "example_013": {
            "dataset": "oasis",
            "sample_name": "OAS1-0088-MR1",
            "degraded": [
                "GaussianBiasFieldw=0.15e=0.05_conjugate-gradient_uncorrected_N4.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_ram_uncorrected_N4.tiff",
                "OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_zero-filled_uncorrected.tiff",
            ],
        },
    },
    #     "example_001": {
    #         "reference": "image_fastmri_knee_1000000_reference.tiff",
    #         "degraded": [
    #             "image_fastmri_knee_1000000_GaussianBiasFieldw=0.15e=0.05_conjugate-gradient_uncorrected_N4.tiff",
    #             "image_fastmri_knee_1000007_OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_ram_uncorrected_N4.tiff",
    #             "image_fastmri_knee_1000007_OffCenterAnisotropicGaussianBiasFieldw=0.2w=0.35c=0.15c=-0.1e=0.05_zero-filled_uncorrected.tiff",
                
    #         ],
    #     },
    #     "example_002": {
    #         "reference": "image_fastmri_brain_AXFLAIR-200-6002452_reference.tiff",
    #         "degraded": [
    #             "/home/melanie.dohmen/ArtifactLab/reports/experiments_run1/image_fastmri_brain_AXFLAIR-200-6002452_GaussianBiasFieldw=0.15e=0.05_ram_uncorrected_N4.tiff",
    #             "/home/melanie.dohmen/ArtifactLab/reports/experiments_run1/image_fastmri_brain_AXFLAIR-200-6002452_GaussianBiasFieldw=0.15e=0.05_ram_uncorrected.tiff",
    #             "/home/melanie.dohmen/ArtifactLab/reports/experiments_run1/image_fastmri_brain_AXFLAIR-200-6002452_GaussianBiasFieldw=0.15e=0.05_conjugate-gradient_uncorrected.tiff",
    #             ],
    #     },
    # },
    "property_04_noise_level": {
            "example_001": {
            "dataset": "fastmri_knee",
            "sample_name": "1000000",
            "degraded": [
                "GaussianNoises=1e-05_conjugate-gradient_uncorrected.tiff",
                "GaussianNoises=1e-05_conjugate-gradient_corrected.tiff",
                "GaussianNoises=0.0001_conjugate-gradient_uncorrected.tiff",
            ],
        },
        "example_002": {
            "dataset": "fastmri_knee",
            "sample_name": "1001122",
            "degraded": [
                "GaussianNoises=1e-05_conjugate-gradient_uncorrected.tiff",
                "GaussianNoises=1e-05_conjugate-gradient_corrected.tiff",
                "GaussianNoises=0.0001_conjugate-gradient_uncorrected.tiff",
            ],
        },
        "example_003": {
            "dataset": "fastmri_knee",
            "sample_name": "1000926",
            "degraded": [
                "GaussianNoises=1e-05_conjugate-gradient_uncorrected.tiff",
                "GaussianNoises=1e-05_conjugate-gradient_corrected.tiff",
                "GaussianNoises=0.0001_conjugate-gradient_uncorrected.tiff",
            ],
        },
        "example_004": {
            "dataset": "fastmri_brain",
            "sample_name": "AXFLAIR-201-6002940",
            "degraded": [
                "GaussianNoises=1e-05_conjugate-gradient_uncorrected.tiff",
                "GaussianNoises=1e-05_conjugate-gradient_corrected.tiff",
                "GaussianNoises=0.0001_conjugate-gradient_uncorrected.tiff",
            ],
        },
        "example_005": {
            "dataset": "fastmri_brain",
            "sample_name": "AXFLAIR-203-6000890",
            "degraded": [
                "GaussianNoises=1e-05_conjugate-gradient_uncorrected.tiff",
                "GaussianNoises=1e-05_conjugate-gradient_corrected.tiff",
                "GaussianNoises=0.0001_conjugate-gradient_uncorrected.tiff",
            ],
        },
        "example_006": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT1-202-2020109",
            "degraded": [
                "GaussianNoises=1e-05_conjugate-gradient_uncorrected.tiff",
                "GaussianNoises=1e-05_conjugate-gradient_corrected.tiff",
                "GaussianNoises=0.0001_conjugate-gradient_uncorrected.tiff",
            ],
        },
        "example_007": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT1POST-205-6000023",
            "degraded": [
                "GaussianNoises=1e-05_conjugate-gradient_uncorrected.tiff",
                "GaussianNoises=1e-05_conjugate-gradient_corrected.tiff",
                "GaussianNoises=0.0001_conjugate-gradient_uncorrected.tiff",
            ],
        },
        "example_008": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT1POST-207-2070143",
            "degraded": [
                "GaussianNoises=1e-05_conjugate-gradient_uncorrected.tiff",
                "GaussianNoises=1e-05_conjugate-gradient_corrected.tiff",
                "GaussianNoises=0.0001_conjugate-gradient_uncorrected.tiff",
            ],
        },
        "example_009": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT2-200-2000290",
            "degraded": [
                "GaussianNoises=1e-05_conjugate-gradient_uncorrected.tiff",
                "GaussianNoises=1e-05_conjugate-gradient_corrected.tiff",
                "GaussianNoises=0.0001_conjugate-gradient_uncorrected.tiff",
            ],
        },
        "example_010": {
            "dataset": "fastmri_brain",
            "sample_name": "AXT2-201-2010519",
            "degraded": [
                "GaussianNoises=1e-05_conjugate-gradient_uncorrected.tiff",
                "GaussianNoises=1e-05_conjugate-gradient_corrected.tiff",
                "GaussianNoises=0.0001_conjugate-gradient_uncorrected.tiff",
            ],
        },
        "example_011": {
            "dataset": "cmrxrecon",
            "sample_name": "P001-cine-lax",
            "degraded": [
                "GaussianNoises=1e-05_conjugate-gradient_uncorrected.tiff",
                "GaussianNoises=1e-05_conjugate-gradient_corrected.tiff",
                "GaussianNoises=0.0001_conjugate-gradient_uncorrected.tiff",
            ],
        },
        "example_012": {
            "dataset": "fastmri_prostate",
            "sample_name": "AXT2-001",
            "degraded": [
                "GaussianNoises=1e-05_conjugate-gradient_uncorrected.tiff",
                "GaussianNoises=1e-05_conjugate-gradient_corrected.tiff",
                "GaussianNoises=0.0001_conjugate-gradient_uncorrected.tiff",
            ],
        },
        "example_013": {
            "dataset": "oasis",
            "sample_name": "OAS1-0088-MR1",
            "degraded": [
                "GaussianNoises=1e-05_conjugate-gradient_uncorrected.tiff",
                "GaussianNoises=1e-05_conjugate-gradient_corrected.tiff",
                "GaussianNoises=0.0001_conjugate-gradient_uncorrected.tiff",
            ],
        },
    },
    # "property_05_roi_homogeneity": {
    #     "example_001": {
    #         "reference": "image_fastmri_knee_1000000_reference.tiff",
    #         "degraded": [
    #             "image_fastmri_knee_1000000_PartialFouriers=high_zero-filled_uncorrected.tiff",
    #             "image_fastmri_knee_1000000_PartialFouriers=high_zero-filled_corrected.tiff",
    #             "image_fastmri_knee_1000000_PartialFouriers=high_tv-pgd_corrected.tiff",
    #         ],
    #     },
    #     "example_002": {
    #         "reference": "image_fastmri_brain_AXFLAIR-200-6002452_reference.tiff",
    #         "degraded": [
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_zero-filled_uncorrected.tiff",
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_zero-filled_corrected.tiff",
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_tv-pgd_corrected.tiff",
    #         ],
    #     },
    # },
    # "property_06_local_signal_preservation": {
    #     "example_001": {
    #         "reference": "image_fastmri_knee_1000000_reference.tiff",
    #         "degraded": [
    #             "image_fastmri_knee_1000000_PartialFouriers=high_zero-filled_uncorrected.tiff",
    #             "image_fastmri_knee_1000000_PartialFouriers=high_zero-filled_corrected.tiff",
    #             "image_fastmri_knee_1000000_PartialFouriers=high_tv-pgd_corrected.tiff",
    #         ],
    #     },
    #     "example_002": {
    #         "reference": "image_fastmri_brain_AXFLAIR-200-6002452_reference.tiff",
    #         "degraded": [
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_zero-filled_uncorrected.tiff",
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_zero-filled_corrected.tiff",
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_tv-pgd_corrected.tiff",
    #         ],
    #     },
    # },
    # "property_07_edges": {
    #     "example_001": {
    #         "reference": "image_fastmri_knee_1000000_reference.tiff",
    #         "degraded": [
    #             "image_fastmri_knee_1000000_PartialFouriers=high_zero-filled_uncorrected.tiff",
    #             "image_fastmri_knee_1000000_PartialFouriers=high_zero-filled_corrected.tiff",
    #             "image_fastmri_knee_1000000_PartialFouriers=high_tv-pgd_corrected.tiff",
    #         ],
    #     },
    #     "example_002": {
    #         "reference": "image_fastmri_brain_AXFLAIR-200-6002452_reference.tiff",
    #         "degraded": [
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_zero-filled_uncorrected.tiff",
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_zero-filled_corrected.tiff",
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_tv-pgd_corrected.tiff",
    #         ],
    #     },
    # },
    # "property_08_contrast_preservation_of_anatomical_structures": {
    #     "example_001": {
    #         "reference": "image_fastmri_knee_1000000_reference.tiff",
    #         "degraded": [
    #             "image_fastmri_knee_1000000_PartialFouriers=high_zero-filled_uncorrected.tiff",
    #             "image_fastmri_knee_1000000_PartialFouriers=high_zero-filled_corrected.tiff",
    #             "image_fastmri_knee_1000000_PartialFouriers=high_tv-pgd_corrected.tiff",
    #         ],
    #     },
    #     "example_002": {
    #         "reference": "image_fastmri_brain_AXFLAIR-200-6002452_reference.tiff",
    #         "degraded": [
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_zero-filled_uncorrected.tiff",
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_zero-filled_corrected.tiff",
    #             "image_fastmri_brain_AXFLAIR-200-6002452_PartialFouriers=high_tv-pgd_corrected.tiff",
    #         ],
    #     },
    # },
}

SOURCE_INFO = {
    "fastmri_knee": {
        "source_name": "fastMRI Knee",
        "source_url": "https://fastmri.med.nyu.edu/",
        "source_license": "internal research and educational purposes only",
        "data_type": "original k-space data",
    },
    "fastmri_brain": {
        "source_name": "fastMRI Brain",
        "source_url": "https://fastmri.med.nyu.edu/",
        "source_license": "internal research and educational purposes only",
        "data_type": "original k-space data",
    },
    "fastmri_prostate": {
        "source_name": "fastMRI Prostate",
        "source_url": "https://fastmri.med.nyu.edu/",
        "source_license": "internal research and educational purposes only",
        "data_type": "k-space data derived from image data",
    },
    "oasis": {
        "source_name": "OASIS",
        "source_url": "https://sites.wustl.edu/oasisbrains/",
        "source_license": "academic research purposes only",
        "data_type": "k-space data derived from image data",
    },
    "cmrxrecon": {
        "source_name": "CMRxRecon",
        "source_url": "https://www.cmrxrecon.org/",
        "source_license": "CC BY-NC-SA 4.0",
        "data_type": "original k-space data",
    },
}


def get_metadata(filename):
    meta_data = get_metadata_from_filename(filename)
    dataset_name = meta_data["dataset"]
    meta_data.update(SOURCE_INFO[dataset_name])
    return meta_data


def create_example(filename, results_dir):
    metadata = get_metadata_from_filename(filename)
    datapaths = {
        "fastmri_knee": "/home/melanie.dohmen/ArtifactLab/data/singlecoil_val",
        "oasis": "/home/melanie.dohmen/ArtifactLab/data/oasis",
        "fastmri_brain": "/home/melanie.dohmen/ArtifactLab/data/fastMRI_multicoil_brain_train0",
        "cmrxrecon": "/home/melanie.dohmen/ArtifactLab/data/CMRxRecon",
        "fastmri_prostate": "/home/melanie.dohmen/ArtifactLab/data/fastMRI_prostate_T2_IDS_001_020",
    }
    sample_name_globs = {
        "fastmri_knee": "/home/melanie.dohmen/ArtifactLab/data/singlecoil_val/*",
        "oasis": "/home/melanie.dohmen/ArtifactLab/data/oasis/*",
        "fastmri_brain": "/home/melanie.dohmen/ArtifactLab/data/fastMRI_multicoil_brain_train0/*",
        "cmrxrecon": "/home/melanie.dohmen/ArtifactLab/data/CMRxRecon/SingleCoil/Cine/TrainingSet/FullSample/*",
        "fastmri_prostate": "/home/melanie.dohmen/ArtifactLab/data/fastMRI_prostate_T2_IDS_001_020/*",
    }

    sample_list = sorted(glob.glob(sample_name_globs[metadata["dataset"]]))
    sample_list = [os.path.basename(s).replace("_", "-") for s in sample_list]
    if metadata["dataset"] == "cmrxrecon":
        sample_list = [s+"-cine-lax" for s in sample_list]
    # find metadata["sample_name"] as substring in sample_list items:
    sample_index = None
    for i, s in enumerate(sample_list):
        if metadata["sample_name"] in s:
            sample_index = i
            break

    if sample_index is None:
        print(f"Warning: sample_name {metadata['sample_name']} not found in sample_list for dataset {metadata['dataset']}.")

    factors = []
    parameters = metadata["parameters"]
    distortions = [{metadata["distortion"]: parameters}]
    if metadata["distortion"] == "ReduceResolution":
        factors = [metadata["parameters"]["factor"]]
        parameters = {}
        distortions = []
    
    add_N4Correction = False
    if "N4" in metadata["add"]:
        if metadata["distortion"] not in ["GaussianBiasField", "OffCenterAnisotropicGaussianBiasField"]:
            print("Warning: setting N4 correction for a distortion that is not a bias field distortion. This may not be necessary.")
        add_N4Correction = True
    
    
    config = {
        "data": {metadata["dataset"]: datapaths[metadata["dataset"]]},
        "distortions": distortions,
        "reconstruction_algorithms": [
            metadata["reconstruction_method"],
        ],
        "add_N4Correction": add_N4Correction,
        "resolution_reduction_factors": factors,
        "samples": [sample_index],
        "verbose": True,
        "overwrite": False,
        "results_dir": results_dir,
    }

    print(f"Creating example for filename {filename}")
    run_all(config)
    print("... done")




result_path = "/home/melanie.dohmen/ArtifactLab/reports/experiments_run1/"
export_path = "/home/melanie.dohmen/ArtifactLab/reports/exported_examples/"
plots_path = "/home/melanie.dohmen/ArtifactLab/reports/exported_examples/plots/"
os.makedirs(result_path, exist_ok=True)
os.makedirs(export_path, exist_ok=True)
os.makedirs(plots_path, exist_ok=True)


metadata_df_list = []

for property_name, property_examples in properties.items():
    property_path = os.path.join(export_path, property_name)
    os.makedirs(property_path, exist_ok=True)

    # plot
    nr_rows = len(property_examples)
    nr_cols = len(property_examples[next(iter(property_examples))]["degraded"]) + 1
    fig, axes = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols * 3, nr_rows * 3))

    for e_idx, (example_name, example_data) in enumerate(property_examples.items()):
        example_path = os.path.join(property_path, example_name)
        os.makedirs(example_path, exist_ok=True)


        # check if images exist, if not create them
        for i, degraded_image in enumerate(example_data["degraded"]):
            degraded_image_filename = f"image_{example_data['dataset']}_{example_data['sample_name']}_{degraded_image}"
            if not os.path.exists(os.path.join(result_path, degraded_image_filename)):
                create_example(degraded_image_filename, result_path)

        reference_filename = f"image_{example_data['dataset']}_{example_data['sample_name']}_reference.tiff"
        if not os.path.exists(os.path.join(result_path, reference_filename)):
            create_example(reference_filename, result_path)

        # Save reference image
        old_reference_image_path = os.path.join(result_path, reference_filename)
        new_reference_image_path = os.path.join(example_path, "reference.tiff")

        imwrite(new_reference_image_path, imread(old_reference_image_path))
        # Code to save the reference image using example_data["reference"]


        metadata = {
            "property": property_name,
            "example": example_name,
            "relative_path": os.path.relpath(new_reference_image_path, result_path),
            **get_metadata(
                old_reference_image_path
            ),  # Function to extract metadata info from filename
        }

        metadata_df_list.append(metadata)

        ref_img = imread(new_reference_image_path).squeeze()
        if ref_img.ndim == 2:
            axes[e_idx, 0].imshow(ref_img, cmap="gray")
        else:
            print("Warning: Reference image is not 2D, cannot display.")
        axes[e_idx, 0].set_axis_off()
        axes[e_idx, 0].set_title("Reference")

        # Save degraded images
        for i, degraded_image in enumerate(example_data["degraded"]):
            degraded_image_filename = f"image_{example_data['dataset']}_{example_data['sample_name']}_{degraded_image}"
            old_degraded_image_path = os.path.join(result_path, degraded_image_filename)
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

            # plot
            deg_img = imread(new_degraded_image_path).squeeze()
            if deg_img.ndim == 2:
                axes[e_idx, i + 1].imshow(deg_img, cmap="gray")
            else:
                print(f"Warning: Degraded image {new_degraded_image_path} is not 2D, cannot display.")
            axes[e_idx, i + 1].set_title(f"Degraded {i + 1}\n{metadata['distortion']}\n{metadata['reconstruction_method']}\n{list(metadata['parameters'].values())}\n{metadata['add']}")
            axes[e_idx, i + 1].set_axis_off()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f"overview_{property_name}.png"))
    plt.close(fig)

metadata_df = pd.DataFrame(metadata_df_list)
metadata_csv_path = os.path.join(export_path, "metadata.csv")
metadata_df.to_csv(metadata_csv_path, index=False)

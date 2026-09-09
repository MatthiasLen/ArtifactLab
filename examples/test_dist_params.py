import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from run_all import run_all

if __name__ == "__main__":
    results_dir = "/home/melanie.dohmen/ArtifactLab/reports/"

    # create config:
    config = {
        "data": {
            # "fastmri_knee": "/home/melanie.dohmen/ArtifactLab/data/singlecoil_val",
            # "oasis": "/home/melanie.dohmen/ArtifactLab/data/oasis",
            "fastmri_brain": "/home/melanie.dohmen/ArtifactLab/data/fastMRI_multicoil_brain_train0",
            # "cmrxrecon": "/home/melanie.dohmen/ArtifactLab/data/CMRxRecon",
            # "fastmri_prostate": "/home/melanie.dohmen/ArtifactLab/data/fastMRI_prostate_T2_IDS_001_020",
        },
        "distortions": [
            {
                "CartesianUndersamplingEquispacedZeroACS": {"keep_fraction": 0.25},
            },
        ],
        "reconstruction_algorithms": [
            "zero-filled",
            "conjugate-gradient",
            "ram",
            # - "dip"
            "tv-pgd",
            "wavelet-fista",
            "tv-fista",
            "tv-pdhg",
            "unet-fastmri",
            "unet-oasis-acceleration4",
            "unet-oasis-acceleration8",
            "unet-oasis-acceleration10",
        ],
        "add_N4Correction": True,
        "resolution_reduction_factors": [],
        "samples": [1],
        "verbose": True,
        "overwrite": True,
        "results_dir": results_dir,
    }

    distortions = [
        # {"IsotropicLP": {
        #     "radius_fraction": [0.01, 0.05, 0.1, 0.15, 0.2, 0.5]
        # }},
        # {"GaussianKspaceBiasField":  {
        #     "width_fraction": [0.1, 0.35, 0.5, 0.35, 0.35, 0.35],
        #     "edge_gain": [ 0.4, 0.4, 0.4, 0.1, 0.2, 0.6, 0.8],
        # }},
        # 0.01 width fraction toooo small!
        {
            "GaussianBiasField": {
                "width_fraction": [
                    0.1,
                    0.15,
                    0.2,
                    0.5,
                    0.1,
                    0.15,
                    0.2,
                    0.5,
                    0.1,
                    0.15,
                    0.2,
                    0.5,
                ],
                "edge_gain": [
                    0.05,
                    0.05,
                    0.05,
                    0.05,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                ],
            }
        },
        # {"OffCenterAnisotropicGaussianBiasField":  {
        #     "width_x_fraction": [ 0.1, 0.15,  0.2, 0.35, 0.1, 0.15, 0.2,  0.35, 0.1,  0.15, 0.2,  0.35],
        #     "width_y_fraction": [0.15,  0.2, 0.35, 0.1,  0.2, 0.35, 0.1,  0.15, 0.35, 0.1, 0.15, 0.2, ],
        #     "center_x_fraction": [0.15, 0.15, 0.15,0.15,  0.15,0.15,0.15,0.15,  0.15,0.15, 0.15, 0.15,],
        #     "center_y_fraction": [-0.1, -0.1, -0.1, -0.1,  -0.1,-0.1,-0.1,-0.1,  -0.1,-0.1,-0.1,-0.1,],
        #     "edge_gain": [0.05,0.05,0.05,0.05,  0.1, 0.1, 0.1, 0.1,   0.5, 0.5, 0.5, 0.5,  ],
        # }},
        # {
        #     "CartesianUndersamplingEquispacedZeroACS": {
        #         "keep_fraction": [0.95, 0.98],
        #     },
        # },
        # {
        #     "CartesianUndersamplingVariableDensity": {
        #         "keep_fraction": [0.95, 0.95, 0.98, 0.98],
        #         "center_fraction": [0.1, 0.125, 0.1, 0.125],
        #     },
        # },
        # {
        #     "CartesianUndersamplingUniformRandom": {
        #         "keep_fraction": [0.95, 0.95, 0.98, 0.98],
        #         "center_fraction": [0.1, 0.125, 0.1, 0.125],
        #     },
        # },
        # {
        #     "CartesianUndersamplingEquispaced": {
        #         "keep_fraction": [0.95, 0.95, 0.98, 0.98],
        #         "center_fraction": [0.1, 0.125, 0.1, 0.125],
        #     },
        # },
        # {
        #     "AnisotropicLP": {
        #         "kx_radius_fraction": [0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0],
        #         "ky_radius_fraction": [1.0, 0.95, 0.6, 0.85, 0.75, 0.5, 0.25, 0.1],
        #     },
        # },
        # {
        #     "HannTaperLP": {
        #         "radius_fraction": [0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 1.0, 1.0],  # 0.35,
        #         "transition_fraction": [0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6],  # 0.4,
        #     }
        # },
        # {
        #     "KaiserTaperLP": {
        #         "radius_fraction": [0.15, 0.2, 0.15, 0.2, 0.25, 0.3, 0.25, 0.3],  # 0.35,
        #         "transition_fraction": [0.2, 0.6, 0.2, 0.6, 0.2, 0.6, 0.2, 0.6],  # 0.4,
        #         "beta": [8.6, 8.6, 8.6, 8.6, 2.0, 2.0, 8.6, 8.6],  # 8.6
        #     }
        # },
        # {
        #     "GaussianNoise": {
        #         "sigma": [0.000005, 0.00001, 0.00002 ],  # [0.00001]
        #     }
        # },
        # {
        #     "IsotropicLP": {
        #         "radius_fraction": [0.15, 0.2],  # 0.1
        #     }
        # },
        # {
        #     "RadialHighPassEmphasis": {
        #         "alpha": [1.5, 2.0, 3.0, 5.0, 10.0],  # 0.4
        #     }
        # },
    ]

    for d_idx, distortion_dict in enumerate(distortions):
        for distortion_name, dist_params in distortion_dict.items():
            nr_param_values = len(dist_params[list(dist_params.keys())[0]])
            config["distortions"] = []
            for v_idx in range(nr_param_values):
                single_value_distortion_dict = {
                    distortion_name: {
                        param: param_values[v_idx] for param, param_values in dist_params.items()
                    }
                }
                config["distortions"].append(single_value_distortion_dict)

            config["results_dir"] = os.path.join(results_dir, f"test_params_{distortion_name}")

            run_all(config)

import torch

from .base import BaseDistortion
from .resolution import (
    HannTaperResolutionReduction,
    IsotropicResolutionReduction,
    AnisotropicResolutionReduction,
    KaiserTaperResolutionReduction,
    RadialHighPassEmphasisDistortion,
)
from .undersampling import CartesianUndersampling, PartialFourierDistortion
from .biasfield import OffCenterAnisotropicGaussianKspaceBiasField, GaussianKspaceBiasField
from .noise import GaussianNoiseDistortion
from .motion import (
    RotationalMotionDistortion,
    SegmentedRotationalMotionDistortion,
    TranslationMotionDistortion,
    SegmentedTranslationMotionDistortion,
)
from .ghosting import PhaseEncodeGhostingDistortion


def choose_distortion(
    name: str,
    keep_fraction: float = 0.25,
    center_fraction: float = 0.125,
    cartesian_axis: int = -2,
) -> BaseDistortion:
    """Build one distortion operator for the inference comparison script.

    The ``cartesian_axis`` is supplied by the active measurement convention:
    FastMRI-native runs use the repository's existing axis, while OASIS-native
    and FastMRI-to-OASIS runs use the centered OASIS axis.
    """

    match name:
        case "PhaseEncodeGhosting":
            return PhaseEncodeGhostingDistortion(
                line_period=2,
                line_offset=1,
                phase_error_radians=torch.pi / 2,
                corrupted_line_scale=1.0,
            )
        case "CartesianUndersamplingVariableDensity":
            return CartesianUndersampling(
                keep_fraction=keep_fraction,
                center_fraction=center_fraction,
                pattern="variable_density_random",
                axis=cartesian_axis,
                seed=42,
            )
        case "CartesianUndersamplingUniformRandom":
            return CartesianUndersampling(
                keep_fraction=keep_fraction,
                center_fraction=center_fraction,
                pattern="uniform_random",
                axis=cartesian_axis,
                seed=42,
            )
        case "CartesianUndersamplingUniformRandomZeroACS":
            return CartesianUndersampling(
                keep_fraction=keep_fraction,
                center_fraction=0.0,
                pattern="uniform_random",
                axis=cartesian_axis,
                seed=42,
            )
        case "CartesianUndersamplingEquispaced":
            return CartesianUndersampling(
                keep_fraction=keep_fraction,
                center_fraction=center_fraction,
                pattern="equispaced",
                axis=cartesian_axis,
                seed=42,
            )
        case "CartesianUndersamplingEquispacedZeroACS":
            return CartesianUndersampling(
                keep_fraction=keep_fraction,
                center_fraction=0.0,
                pattern="equispaced",
                axis=cartesian_axis,
                seed=42,
            )
        case "PartialFourier":
            return PartialFourierDistortion(
                partial_fraction=0.7,
                center_fraction=center_fraction,
                axis=cartesian_axis,
                side="high",
            )
        case "AnisotropicLP":
            return AnisotropicResolutionReduction(
                kx_radius_fraction=1.0,
                ky_radius_fraction=0.25,
            )
        case "HannTaperLP":
            return HannTaperResolutionReduction(
                radius_fraction=0.35,
                transition_fraction=0.4,
            )
        case "KaiserTaperLP":
            return KaiserTaperResolutionReduction(
                radius_fraction=0.35,
                transition_fraction=0.4,
                beta=8.6,
            )
        case "RadialHighPassEmphasis":
            return RadialHighPassEmphasisDistortion(alpha=0.4)
        case "IsotropicLP":
            return IsotropicResolutionReduction(radius_fraction=0.1)
        case "OffCenterAnisotropicGaussianKspaceBiasField":
            return OffCenterAnisotropicGaussianKspaceBiasField(
                width_x_fraction=0.2,
                width_y_fraction=0.35,
                center_x_fraction=0.15,
                center_y_fraction=-0.1,
                edge_gain=0.3,
            )
        case "TranslationMotion":
            return TranslationMotionDistortion(shift_x_pixels=60, shift_y_pixels=10)
        case "RotationalMotion":
            return RotationalMotionDistortion(angle_radians=torch.pi / 6)
        case "SegmentedRotationalMotion":
            return SegmentedRotationalMotionDistortion(
                angle_radians=(0.0, torch.pi / 20, -torch.pi / 24, torch.pi / 16),
            )
        case "SegmentedTranslationMotion":
            return SegmentedTranslationMotionDistortion(
                shift_x_pixels=(0.0, 20.0, 50.0, -50.0),
                shift_y_pixels=(0.0, 10.0, -20.0, 20.0),
            )
        case "GaussianKspaceBiasField":
            return GaussianKspaceBiasField(width_fraction=0.35, edge_gain=0.4)
        case "GaussianNoise":
            return GaussianNoiseDistortion(sigma=0.00001)
        case "BaseDistortion":
            return BaseDistortion()

        case _:
            raise ValueError(f"Unknown distortion {name!r}")

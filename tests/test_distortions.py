import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import sqrt
import pytest
import torch

from mri_recon.distortions import (
    AnisotropicResolutionReduction,
    BaseDistortion,
    DistortedKspaceMultiCoilMRI,
    GaussianKspaceBiasField,
    GaussianNoiseDistortion,
    HannTaperResolutionReduction,
    IsotropicResolutionReduction,
    KaiserTaperResolutionReduction,
    OffCenterAnisotropicGaussianKspaceBiasField,
    PhaseEncodeGhostingDistortion,
    RadialHighPassEmphasisDistortion,
    RotationalMotionDistortion,
    SegmentedTranslationMotionDistortion,
    SelfAdjointMultiplicativeMaskDistortion,
    TranslationMotionDistortion,
)

DISTORTIONS = [
    "None",
    "Isotropic LP",
    "Anisotropic LP",
    "Hann taper LP",
    "Kaiser taper LP",
    "Radial high-pass emphasis",
    "Gaussian bias field",
    "Off-center anisotropic Gaussian bias field",
    "Phase-encode ghosting",
    "Translation motion",
    "Rotational motion",
    "Segmented translation motion",
]

EXACT_OPERATOR_DISTORTIONS = {
    "None",
    "Isotropic LP",
    "Anisotropic LP",
    "Phase-encode ghosting",
    "Translation motion",
    "Segmented translation motion",
    "Rotational motion",
}
NON_EXPANSIVE_DISTORTIONS = {
    "Off-center anisotropic Gaussian bias field",
    "Gaussian bias field",
}


def choose_distortion(name):
    match name:
        case "None":
            return BaseDistortion()
        case "Isotropic LP":
            return IsotropicResolutionReduction(radius_fraction=0.6)
        case "Anisotropic LP":
            return AnisotropicResolutionReduction(
                kx_radius_fraction=1.0,
                ky_radius_fraction=0.35,
            )
        case "Hann taper LP":
            return HannTaperResolutionReduction(
                radius_fraction=0.6,
                transition_fraction=0.25,
            )
        case "Kaiser taper LP":
            return KaiserTaperResolutionReduction(
                radius_fraction=0.6,
                transition_fraction=0.25,
                beta=8.6,
            )
        case "Radial high-pass emphasis":
            return RadialHighPassEmphasisDistortion(alpha=0.4, exponent=2.0)
        case "Gaussian bias field":
            return GaussianKspaceBiasField(width_fraction=0.35, edge_gain=0.4)
        case "Off-center anisotropic Gaussian bias field":
            return OffCenterAnisotropicGaussianKspaceBiasField(
                width_x_fraction=0.2,
                width_y_fraction=0.35,
                center_x_fraction=0.15,
                center_y_fraction=-0.1,
                edge_gain=0.3,
            )
        case "Phase-encode ghosting":
            return PhaseEncodeGhostingDistortion(
                line_period=2,
                line_offset=1,
                phase_error_radians=torch.pi / 2,
                corrupted_line_scale=1.0,
            )
        case "Translation motion":
            return TranslationMotionDistortion(shift_x_pixels=8.0, shift_y_pixels=4.0)
        case "Rotational motion":
            return RotationalMotionDistortion(angle_radians=torch.pi / 6)
        case "Segmented translation motion":
            return SegmentedTranslationMotionDistortion(
                shift_x_pixels=(0.0, 2.0, 5.0, 5.0),
                shift_y_pixels=(0.0, 1.0, 2.0, 2.0),
            )
        case _:
            raise ValueError(f"Unknown distortion {name!r}")


@pytest.fixture
def device():
    return "cpu"


@pytest.mark.parametrize("name", DISTORTIONS)
@pytest.mark.parametrize(
    "img_size", [(1, 2, 256, 256), (1, 2, 4, 256, 256)]
)  # singlecoil and multicoil
def test_distortion_properties(name, img_size, device):
    """
    Test exact operators for adjointness and norm preservation, and verify that
    approximate or attenuation operators remain shape-preserving and non-expansive.
    """
    distortion = choose_distortion(name)
    y = torch.randn(img_size, device=device)
    x_dummy = torch.randn(1, 2, *img_size[-2:], device=device)

    if name in EXACT_OPERATOR_DISTORTIONS:
        assert distortion.adjointness_test(x_dummy) < 0.01
        assert abs(distortion.compute_norm(x_dummy, squared=False) - 1) < 0.01
    elif name in NON_EXPANSIVE_DISTORTIONS:
        y_distorted = distortion.A(x_dummy)
        assert y_distorted.shape == x_dummy.shape
        assert torch.max(torch.abs(y_distorted)) <= torch.max(torch.abs(x_dummy)) + 1e-6

    if len(img_size) == 4:  # singlecoil
        coil_maps = None
    elif len(img_size) == 5:  # multicoil
        coil_maps = torch.ones(1, *img_size[-3:], device=device, dtype=torch.complex64) / sqrt(
            img_size[-3]
        )
    physics = DistortedKspaceMultiCoilMRI(
        distortion=distortion, img_size=(1, 2, *y.shape[-2:]), coil_maps=coil_maps, device=device
    )

    if name in EXACT_OPERATOR_DISTORTIONS:
        assert physics.adjointness_test(x_dummy) < 0.01
        assert abs(physics.compute_norm(x_dummy, squared=False) - 1) < 0.01
    elif name in NON_EXPANSIVE_DISTORTIONS:
        y_physics = physics.A(x_dummy)
        y_clean = DistortedKspaceMultiCoilMRI(
            distortion=BaseDistortion(),
            img_size=(1, 2, *y.shape[-2:]),
            coil_maps=coil_maps,
            device=device,
        ).A(x_dummy)
        assert torch.max(torch.abs(y_physics)) <= torch.max(torch.abs(y_clean)) + 1e-6


def test_gaussian_noise_distortion_preserves_shape_and_changes_values(device):
    distortion = GaussianNoiseDistortion(sigma=0.1)
    y = torch.zeros((1, 2, 64, 64), device=device)

    y_distorted = distortion.A(y)

    assert y_distorted.shape == y.shape
    assert y_distorted.dtype == y.dtype
    assert not torch.equal(y_distorted, y)


def test_gaussian_noise_distortion_zero_sigma_is_identity(device):
    distortion = GaussianNoiseDistortion(sigma=0.0)
    y = torch.randn((1, 2, 64, 64), device=device)

    y_distorted = distortion.A(y)

    assert torch.equal(y_distorted, y)


def test_anisotropic_resolution_reduction_zeroes_only_filtered_axis(device):
    distortion = AnisotropicResolutionReduction(
        kx_radius_fraction=1.0,
        ky_radius_fraction=0.3,
    )
    y = torch.ones((1, 2, 9, 11), device=device)

    y_distorted = distortion.A(y)
    mask = distortion._mask(y.shape, y.device)

    assert torch.all(y_distorted == y * mask)
    assert torch.all(mask[y.shape[-2] // 2, :] == 1)
    assert torch.all(mask[0, :] == 0)


def test_anisotropic_resolution_reduction_identity_at_full_cutoffs(device):
    distortion = AnisotropicResolutionReduction(
        kx_radius_fraction=1.0,
        ky_radius_fraction=1.0,
    )
    y = torch.randn((1, 2, 32, 32), device=device)

    y_distorted = distortion.A(y)

    assert torch.equal(y_distorted, y)


def test_hann_taper_resolution_reduction_has_smooth_transition(device):
    distortion = HannTaperResolutionReduction(
        radius_fraction=0.8,
        transition_fraction=0.5,
    )

    mask = distortion._mask((1, 2, 33, 33), torch.device(device))

    assert mask[16, 16] == pytest.approx(1.0)
    assert mask[0, 0] == pytest.approx(0.0)
    assert torch.any((mask > 0.0) & (mask < 1.0))


def test_hann_taper_resolution_reduction_zero_transition_matches_hard_cutoff(device):
    hard = IsotropicResolutionReduction(radius_fraction=0.6)
    smooth = HannTaperResolutionReduction(radius_fraction=0.6, transition_fraction=0.0)
    y = torch.randn((1, 2, 64, 64), device=device)

    assert torch.equal(smooth.A(y), hard.A(y))


def test_kaiser_taper_resolution_reduction_has_smooth_transition(device):
    distortion = KaiserTaperResolutionReduction(
        radius_fraction=0.8,
        transition_fraction=0.5,
        beta=8.6,
    )

    mask = distortion._mask((1, 2, 33, 33), torch.device(device))

    assert mask[16, 16] == pytest.approx(1.0)
    assert mask[0, 0] == pytest.approx(0.0)
    assert torch.any((mask > 0.0) & (mask < 1.0))


def test_kaiser_taper_resolution_reduction_zero_transition_matches_hard_cutoff(device):
    hard = IsotropicResolutionReduction(radius_fraction=0.6)
    smooth = KaiserTaperResolutionReduction(
        radius_fraction=0.6,
        transition_fraction=0.0,
        beta=8.6,
    )
    y = torch.randn((1, 2, 64, 64), device=device)

    assert torch.equal(smooth.A(y), hard.A(y))


def test_radial_high_pass_emphasis_distortion_boosts_edges_more_than_center(device):
    distortion = RadialHighPassEmphasisDistortion(alpha=0.4, exponent=2.0)
    shape = (1, 2, 33, 33)
    center_y = shape[-2] // 2
    center_x = shape[-1] // 2

    mask = distortion._mask(shape, torch.device(device))

    assert mask[center_y, center_x] == pytest.approx(1.0)
    assert mask[0, 0] == pytest.approx(1.0 + distortion.alpha)
    assert torch.all(mask >= 1.0)


def test_radial_high_pass_emphasis_distortion_zero_alpha_is_identity(device):
    distortion = RadialHighPassEmphasisDistortion(alpha=0.0, exponent=2.0)
    y = torch.randn((1, 2, 64, 64), device=device)

    assert torch.equal(distortion.A(y), y)


def test_centered_isotropic_bias_matches_anisotropic_special_case(device):
    centered = GaussianKspaceBiasField(width_fraction=0.35, edge_gain=0.4)
    anisotropic = OffCenterAnisotropicGaussianKspaceBiasField(
        width_x_fraction=0.35,
        width_y_fraction=0.35,
        center_x_fraction=0.0,
        center_y_fraction=0.0,
        edge_gain=0.4,
    )
    y = torch.randn((1, 2, 64, 64), device=device)

    y_centered = centered.A(y)
    y_anisotropic = anisotropic.A(y)

    assert torch.allclose(y_centered, y_anisotropic)


def test_translation_motion_zero_shift_is_identity(device):
    distortion = TranslationMotionDistortion(shift_x_pixels=0.0, shift_y_pixels=0.0)
    y = torch.randn((1, 2, 64, 64), device=device)

    y_distorted = distortion.A(y)

    assert torch.equal(y_distorted, y)


def test_translation_motion_preserves_kspace_magnitude(device):
    distortion = TranslationMotionDistortion(shift_x_pixels=8.0, shift_y_pixels=4.0)
    y = torch.randn((1, 2, 64, 64), device=device)

    y_distorted = distortion.A(y)
    y_complex = torch.view_as_complex(y.movedim(1, -1).contiguous())
    y_distorted_complex = torch.view_as_complex(y_distorted.movedim(1, -1).contiguous())

    assert torch.allclose(torch.abs(y_distorted_complex), torch.abs(y_complex))


def test_translation_motion_produces_requested_image_shift(device):
    distortion = TranslationMotionDistortion(shift_x_pixels=8.0, shift_y_pixels=4.0)
    image = torch.zeros((64, 64), dtype=torch.complex64, device=device)
    image[20, 18] = 1.0

    kspace = torch.fft.fftshift(torch.fft.fft2(image))
    y = torch.view_as_real(kspace).movedim(-1, 0).unsqueeze(0).contiguous()

    y_distorted = distortion.A(y)
    kspace_distorted = torch.view_as_complex(y_distorted[0].movedim(0, -1).contiguous())
    image_distorted = torch.fft.ifft2(torch.fft.ifftshift(kspace_distorted))
    max_position = torch.nonzero(torch.abs(image_distorted) == torch.abs(image_distorted).max())[
        0
    ].tolist()

    assert max_position == [24, 26]


def test_translation_motion_rejects_invalid_kspace_shape(device):
    distortion = TranslationMotionDistortion(shift_x_pixels=1.0, shift_y_pixels=1.0)
    y = torch.randn((1, 64, 64), device=device)

    with pytest.raises(ValueError, match="Expected k-space with shape"):
        distortion.A(y)


def test_translation_motion_rejects_invalid_channel_dimension(device):
    distortion = TranslationMotionDistortion(shift_x_pixels=1.0, shift_y_pixels=1.0)
    y = torch.randn((1, 3, 64, 64), device=device)

    with pytest.raises(ValueError, match="channel dimension of size 2"):
        distortion.A(y)


def test_translation_motion_rejects_non_floating_tensor(device):
    distortion = TranslationMotionDistortion(shift_x_pixels=1.0, shift_y_pixels=1.0)
    y = torch.zeros((1, 2, 64, 64), device=device, dtype=torch.int64)

    with pytest.raises(TypeError, match="floating-point"):
        distortion.A(y)


def test_rotational_motion_zero_angle_is_identity(device):
    distortion = RotationalMotionDistortion(angle_radians=0.0)
    y = torch.randn((1, 2, 64, 64), device=device)

    assert torch.equal(distortion.A(y), y)
    assert torch.equal(distortion.A_adjoint(y), y)


def test_rotational_motion_rejects_invalid_kspace_shape(device):
    distortion = RotationalMotionDistortion(angle_radians=torch.pi / 8)
    y = torch.randn((1, 64, 64), device=device)

    with pytest.raises(ValueError, match="Expected k-space with shape"):
        distortion.A(y)


def test_rotational_motion_rejects_non_floating_tensor(device):
    distortion = RotationalMotionDistortion(angle_radians=torch.pi / 8)
    y = torch.zeros((1, 2, 64, 64), device=device, dtype=torch.int64)

    with pytest.raises(TypeError, match="floating-point"):
        distortion.A(y)


def test_rotational_motion_rotates_image_content(device):
    angle_radians = -0.5 * torch.pi
    distortion = RotationalMotionDistortion(angle_radians=angle_radians)
    image = torch.zeros((63, 63), dtype=torch.complex64, device=device)
    image[31, 40] = 1.0

    kspace = torch.fft.fftshift(torch.fft.fft2(image))
    y = torch.view_as_real(kspace).movedim(-1, 0).unsqueeze(0).contiguous()

    y_distorted = distortion.A(y)
    kspace_distorted = torch.view_as_complex(y_distorted[0].movedim(0, -1).contiguous())
    image_distorted = torch.fft.ifft2(torch.fft.ifftshift(kspace_distorted))
    magnitude = torch.abs(image_distorted)
    max_index = magnitude.reshape(-1).argmax()
    max_position = torch.tensor(torch.unravel_index(max_index, magnitude.shape), device=device)

    assert torch.equal(max_position, torch.tensor([40, 32], device=device))


def test_rotational_motion_uses_matched_adjoint(device):
    distortion = RotationalMotionDistortion(angle_radians=torch.pi / 6)
    x = torch.randn((1, 2, 64, 64), device=device)
    y = torch.randn((1, 2, 64, 64), device=device)

    lhs = torch.sum(distortion.A(x) * y)
    rhs = torch.sum(x * distortion.A_adjoint(y))

    assert torch.allclose(lhs, rhs, atol=1e-4, rtol=1e-4)


def test_phase_encode_ghosting_zero_phase_and_unit_scale_is_identity(device):
    distortion = PhaseEncodeGhostingDistortion(
        line_period=2,
        line_offset=1,
        phase_error_radians=0.0,
        corrupted_line_scale=1.0,
    )
    y = torch.randn((1, 2, 64, 64), device=device)

    y_distorted = distortion.A(y)

    assert torch.equal(y_distorted, y)


def test_phase_encode_ghosting_modulates_selected_lines(device):
    distortion = PhaseEncodeGhostingDistortion(
        line_period=3,
        line_offset=1,
        phase_error_radians=torch.pi / 2,
        corrupted_line_scale=0.75,
        ghost_axis=-2,
    )
    y = torch.randn((2, 2, 3, 18, 20), device=device)

    y_distorted = distortion.A(y)
    y_complex = torch.view_as_complex(y.movedim(1, -1).contiguous())
    y_distorted_complex = torch.view_as_complex(y_distorted.movedim(1, -1).contiguous())

    line_factor = torch.polar(
        torch.tensor(0.75, device=device),
        torch.tensor(torch.pi / 2, device=device),
    )
    for line_index in range(y.shape[-2]):
        if (line_index - distortion.line_offset) % distortion.line_period == 0:
            expected = y_complex[:, :, line_index, :] * line_factor
        else:
            expected = y_complex[:, :, line_index, :]
        actual = y_distorted_complex[:, :, line_index, :]
        assert torch.allclose(actual, expected)


def test_phase_encode_ghosting_creates_half_fov_replica_for_partial_alternating_phase(device):
    distortion = PhaseEncodeGhostingDistortion(
        line_period=2,
        line_offset=1,
        phase_error_radians=torch.pi / 2,
        corrupted_line_scale=1.0,
    )
    image = torch.zeros((32, 32), dtype=torch.complex64, device=device)
    image[5, 7] = 1.0

    kspace = torch.fft.fftshift(torch.fft.fft2(image))
    y = torch.view_as_real(kspace).movedim(-1, 0).unsqueeze(0).contiguous()

    y_distorted = distortion.A(y)
    ghosted_kspace = torch.view_as_complex(y_distorted[0].movedim(0, -1).contiguous())
    ghosted_image = torch.fft.ifft2(torch.fft.ifftshift(ghosted_kspace))

    peak_locations = torch.nonzero(torch.abs(ghosted_image) > 0.3)

    assert peak_locations.shape[0] == 2
    assert [5, 7] in peak_locations.tolist()
    assert [21, 7] in peak_locations.tolist()


def test_segmented_translation_motion_zero_shift_is_identity(device):
    distortion = SegmentedTranslationMotionDistortion(
        shift_x_pixels=(0.0, 0.0, 0.0),
        shift_y_pixels=(0.0, 0.0, 0.0),
    )
    y = torch.randn((1, 2, 64, 64), device=device)

    y_distorted = distortion.A(y)

    assert torch.equal(y_distorted, y)


def test_segmented_translation_motion_matches_phase_ramp_per_phase_encode_segment(device):
    distortion = SegmentedTranslationMotionDistortion(
        shift_x_pixels=(0.0, 1.5, -2.0),
        shift_y_pixels=(0.0, 3.0, -1.0),
        segment_axis=-2,
    )
    y = torch.randn((2, 2, 3, 18, 20), device=device)

    y_distorted = distortion.A(y)
    y_complex = torch.view_as_complex(y.movedim(1, -1).contiguous())
    y_distorted_complex = torch.view_as_complex(y_distorted.movedim(1, -1).contiguous())

    for segment_slice, shift_x, shift_y in zip(
        distortion._segment_slices(y.shape),
        distortion.shift_x_pixels,
        distortion.shift_y_pixels,
        strict=True,
    ):
        ramp = distortion._phase_ramp(y.shape, y.device, shift_x, shift_y)
        expected = y_complex[:, :, segment_slice, :] * ramp[segment_slice, :]
        actual = y_distorted_complex[:, :, segment_slice, :]
        assert torch.allclose(actual, expected)


def test_segmented_translation_motion_matches_phase_ramp_per_readout_segment(device):
    distortion = SegmentedTranslationMotionDistortion(
        shift_x_pixels=(0.0, 2.0),
        shift_y_pixels=(0.0, 1.0),
        segment_axis=-1,
    )
    y = torch.randn((1, 2, 16, 14), device=device)

    y_distorted = distortion.A(y)
    y_complex = torch.view_as_complex(y.movedim(1, -1).contiguous())
    y_distorted_complex = torch.view_as_complex(y_distorted.movedim(1, -1).contiguous())

    for segment_slice, shift_x, shift_y in zip(
        distortion._segment_slices(y.shape),
        distortion.shift_x_pixels,
        distortion.shift_y_pixels,
        strict=True,
    ):
        ramp = distortion._phase_ramp(y.shape, y.device, shift_x, shift_y)
        expected = y_complex[:, :, segment_slice] * ramp[:, segment_slice]
        actual = y_distorted_complex[:, :, segment_slice]
        assert torch.allclose(actual, expected)


def test_segmented_translation_motion_rejects_too_many_segments(device):
    distortion = SegmentedTranslationMotionDistortion(
        shift_x_pixels=(0.0, 1.0, 2.0, 3.0, 4.0),
        shift_y_pixels=(0.0, 1.0, 2.0, 3.0, 4.0),
    )
    y = torch.randn((1, 2, 4, 64), device=device)

    with pytest.raises(ValueError, match="non-empty segments"):
        distortion.A(y)


def test_segmented_translation_motion_keeps_zero_motion_segment_and_modulates_shifted_segment(
    device,
):
    distortion = SegmentedTranslationMotionDistortion(
        shift_x_pixels=(0.0, 4.0),
        shift_y_pixels=(0.0, 2.0),
    )
    y = torch.randn((1, 2, 64, 64), device=device)

    y_distorted = distortion.A(y)
    y_complex = torch.view_as_complex(y.movedim(1, -1).contiguous())
    y_distorted_complex = torch.view_as_complex(y_distorted.movedim(1, -1).contiguous())
    first_segment, second_segment = distortion._segment_slices(y.shape)

    first_segment_expected = y_complex[:, first_segment, :]
    first_segment_actual = y_distorted_complex[:, first_segment, :]

    ramp = distortion._phase_ramp(
        y.shape,
        y.device,
        shift_x_pixels=distortion.shift_x_pixels[1],
        shift_y_pixels=distortion.shift_y_pixels[1],
    )
    second_segment_expected = y_complex[:, second_segment, :] * ramp[second_segment, :]
    second_segment_actual = y_distorted_complex[:, second_segment, :]

    assert torch.allclose(first_segment_actual, first_segment_expected)
    assert torch.allclose(second_segment_actual, second_segment_expected)


@pytest.mark.parametrize(
    "distortion_cls",
    [
        IsotropicResolutionReduction,
        AnisotropicResolutionReduction,
        HannTaperResolutionReduction,
        KaiserTaperResolutionReduction,
        RadialHighPassEmphasisDistortion,
    ],
)
def test_resolution_reduction_classes_inherit_from_self_adjoint_multiplicative_mask(
    distortion_cls,
):
    """Verify that all resolution-reduction classes are subclasses of the shared super class."""
    assert issubclass(distortion_cls, SelfAdjointMultiplicativeMaskDistortion)
    assert issubclass(distortion_cls, BaseDistortion)


def test_self_adjoint_multiplicative_mask_distortion_requires_mask_implementation(device):
    """Verify that the base super class raises NotImplementedError when _mask is not overridden."""

    class IncompleteDistortion(SelfAdjointMultiplicativeMaskDistortion):
        pass

    distortion = IncompleteDistortion()
    y = torch.randn((1, 2, 8, 8), device=device)

    with pytest.raises(NotImplementedError):
        distortion.A(y)


def test_self_adjoint_multiplicative_mask_distortion_a_adjoint_equals_a(device):
    """Verify that A_adjoint equals A for a concrete SelfAdjointMultiplicativeMaskDistortion."""
    distortion = IsotropicResolutionReduction(radius_fraction=0.7)
    y = torch.randn((1, 2, 32, 32), device=device)

    assert torch.equal(distortion.A(y), distortion.A_adjoint(y))

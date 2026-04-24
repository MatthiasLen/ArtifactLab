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
    IsotropicResolutionReduction,
    OffCenterAnisotropicGaussianKspaceBiasField,
    PhaseEncodeGhostingDistortion,
    SegmentedTranslationMotionDistortion,
    TranslationMotionDistortion,
)

DISTORTIONS = [
    "None",
    "Isotropic LP",
    "Anisotropic LP",
    "Gaussian bias field",
    "Off-center anisotropic Gaussian bias field",
    "Phase-encode ghosting",
    "Translation motion",
    "Segmented translation motion",
]


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
    Test that distortion itself and distortion inserted into MultiCoilMRI both satisfy:
    - Adjointness = 0
    - Norm = 1
    """
    distortion = choose_distortion(name)
    y = torch.randn(img_size, device=device)
    x_dummy = torch.randn(1, 2, *img_size[-2:], device=device)

    assert distortion.adjointness_test(x_dummy) < 0.01
    if name in {"Gaussian bias field", "Off-center anisotropic Gaussian bias field"}:
        y_distorted = distortion.A(x_dummy)
        assert torch.max(torch.abs(y_distorted)) <= torch.max(torch.abs(x_dummy)) + 1e-6
    else:
        assert abs(distortion.compute_norm(x_dummy, squared=False) - 1) < 0.01

    if len(img_size) == 4:  # singlecoil
        coil_maps = None
    elif len(img_size) == 5:  # multicoil
        coil_maps = torch.ones(1, *img_size[-3:], device=device, dtype=torch.complex64) / sqrt(
            img_size[-3]
        )
    physics = DistortedKspaceMultiCoilMRI(
        distortion=distortion, img_size=(1, 2, *y.shape[-2:]), coil_maps=coil_maps, device=device
    )

    assert physics.adjointness_test(x_dummy) < 0.01
    if name in {"Gaussian bias field", "Off-center anisotropic Gaussian bias field"}:
        y_physics = physics.A(x_dummy)
        y_clean = DistortedKspaceMultiCoilMRI(
            distortion=BaseDistortion(),
            img_size=(1, 2, *y.shape[-2:]),
            coil_maps=coil_maps,
            device=device,
        ).A(x_dummy)
        assert torch.max(torch.abs(y_physics)) <= torch.max(torch.abs(y_clean)) + 1e-6
    else:
        assert abs(physics.compute_norm(x_dummy, squared=False) - 1) < 0.01


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

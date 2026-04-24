import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import sqrt
import pytest
import torch

from mri_recon.distortions import (
    BaseDistortion,
    DistortedKspaceMultiCoilMRI,
    GaussianKspaceBiasField,
    GaussianNoiseDistortion,
    IsotropicResolutionReduction,
    OffCenterAnisotropicGaussianKspaceBiasField,
    TranslationMotionDistortion,
)

DISTORTIONS = [
    "None",
    "Isotropic LP",
    "Gaussian bias field",
    "Off-center anisotropic Gaussian bias field",
    "Translation motion",
]


def choose_distortion(name):
    match name:
        case "None":
            return BaseDistortion()
        case "Isotropic LP":
            return IsotropicResolutionReduction(radius_fraction=0.6)
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
        case "Translation motion":
            return TranslationMotionDistortion(shift_x_pixels=8.0, shift_y_pixels=4.0)
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

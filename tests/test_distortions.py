import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import sqrt
import pytest
import torch

from mri_recon.distortions import (
    BaseDistortion,
    ComplexGaussianNoiseDistortion,
    DistortedKspaceMultiCoilMRI,
    GaussianNoiseDistortion,
    IsotropicResolutionReduction,
)

DISTORTIONS = [
    "None",
    "Isotropic LP",
]


def choose_distortion(name):
    match name:
        case "None":
            return BaseDistortion()
        case "Isotropic LP":
            return IsotropicResolutionReduction(radius_fraction=0.6)
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


def test_complex_gaussian_noise_distortion_preserves_shape_and_changes_values(device):
    distortion = ComplexGaussianNoiseDistortion(sigma=0.1)
    y = torch.zeros((1, 2, 64, 64), device=device)

    y_distorted = distortion.A(y)

    assert y_distorted.shape == y.shape
    assert y_distorted.dtype == y.dtype
    assert not torch.equal(y_distorted, y)


def test_complex_gaussian_noise_distortion_zero_sigma_is_identity(device):
    distortion = ComplexGaussianNoiseDistortion(sigma=0.0)
    y = torch.randn((1, 2, 64, 64), device=device)

    y_distorted = distortion.A(y)

    assert torch.equal(y_distorted, y)

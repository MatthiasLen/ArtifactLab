import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import deepinv as dinv

from mri_recon.reconstruction._fastmri_unet import Unet
from mri_recon.reconstruction.deep import (
    RAMReconstructor,
    DeepImagePriorReconstructor,
    FastMRISinglecoilUnetReconstructor,
)
from mri_recon.reconstruction.classic import (
    ZeroFilledReconstructor,
    ConjugateGradientReconstructor,
    TVPGDReconstructor,
    WaveletFISTAReconstructor,
    TVFISTAReconstructor,
)
from mri_recon.distortions import DistortedKspaceMultiCoilMRI

ALGORITHMS = [
    "zero-filled",
    "conjugate-gradient",
    "ram",
    "dip",
    "tv-pgd",
    "wavelet-fista",
    "tv-fista",
]


def choose_algorithm(name, img_size, device):
    match name:
        case "zero-filled":
            return ZeroFilledReconstructor()
        case "conjugate-gradient":
            return ConjugateGradientReconstructor(max_iter=20)
        case "ram":
            return RAMReconstructor(default_sigma=0.05, device=device)
        case "dip":
            return DeepImagePriorReconstructor(img_size=img_size[-2:], n_iter=100)
        case "tv-pgd":
            return TVPGDReconstructor(n_iter=100, verbose=False)
        case "tv-fista":
            return TVFISTAReconstructor(n_iter=100, verbose=False)
        case "wavelet-fista":
            return WaveletFISTAReconstructor(n_iter=100, verbose=False, device=device)
        case _:
            raise ValueError(f"Unknown algorithm {name!r}")


@pytest.fixture
def device():
    return "cpu"


@pytest.mark.parametrize("name", ALGORITHMS)
def test_reconstructors(name, device):
    """
    Test that reconstruction algorithms work end to end on a dummy example.
    """
    x = dinv.utils.load_example(
        "butterfly.png", img_size=(32, 32), grayscale=True, resize_mode="resize", device=device
    )
    x = torch.cat([x, torch.zeros_like(x)], dim=1)  # dummy complex data

    model = choose_algorithm(name, img_size=x.shape[1:], device=device)

    physics = DistortedKspaceMultiCoilMRI(
        img_size=(1, 2, *x.shape[-2:]), coil_maps=1, device=device
    )

    y = physics(x)

    x_hat = model(y, physics)

    assert x_hat.shape == x.shape


def test_fastmri_singlecoil_unet_reconstructor(device, tmp_path, monkeypatch):
    """
    Test the FastMRI UNet reconstructor end to end using local fixture weights.
    """
    monkeypatch.setattr(
        FastMRISinglecoilUnetReconstructor,
        "UNET_KWARGS",
        {
            "in_chans": 1,
            "out_chans": 1,
            "chans": 8,
            "num_pool_layers": 2,
            "drop_prob": 0.0,
        },
    )

    weights_path = tmp_path / "fastmri_unet_test_weights.pt"
    torch.save(Unet(**FastMRISinglecoilUnetReconstructor.UNET_KWARGS).state_dict(), weights_path)

    x = dinv.utils.load_example(
        "butterfly.png", img_size=(32, 32), grayscale=True, resize_mode="resize", device=device
    )
    x = torch.cat([x, torch.zeros_like(x)], dim=1)

    model = FastMRISinglecoilUnetReconstructor(
        device=device,
        state_dict_file=str(weights_path),
    )
    physics = DistortedKspaceMultiCoilMRI(
        img_size=(1, 2, *x.shape[-2:]), coil_maps=1, device=device
    )

    y = physics(x)
    x_hat = model(y, physics)

    assert x_hat.shape == x.shape
    assert torch.allclose(x_hat[:, 1], torch.zeros_like(x_hat[:, 1]))

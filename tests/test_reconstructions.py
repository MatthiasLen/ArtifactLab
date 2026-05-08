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
    OASISSinglecoilUnetReconstructor,
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


def test_oasis_singlecoil_unet_reconstructor_loads_lightning_checkpoint(device, tmp_path):
    """
    Test the OASIS UNet reconstructor with a Lightning-style checkpoint.
    """
    weights_path = tmp_path / "oasis_unet_test_weights.ckpt"
    state_dict = Unet(**OASISSinglecoilUnetReconstructor.UNET_KWARGS).state_dict()
    torch.save(
        {"state_dict": {f"unet.{key}": value for key, value in state_dict.items()}},
        weights_path,
    )

    x = dinv.utils.load_example(
        "butterfly.png", img_size=(32, 32), grayscale=True, resize_mode="resize", device=device
    )
    x = torch.cat([x, torch.zeros_like(x)], dim=1)

    model = OASISSinglecoilUnetReconstructor(
        checkpoint_file=str(weights_path),
        device=device,
    )
    physics = DistortedKspaceMultiCoilMRI(
        img_size=(1, 2, *x.shape[-2:]), coil_maps=1, device=device
    )

    y = physics(x)
    x_hat = model(y, physics)

    assert x_hat.shape == x.shape
    assert torch.allclose(x_hat[:, 1], torch.zeros_like(x_hat[:, 1]))


def test_oasis_resolve_default_checkpoint_downloads_manifest_and_checkpoint(tmp_path, monkeypatch):
    asset_root = tmp_path / "reconstruction_only"
    manifest_path = asset_root / "checkpoints" / "manifest.json"
    checkpoint_path = asset_root / "checkpoints" / "oasis_balanced_seed24_accel4.ckpt"
    manifest_bytes = (
        b'{"checkpoints": {"4": {"filename": "checkpoints/oasis_balanced_seed24_accel4.ckpt"}}}'
    )
    checkpoint_bytes = b"fake-oasis-checkpoint"
    downloads = []

    monkeypatch.setattr(
        OASISSinglecoilUnetReconstructor,
        "MANIFEST_SHA256",
        __import__("hashlib").sha256(manifest_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        OASISSinglecoilUnetReconstructor,
        "CHECKPOINT_SHA256",
        {"4": __import__("hashlib").sha256(checkpoint_bytes).hexdigest()},
    )
    monkeypatch.setattr(
        OASISSinglecoilUnetReconstructor,
        "CHECKPOINT_FILE_IDS",
        {"4": "checkpoint-file-id"},
    )

    def fake_download(file_id, destination, expected_sha256, **kwargs):
        downloads.append((file_id, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if file_id == "1zefZh7Vh5k2ssXKpLxV3Xnwf3S6dqu6I":
            destination.write_bytes(manifest_bytes)
        elif file_id == "checkpoint-file-id":
            destination.write_bytes(checkpoint_bytes)
        else:
            raise AssertionError(f"Unexpected file_id: {file_id}")

    monkeypatch.setattr(
        "mri_recon.reconstruction.deep.download_google_drive_file_with_sha256",
        fake_download,
    )

    resolved = OASISSinglecoilUnetReconstructor.resolve_default_checkpoint(
        4, manifest_path=manifest_path
    )

    assert resolved == checkpoint_path.resolve()
    assert manifest_path.read_bytes() == manifest_bytes
    assert checkpoint_path.read_bytes() == checkpoint_bytes
    assert downloads == [
        ("1zefZh7Vh5k2ssXKpLxV3Xnwf3S6dqu6I", manifest_path.resolve()),
        ("checkpoint-file-id", checkpoint_path.resolve()),
    ]


def test_oasis_singlecoil_unet_reconstructor_uses_packaged_checkpoint_defaults(
    device, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        OASISSinglecoilUnetReconstructor,
        "UNET_KWARGS",
        {
            "in_chans": 1,
            "out_chans": 1,
            "chans": 8,
            "num_pool_layers": 2,
            "drop_prob": 0.0,
        },
    )

    weights_path = tmp_path / "oasis_unet_packaged_weights.ckpt"
    state_dict = Unet(**OASISSinglecoilUnetReconstructor.UNET_KWARGS).state_dict()
    torch.save(
        {"state_dict": {f"unet.{key}": value for key, value in state_dict.items()}},
        weights_path,
    )

    captured = {}

    def fake_resolve(cls, acceleration, manifest_path=None):
        captured["acceleration"] = acceleration
        captured["manifest_path"] = manifest_path
        return weights_path

    monkeypatch.setattr(
        OASISSinglecoilUnetReconstructor,
        "resolve_default_checkpoint",
        classmethod(fake_resolve),
    )

    model = OASISSinglecoilUnetReconstructor(
        acceleration=8,
        manifest_path=str(tmp_path / "manifest.json"),
        device=device,
    )

    assert captured == {
        "acceleration": 8,
        "manifest_path": tmp_path / "manifest.json",
    }
    assert isinstance(model.model, Unet)

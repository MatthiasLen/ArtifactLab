import json
from pathlib import Path
from typing import Optional

import deepinv as dinv
import torch

from ._fastmri_unet import Unet
from ..utils import (
    download_file_with_sha256,
    download_google_drive_file_with_sha256,
    matches_sha256,
)


class RAMReconstructor(dinv.models.Reconstructor):
    """
    Wrapper for RAM from DeepInverse.
    Normalises input by magnitude of adjoint.

    :param float default_sigma: default sigma for RAM input. Overriden if physics already has a sigma (e.g. in a Gaussian noise model) at inference time.
    """

    def __init__(self, default_sigma=0.05, device: torch.device = None) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.model = dinv.models.RAM(device=device)
        self.default_sigma = default_sigma

    def forward(self, y, physics):
        _x_adj = physics.A_adjoint(y)
        scale = torch.quantile(_x_adj, 0.99)

        physics_norm = physics.compute_norm(torch.randn_like(_x_adj)).item()
        physics_adjointness = physics.adjointness_test(torch.randn_like(_x_adj)).item()

        if physics_norm > 1.2 or physics_norm < 0.8:
            raise ValueError(
                f"RAM reconstructor requires physics norm = 1 but got {physics_norm:.4f}"
            )
        if physics_adjointness > 0.1 or physics_adjointness < -0.1:
            raise ValueError(
                f"RAM reconstructor requires physics adjointness = 0 but got {physics_adjointness:.4f}"
            )

        sigma = (
            None
            if hasattr(physics, "noise_model") and hasattr(physics.noise_model, "sigma")
            else self.default_sigma
        )

        with torch.no_grad():
            return self.model(y / scale, physics, sigma=sigma) * scale


class DeepImagePriorReconstructor(dinv.models.Reconstructor):
    """
    Wrapper for Deep Image Prior from DeepInverse.

    :param tuple img_size: image size of the output. Defaults to (640, 368)
    :param int n_iter: number of iterations to fit the DIP. Defaults to 100.
    """

    def __init__(
        self,
        img_size: tuple = (640, 368),
        n_iter: int = 100,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        lr = 1e-2  # learning rate for the optimizer.
        channels = 64  # number of channels per layer in the decoder.
        in_size = [2, 2]  # size of the input to the decoder.

        self.model = dinv.models.DeepImagePrior(
            dinv.models.ConvDecoder(
                img_size=(2, *img_size[-2:]), in_size=in_size, channels=channels
            ),
            learning_rate=lr,
            iterations=n_iter,
            verbose=verbose,
            input_size=[channels] + in_size,
        )

    def forward(self, y, physics):
        return self.model(y, physics)


class FastMRISinglecoilUnetReconstructor(dinv.models.Reconstructor):
    """
    Wrapper for pretrained UNet from FastMRI singlecoil knee challenge.

    Note: this model was trained for accelerated MRI reconstruction and may not have good performance on other degradations.

    Note: this model discards complex information and only returns the magnitude image.

    NOTE: this model was trained on both train+val splits of the challenge (i.e. trained on singlecoil_train, singlecoil_val).

    The pretrained fastMRI model expects magnitude images that are normalized per slice,
    so this wrapper matches that preprocessing and rescales the output back to the
    original adjoint-image intensity range.

    See https://github.com/facebookresearch/fastMRI/tree/main/fastmri_examples for more details.
    """

    MODEL_URL = (
        "https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/"
        "knee_sc_leaderboard_state_dict.pt"
    )
    MODEL_SHA256 = "8f41f67d8eab2cca31ffff632a733a8712b1171c11f13e95b6f90fdf63399f9e"
    MODEL_FILENAME = "knee_sc_leaderboard_state_dict.pt"
    UNET_KWARGS = {
        "in_chans": 1,
        "out_chans": 1,
        "chans": 256,
        "num_pool_layers": 4,
        "drop_prob": 0.0,
    }

    def __init__(self, device: torch.device = None, state_dict_file: str = None) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.model = Unet(**self.UNET_KWARGS)

        state_dict_path = (
            Path(state_dict_file).expanduser()
            if state_dict_file is not None
            else Path(__file__).resolve().parents[2] / self.MODEL_FILENAME
        )

        if state_dict_file is None:
            if not matches_sha256(state_dict_path, self.MODEL_SHA256):
                download_file_with_sha256(
                    self.MODEL_URL,
                    state_dict_path,
                    self.MODEL_SHA256,
                    label="FastMRI UNet checkpoint",
                )
        elif not state_dict_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {state_dict_path}")

        self.model.load_state_dict(
            torch.load(state_dict_path, map_location=device, weights_only=True)
        )
        self.model.eval()
        self.model.to(device)

    def forward(self, y: torch.Tensor, physics: dinv.physics.Physics) -> torch.Tensor:
        x_in = physics.A_adjoint(y)

        x_in = dinv.utils.complex_abs(x_in, keepdim=True)

        # Match the fastMRI normalization used for training, then rescale the
        # predicted magnitude image back to the original adjoint-image intensity range.
        mu = x_in.mean(dim=(-2, -1), keepdim=True)
        std = x_in.std(dim=(-2, -1), keepdim=True) + 1e-8
        x_in = (x_in - mu) / std

        with torch.no_grad():
            out = self.model(x_in) * std + mu  # (B, 1, H, W)

        return torch.cat([out, torch.zeros_like(out)], dim=1)


def _load_unet_checkpoint_state(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load U-Net weights from a plain or Lightning-style checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Accept either a checkpoint with "state_dict" or a plain state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # If Lightning-style keys like 'unet.*' exist, strip the 'unet.' prefix
    if any(key.startswith("unet.") for key in state_dict):
        return {
            key[len("unet.") :]: value
            for key, value in state_dict.items()
            if key.startswith("unet.")
        }

    return state_dict


class OASISSinglecoilUnetReconstructor(dinv.models.Reconstructor):
    """Wrapper for a trained OASIS single-coil U-Net model.

    The model reuses the repository's fastMRI-derived :class:`Unet` module, but
    can also download the packaged checkpoint manifest and checkpoint on demand
    when no explicit checkpoint path is supplied. The forward pass converts
    k-space to a zero-filled magnitude image, applies per-slice instance
    normalization, runs the U-Net, then rescales the prediction back to the
    adjoint-image intensity range.

    Parameters
    ----------
    checkpoint_file : str, optional
        Path to the trained OASIS U-Net checkpoint. If omitted, the reconstructor
        downloads the packaged checkpoint for ``acceleration``.
    acceleration : int, optional
        Packaged checkpoint acceleration factor used when ``checkpoint_file`` is omitted.
    manifest_path : str, optional
        Override path for the downloaded or cached packaged checkpoint manifest.
    device : torch.device, optional
        Device on which to run inference.
    """

    UNET_KWARGS = {
        "in_chans": 1,
        "out_chans": 1,
        "chans": 32,
        "num_pool_layers": 4,
        "drop_prob": 0.0,
    }
    ASSET_ROOT = Path(__file__).resolve().parents[2] / "reconstruction_only"
    CHECKPOINTS_DIR = ASSET_ROOT / "checkpoints"
    MANIFEST_PATH = CHECKPOINTS_DIR / "manifest.json"
    MANIFEST_FILE_ID = "1zefZh7Vh5k2ssXKpLxV3Xnwf3S6dqu6I"
    MANIFEST_SHA256 = "d5180c49fcaafe7ba439319dcf4afe4d7489473bea437418d836070ecd506952"
    CHECKPOINT_FILE_IDS = {
        "4": "11s6YeM6_YJeD4wcrn24jyMjyj_vX2ANU",
        "8": "1w8PDiYpr2xBPXahzRllhZjQT1yoMGXg-",
        "10": "1djJ2i0uYP4PT070CS0xx9nNJ41JmSFhh",
    }
    CHECKPOINT_SHA256 = {
        "4": "4fcefa9860cb7895e581a0de8f90bd7f188ae1c0b5e428a4a07519dd2561ac29",
        "8": "2cd4c44e3c7a3870adbe5090b2bfaae044f5e3f0b4bcaf2b1fc29969e5e6b9ca",
        "10": "90e3d9b17aa0f9fd43aaf090c152edcbdead1b9be41a076594e41098db7befa8",
    }

    @classmethod
    def ensure_manifest(cls, manifest_path: Optional[Path] = None) -> Path:
        """Ensure the packaged OASIS checkpoint manifest exists locally and is verified."""

        resolved_manifest_path = (
            manifest_path.expanduser().resolve() if manifest_path is not None else cls.MANIFEST_PATH
        )
        if not matches_sha256(resolved_manifest_path, cls.MANIFEST_SHA256):
            download_google_drive_file_with_sha256(
                cls.MANIFEST_FILE_ID,
                resolved_manifest_path,
                cls.MANIFEST_SHA256,
                label="OASIS checkpoint manifest",
            )
        return resolved_manifest_path

    @classmethod
    def resolve_default_checkpoint(
        cls,
        acceleration: int,
        manifest_path: Optional[Path] = None,
    ) -> Path:
        """Resolve and download the packaged OASIS checkpoint for a given acceleration."""

        resolved_manifest_path = cls.ensure_manifest(manifest_path)
        with resolved_manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        key = str(acceleration)
        checkpoints = manifest.get("checkpoints", {})
        if key not in checkpoints:
            available = ", ".join(sorted(checkpoints))
            raise ValueError(
                f"No packaged checkpoint for acceleration {acceleration}. Available: {available}."
            )

        if key not in cls.CHECKPOINT_FILE_IDS or key not in cls.CHECKPOINT_SHA256:
            raise ValueError(
                f"No automated download metadata is configured for acceleration {acceleration}."
            )

        filename = Path(checkpoints[key]["filename"])
        checkpoint_path = (
            filename
            if filename.is_absolute()
            else (resolved_manifest_path.parent.parent / filename)
        ).resolve()

        if not matches_sha256(checkpoint_path, cls.CHECKPOINT_SHA256[key]):
            download_google_drive_file_with_sha256(
                cls.CHECKPOINT_FILE_IDS[key],
                checkpoint_path,
                cls.CHECKPOINT_SHA256[key],
                label=f"OASIS checkpoint x{acceleration}",
            )

        return checkpoint_path

    def __init__(
        self,
        checkpoint_file: str | None = None,
        acceleration: int = 4,
        manifest_path: str | None = None,
        device: torch.device = None,
    ) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cpu")
        self.device = device

        if checkpoint_file is None:
            resolved_manifest_path = (
                Path(manifest_path).expanduser() if manifest_path is not None else None
            )
            checkpoint_path = self.resolve_default_checkpoint(
                acceleration=acceleration,
                manifest_path=resolved_manifest_path,
            )
        else:
            checkpoint_path = Path(checkpoint_file).expanduser()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Use the helper to obtain a normalized state_dict (handles plain or Lightning)
        state_dict = _load_unet_checkpoint_state(checkpoint_path, device)

        self.model = Unet(**self.UNET_KWARGS)
        self.model.load_state_dict(
            state_dict,
            strict=True,
        )
        self.model.eval()
        self.model.to(device)

    def forward(self, y: torch.Tensor, physics: dinv.physics.Physics) -> torch.Tensor:
        """Reconstruct a magnitude image from measured k-space.

        Parameters
        ----------
        y : torch.Tensor
            Measured k-space tensor.
        physics : dinv.physics.Physics
            Physics operator used to form the adjoint input image.

        Returns
        -------
        torch.Tensor
            Complex-valued reconstruction with zero imaginary channel.
        """

        x_in = dinv.utils.complex_abs(physics.A_adjoint(y), keepdim=True)
        mu = x_in.mean(dim=(-2, -1), keepdim=True)
        std = x_in.std(dim=(-2, -1), keepdim=True) + 1e-11
        x_in = (x_in - mu) / std

        with torch.no_grad():
            out = self.model(x_in) * std + mu

        return torch.cat([out, torch.zeros_like(out)], dim=1)

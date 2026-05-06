import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen

import deepinv as dinv
import torch

from ._fastmri_unet import Unet


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
            if not self._matches_sha256(state_dict_path, self.MODEL_SHA256):
                self._download_model(self.MODEL_URL, state_dict_path, self.MODEL_SHA256)
        elif not state_dict_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {state_dict_path}")

        self.model.load_state_dict(
            torch.load(state_dict_path, map_location=device, weights_only=True)
        )
        self.model.eval()
        self.model.to(device)

    @staticmethod
    def _matches_sha256(path: Path, expected_sha256: str) -> bool:
        if not path.exists():
            return False

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == expected_sha256

    @classmethod
    def _download_model(cls, url: str, fname: Path, expected_sha256: str) -> None:
        fname.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=fname.parent, suffix=".tmp"
        ) as handle:
            tmp_path = Path(handle.name)

            try:
                with urlopen(url, timeout=30) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(chunk)

                if not cls._matches_sha256(tmp_path, expected_sha256):
                    raise ValueError(f"Downloaded checkpoint failed SHA256 verification: {fname}")

                tmp_path.replace(fname)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

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

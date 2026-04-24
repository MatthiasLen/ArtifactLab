import torch
import deepinv as dinv


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


import torch
import requests
from pathlib import Path
from tqdm import tqdm
from .unet import Unet


class FastMRISinglecoilUnetReconstructor(dinv.models.Reconstructor):
    """
    Wrapper for pretrained singlecoil UNet from FastMRI challenge.
    """

    MODEL_URL = (
        "https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/"
        "knee_sc_leaderboard_state_dict.pt"
    )
    STATE_DICT_FNAME = "knee_sc_leaderboard_state_dict.pt"

    def __init__(self, device: torch.device = None, state_dict_file: str = None) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.model = Unet(
            in_chans=1,
            out_chans=1,
            chans=256,
            num_pool_layers=4,
            drop_prob=0.0,
        )

        if state_dict_file is None:
            state_dict_file = self.STATE_DICT_FNAME
            if not Path(state_dict_file).exists():
                self._download_model(self.MODEL_URL, state_dict_file)

        self.model.load_state_dict(
            torch.load(state_dict_file, map_location=device)
        )
        self.model.eval()
        self.model.to(device)

    @staticmethod
    def _download_model(url: str, fname: str):
        response = requests.get(url, timeout=10, stream=True)
        total = int(response.headers.get("content-length", 0))
        with open(fname, "wb") as fh, tqdm(
            desc="Downloading UNet weights", total=total, unit="iB", unit_scale=True
        ) as bar:
            for chunk in response.iter_content(1024 * 1024):
                fh.write(chunk)
                bar.update(len(chunk))

    def forward(self, y: torch.Tensor, physics: dinv.physics.Physics):
        # y: (B, 2, H, W) real dtype, singlecoil k-space (real/imag)
        x_in = physics.A_adjoint(y)
        # x_in: (B, 2, H, W) real/imag image domain

        # Compute magnitude: (B, 1, H, W)
        magnitude = torch.sqrt(x_in[:, 0:1] ** 2 + x_in[:, 1:2] ** 2)

        # Normalise per sample (matches FastMRI UnetDataTransform)
        mean = magnitude.mean(dim=(-2, -1), keepdim=True)
        std = magnitude.std(dim=(-2, -1), keepdim=True) + 1e-8
        magnitude_norm = (magnitude - mean) / std

        with torch.no_grad():
            out_norm = self.model(magnitude_norm)  # (B, 1, H, W)

        # Unnormalise
        out_magnitude = out_norm * std + mean  # (B, 1, H, W)

        # Return as (B, 2, H, W) with zero imaginary channel
        out = torch.cat([out_magnitude, torch.zeros_like(out_magnitude)], dim=1)

        return out
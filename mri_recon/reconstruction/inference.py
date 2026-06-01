from __future__ import annotations

import deepinv as dinv
import torch

from .classic import (
    ConjugateGradientReconstructor,
    TVFISTAReconstructor,
    TVPDHGReconstructor,
    TVPGDReconstructor,
    WaveletFISTAReconstructor,
    ZeroFilledReconstructor,
)
from .deep import (
    DeepImagePriorReconstructor,
    FastMRISinglecoilUnetReconstructor,
    OASISSinglecoilUnetReconstructor,
    RAMReconstructor,
)

FASTMRI_UNET_ALGORITHM = "unet-fastmri"
OASIS_UNET_ALGORITHMS = {
    "unet-oasis-acceleration4": 4,
    "unet-oasis-acceleration8": 8,
    "unet-oasis-acceleration10": 10,
}
EXPLICIT_UNET_ALGORITHMS = (FASTMRI_UNET_ALGORITHM, *tuple(OASIS_UNET_ALGORITHMS))


def uses_oasis_centered_path(
    dataset: str,
    algorithm: str,
) -> bool:
    """Return whether inference should use the centered OASIS k-space path.

    OASIS samples always use the centered FFT convention. FastMRI only switches
    to that path when the selected algorithm is one of the explicit OASIS U-Net
    variants.
    """

    if dataset == "oasis":
        return True
    return algorithm in OASIS_UNET_ALGORITHMS


def compatible_dataset_with_reconstructor(dataset: str, reconstructor_name: str) -> bool:
    """Check if dataset and trained reconstructor are compatible"""

    # fast mri u-net is only trained with knee data
    if reconstructor_name == FASTMRI_UNET_ALGORITHM:
        if dataset == "fastmri_knee":
            return True
        else:
            return False

    # oasis is only trained with brain data
    elif reconstructor_name in OASIS_UNET_ALGORITHMS:
        if dataset in ["fastmri_brain", "oasis"]:
            return True
        else:
            return False

    # all other (classic) reconstructors work with any dataset:
    else:
        return True


def choose_reconstructor(
    name: str,
    img_size: tuple = (640, 368),
    device: torch.device | str = "cpu",
    verbose: bool = False,
    dataset: str | None = None,
) -> dinv.models.Reconstructor:
    """Create a reconstructor while enforcing the supported dataset/model matrix.

    Parameters
    ----------
    name : str
        High-level algorithm identifier used by the example entry point.
    img_size : tuple, optional
        Spatial image size used by reconstructors that need an explicit image
        shape, such as DIP.
    device : torch.device | str, optional
        Device on which to instantiate the reconstructor.
    verbose : bool, optional
        Forwarded to reconstructors that expose a verbose mode.
    dataset : str, optional
        Dataset being evaluated. This only affects compatibility checks for
        explicit algorithm names that are dataset-specific.
    """

    if dataset is not None and not compatible_dataset_with_reconstructor(dataset, name):
        raise ValueError(
            f"Reconstructor {name} is not compatible with dataset {dataset}, because it was trained with a different image domain."
        )

    match name:
        case "zero-filled":
            return ZeroFilledReconstructor()
        case "conjugate-gradient":
            return ConjugateGradientReconstructor(max_iter=20)
        case "ram":
            return RAMReconstructor(default_sigma=0.05, device=device)
        case "dip":
            return DeepImagePriorReconstructor(img_size=img_size, n_iter=100, verbose=verbose)
        case "tv-pgd":
            return TVPGDReconstructor(n_iter=100, verbose=verbose)
        case "tv-fista":
            return TVFISTAReconstructor(n_iter=200, verbose=verbose)
        case "tv-pdhg":
            return TVPDHGReconstructor(n_iter=100, verbose=verbose)
        case "wavelet-fista":
            return WaveletFISTAReconstructor(n_iter=100, device=device, verbose=verbose)
        case _ if name == FASTMRI_UNET_ALGORITHM:
            return FastMRISinglecoilUnetReconstructor(device=device)
        case _ if name in OASIS_UNET_ALGORITHMS:
            return OASISSinglecoilUnetReconstructor(
                acceleration=OASIS_UNET_ALGORITHMS[name],
                device=device,
            )
        case _:
            raise ValueError(f"Unknown algorithm {name!r}")

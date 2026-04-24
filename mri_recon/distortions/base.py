import torch
import deepinv as dinv


def _frequency_grids(shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fftshifted Cartesian frequency grids in cycles/pixel."""

    ny, nx = shape[-2:]
    kx = torch.fft.fftshift(torch.fft.fftfreq(nx))
    ky = torch.fft.fftshift(torch.fft.fftfreq(ny))
    return torch.meshgrid(kx, ky, indexing="xy")


def _radial_frequency(shape: tuple[int, ...]) -> torch.Tensor:
    """Return radial frequency normalized to ``[0, 1]`` on the sampled grid."""

    kx, ky = _frequency_grids(shape)
    radius = torch.sqrt(kx * kx + ky * ky)
    max_radius = float(torch.max(radius))
    if max_radius <= 0.0:
        return torch.zeros_like(radius)
    return radius / max_radius


def _normalized_axis_frequencies(
    shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return absolute Cartesian frequencies normalized to ``[0, 1]`` per axis."""

    kx, ky = _frequency_grids(shape)
    max_abs_kx = float(torch.max(torch.abs(kx)))
    max_abs_ky = float(torch.max(torch.abs(ky)))

    if max_abs_kx <= 0.0:
        normalized_kx = torch.zeros_like(kx)
    else:
        normalized_kx = torch.abs(kx) / max_abs_kx

    if max_abs_ky <= 0.0:
        normalized_ky = torch.zeros_like(ky)
    else:
        normalized_ky = torch.abs(ky) / max_abs_ky

    return normalized_kx, normalized_ky


def _normalized_frequency_grids(
    shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Cartesian frequency grids normalized by the sampled max radius."""

    kx, ky = _frequency_grids(shape)
    max_radius = torch.sqrt(kx * kx + ky * ky).max()
    if float(max_radius) <= 0.0:
        return torch.zeros_like(kx), torch.zeros_like(ky)
    return kx / max_radius, ky / max_radius


def _validate_cartesian_kspace_tensor(y: torch.Tensor) -> None:
    """Validate the repository's 2D Cartesian k-space tensor convention."""

    if y.ndim not in (4, 5):
        raise ValueError(
            "Expected k-space with shape (B, 2, H, W) or (B, 2, N, H, W), "
            f"got tensor with shape {tuple(y.shape)}"
        )
    if y.shape[1] != 2:
        raise ValueError(
            "Expected real/imaginary channel dimension of size 2 at axis 1, "
            f"got shape {tuple(y.shape)}"
        )
    if not torch.is_floating_point(y):
        raise TypeError(f"Expected floating-point real/imaginary tensor, got dtype {y.dtype}")
    if y.shape[-2] <= 0 or y.shape[-1] <= 0:
        raise ValueError(f"Spatial k-space dimensions must be positive, got shape {tuple(y.shape)}")


class BaseDistortion(dinv.physics.LinearPhysics):
    """Base class for deterministic k-space distortions.

    This class represents distortions applied directly to measured k-space. By
    default, it acts as the identity operator.
    """

    def A(self, y: torch.Tensor) -> torch.Tensor:
        """Distortion forward pass."""
        return y

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the distortion's adjoint operation. If the distortion is elementwise, this will be equal to apply."""
        # @Andrewwango is this assuming that the operator is self-adjoint if no override is provided?
        # I think I read somewhere that deep inverse is approximating the adjoint is not given.
        # Hence, shall we remove the default implementation of A_adjoint in the base class ?
        return self.A(y)


class DistortedKspaceMultiCoilMRI(dinv.physics.MultiCoilMRI):
    r"""
    Multi-coil MRI with additional kspace distortion.

    Multi-coil 2D or 3D MRI operator.

    The linear operator operates in 2D slices or 3D volumes and is defined as:

    .. math::

        y_n = \text{diag}(p) F \text{diag}(s_n) x

    for :math:`n=1,\dots,N` coils, where :math:`y_n` are the measurements from the cth coil, :math:`\text{diag}(p)` is the acceleration mask, :math:`F` is the Fourier transform and :math:`\text{diag}(s_n)` is the nth coil sensitivity.

    The data ``x`` should be of shape (B,C,H,W) or (B,C,D,H,W) where C=2 is the channels (real and imaginary) and D is optional dimension for 3D MRI.
    Then, the resulting measurements ``y`` will be of shape (B,C,N,(D,)H,W) where N is the coils dimension.

    :param torch.Tensor mask: binary sampling mask which should have shape (H,W), (C,H,W), (B,C,H,W), or (B,C,...,H,W). If None, generate mask of ones with ``img_size``.
    :param torch.Tensor, str coil_maps: either ``Tensor``, integer, or ``None``. If complex valued (i.e. of complex dtype) coil sensitvity maps which should have shape (H,W), (N,H,W), (B,N,H,W) or (B,N,...,H,W).
        If None, generate flat coil maps of ones with ``img_size``. If integer, simulate birdcage coil maps with integer number of coils (this requires ``sigpy`` installed).
    :param tuple img_size: if ``mask`` or ``coil_maps`` not specified, flat ``mask`` or ``coil_maps`` of ones are created using ``img_size``,
        where ``img_size`` can be of any shape specified above. If ``mask`` or ``coil_maps`` provided, ``img_size`` is ignored.
    :param bool three_d: if ``True``, calculate Fourier transform in 3D for 3D data (i.e. data of shape (B,C,D,H,W) where D is depth).
    :param torch.device, str device: specify which device you want to use (i.e, cpu or gpu).
    """

    def __init__(self, distortion: BaseDistortion = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if distortion is None:
            distortion = BaseDistortion()
        self.distortion = distortion

    def A(self, x: torch.Tensor) -> torch.Tensor:
        y = super().A(x)
        y = y.squeeze(2)  # remove coil dim if singlecoil
        return self.distortion(y)

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        if len(y.shape) == (5 if self.three_d else 4):
            y = y.unsqueeze(2)  # add coil dim if singlecoil

        y = self.distortion.A_adjoint(y)
        return super().A_adjoint(y)

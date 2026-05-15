"""OASIS dataset and centered FFT adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mri_recon.distortions import BaseDistortion


class OasisSliceDataset(Dataset):
    """Load 2D OASIS slices from Analyze/NIfTI volumes.

    Parameters
    ----------
    data_path : Path
        Root directory containing OASIS subject folders.
    split_csv : Path
        CSV file listing OASIS subjects and slice counts.
    sample_rate : float, optional
        Fraction of slices to keep from each volume. Values below ``1`` keep a
        centered slice range. Defaults to ``1``.
    """

    def __init__(
        self,
        data_path: Path,
        split_csv: Path,
        sample_rate: float = 1.0,
    ) -> None:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError(
                "OASIS loading requires nibabel. Install the project dependencies "
                "or add nibabel to your environment before using OasisSliceDataset."
            ) from exc

        self._nib = nib
        self.data_path = Path(data_path)
        self.split_csv = Path(split_csv)
        if not 0 < sample_rate <= 1.0:
            raise ValueError("sample_rate must be in the range (0, 1].")
        self.sample_rate = sample_rate
        self.subject_paths = self._discover_subject_paths()
        self.raw_samples = self._create_sample_list()

    def __len__(self) -> int:
        """Return the number of available slices."""

        return len(self.raw_samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one complex-valued OASIS slice in repo tensor convention."""

        subject_id, slice_num = self.raw_samples[index]
        volume = self._get_volume(subject_id)
        target_np = np.ascontiguousarray(volume[slice_num], dtype=np.float32)
        real = torch.from_numpy(target_np)
        x = torch.stack([real, torch.zeros_like(real)], dim=0)
        return {"x": x.float(), "subject_id": subject_id, "slice_num": slice_num}

    def _discover_subject_paths(self) -> dict[str, Path]:
        subject_paths = {}
        for subject_dir in sorted(self.data_path.iterdir()):
            if not subject_dir.is_dir():
                continue
            image_glob = subject_dir / "PROCESSED" / "MPRAGE" / "T88_111"
            matches = sorted(image_glob.glob("*t88_gfc.img"))
            if matches:
                subject_paths[subject_dir.name] = matches[0]

        if not subject_paths:
            raise FileNotFoundError(
                "Could not find OASIS subject folders under "
                f"{self.data_path} matching PROCESSED/MPRAGE/T88_111/*t88_gfc.img."
            )
        return subject_paths

    def _create_sample_list(self) -> list[tuple[str, int]]:
        samples = []
        rows = []
        with self.split_csv.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = [item.strip() for item in line.split(",")]
                if not row or not row[0]:
                    continue
                try:
                    rows.append((row[0], int(row[-1])))
                except ValueError:
                    continue

        for subject_id, num_slices in rows:
            if subject_id not in self.subject_paths:
                raise FileNotFoundError(
                    f"Could not find OASIS subject {subject_id!r} from split CSV under "
                    f"{self.data_path}."
                )
            mid = round(num_slices / 2)
            half_span = round(num_slices * self.sample_rate / 2)
            start = 0 if self.sample_rate >= 1.0 else max(0, mid - half_span)
            stop = num_slices if self.sample_rate >= 1.0 else min(num_slices, mid + half_span)
            for slice_num in range(start, stop):
                samples.append((subject_id, slice_num))
        return samples

    def _num_slices(self, image_path: Path) -> int:
        shape = tuple(dim for dim in self._nib.load(str(image_path)).shape if dim != 1)
        if len(shape) < 2:
            raise ValueError(f"Expected at least 2D OASIS image, got shape {shape}.")
        return shape[1]

    def _get_volume(self, subject_id: str) -> np.ndarray:
        image_data = self._nib.load(str(self.subject_paths[subject_id])).get_fdata(dtype=np.float32)
        volume = np.ascontiguousarray(
            np.transpose(np.squeeze(image_data), (1, 0, 2)),
            dtype=np.float32,
        )
        return volume


def image_to_kspace(x: torch.Tensor) -> torch.Tensor:
    """Convert channel-first complex images to centered k-space.

    Parameters
    ----------
    x : torch.Tensor
        Complex image tensor with shape ``(B, 2, H, W)``.

    Returns
    -------
    torch.Tensor
        Centered k-space tensor with shape ``(B, 2, H, W)``.
    """

    x_complex = torch.view_as_complex(x.movedim(1, -1).contiguous())
    y_complex = torch.fft.fftshift(
        torch.fft.fft2(x_complex, dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )
    return torch.view_as_real(y_complex).movedim(-1, 1).contiguous()


def kspace_to_image(y: torch.Tensor) -> torch.Tensor:
    """Convert centered channel-first k-space to complex images.

    Parameters
    ----------
    y : torch.Tensor
        Centered k-space tensor with shape ``(B, 2, H, W)``.

    Returns
    -------
    torch.Tensor
        Complex image tensor with shape ``(B, 2, H, W)``.
    """

    y_complex = torch.view_as_complex(y.movedim(1, -1).contiguous())
    x_complex = torch.fft.ifft2(
        torch.fft.ifftshift(y_complex, dim=(-2, -1)),
        dim=(-2, -1),
        norm="ortho",
    )
    return torch.view_as_real(x_complex).movedim(-1, 1).contiguous()


class OasisCenteredFFTPhysics:
    """Physics adapter matching the OASIS U-Net FFT convention.

    Parameters
    ----------
    distortion : BaseDistortion
        K-space distortion applied after the centered FFT.
    """

    def __init__(self, distortion: BaseDistortion) -> None:
        self.distortion = distortion

    def A(self, x: torch.Tensor) -> torch.Tensor:
        """Apply centered FFT and k-space distortion.

        Parameters
        ----------
        x : torch.Tensor
            Complex image tensor with shape ``(B, 2, H, W)``.

        Returns
        -------
        torch.Tensor
            Distorted centered k-space tensor.
        """

        return self.distortion.A(image_to_kspace(x))

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint distortion and centered inverse FFT.

        Parameters
        ----------
        y : torch.Tensor
            Distorted centered k-space tensor.

        Returns
        -------
        torch.Tensor
            Complex image tensor with shape ``(B, 2, H, W)``.
        """

        return kspace_to_image(self.distortion.A_adjoint(y))

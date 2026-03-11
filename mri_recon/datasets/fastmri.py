"""FastMRI dataset implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseDataset

try:
    import h5py
except ImportError:  # pragma: no cover - exercised via runtime guard.
    h5py = None


class FastMRIDataset(BaseDataset):
    """Dataset wrapper for real FastMRI HDF5 volumes.

    The upstream fastMRI project stores one acquisition per ``.h5`` file with
    slice-wise ``kspace`` data and optional ``mask`` and reconstruction targets.
    This class provides a compact interface around that format while keeping the
    project independent from the upstream PyTorch dataset wrapper.
    """

    sample_extension = ".h5"

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        challenge: str = "multicoil",
        target_key: str | None = None,
    ) -> None:
        super().__init__(root_dir=root_dir)
        if challenge not in {"singlecoil", "multicoil"}:
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        self.split = split
        self.challenge = challenge
        self.target_key = target_key or (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )

    def download(self, source: str | Path, destination: str | Path | None = None) -> Path:
        """Copy or download FastMRI data for the configured split."""

        target = Path(destination) if destination is not None else self.root_dir / self.split
        return super().download(source=source, destination=target)

    def get_sample_path(self, sample_id: str) -> Path:
        """Return the HDF5 path for a FastMRI volume."""

        sample_name = sample_id if sample_id.endswith(self.sample_extension) else (
            f"{sample_id}{self.sample_extension}"
        )
        return self.root_dir / self.split / sample_name

    def read_sample(self, sample_id: str, slice_index: int = 0) -> dict[str, Any]:
        """Read a FastMRI sample from disk.

        The returned dictionary mirrors the key parts of the upstream fastMRI
        slice dataset: one slice of ``kspace`` data, an optional Cartesian
        ``mask``, an optional reconstruction target, and per-volume metadata.
        """

        self._require_h5py()
        sample_path = self.get_sample_path(sample_id)
        if not sample_path.exists():
            raise FileNotFoundError(f"FastMRI sample does not exist: {sample_path}")

        with h5py.File(sample_path, "r") as handle:
            if "kspace" not in handle:
                raise ValueError("FastMRI volume is missing required dataset: kspace")

            num_slices = int(handle["kspace"].shape[0])
            if slice_index < 0 or slice_index >= num_slices:
                raise IndexError(
                    f"slice_index {slice_index} is out of range for {num_slices} slices"
                )

            target = None
            if self.target_key in handle:
                target = self._serialize_value(handle[self.target_key][slice_index])

            mask = None
            if "mask" in handle:
                mask = self._serialize_value(handle["mask"][()])

            metadata = {
                "num_slices": num_slices,
                "kspace_shape": tuple(int(dimension) for dimension in handle["kspace"].shape),
                "challenge": self.challenge,
                "split": self.split,
                "target_key": self.target_key if self.target_key in handle else None,
                **{key: self._serialize_value(value) for key, value in handle.attrs.items()},
            }
            if "ismrmrd_header" in handle:
                metadata["ismrmrd_header"] = self._serialize_value(handle["ismrmrd_header"][()])

            return {
                "sample_id": sample_path.stem,
                "filename": sample_path.name,
                "slice_index": slice_index,
                "kspace": self._serialize_value(handle["kspace"][slice_index]),
                "mask": mask,
                "target": target,
                "metadata": metadata,
            }

    def _require_h5py(self) -> None:
        """Ensure the optional HDF5 dependency is available."""

        if h5py is None:
            raise ImportError(
                "FastMRIDataset requires h5py. Install dependencies from "
                "requirements.txt before using this dataset."
            )

    def _serialize_value(self, value: Any) -> Any:
        """Convert HDF5-backed values into plain Python objects."""

        if isinstance(value, bytes):
            return value.decode("utf-8")
        if hasattr(value, "tolist"):
            return self._serialize_value(value.tolist())
        if isinstance(value, complex):
            return [float(value.real), float(value.imag)]
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._serialize_value(item) for item in value)
        return value

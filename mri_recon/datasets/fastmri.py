"""FastMRI dataset implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .base import BaseDataset

try:
    from fastmri.data import SliceDataset
except ImportError:  # pragma: no cover - exercised via runtime guard.
    SliceDataset = None


DEFAULT_FASTMRI_ROOT = Path.home() / ".cache" / "mri_recon" / "fastmri"


class FastMRIDataset(BaseDataset):
    """Small wrapper around ``fastmri.data.SliceDataset``."""

    supported_sample_extensions = (".h5", ".hdf5")

    def __init__(
        self,
        root_dir: str | Path | None = None,
        split: str = "train",
        challenge: str = "singlecoil",
        transform: Callable[..., Any] | None = None,
        use_dataset_cache: bool = False,
        sample_rate: float | None = None,
        volume_sample_rate: float | None = None,
        dataset_cache_file: str | Path = "dataset_cache.pkl",
        num_cols: tuple[int, ...] | None = None,
        raw_sample_filter: Callable[[Any], bool] | None = None,
        sample_url: str | Path | None = None,
        auto_download: bool = False,
    ) -> None:
        super().__init__(root_dir=root_dir or DEFAULT_FASTMRI_ROOT)
        if challenge not in {"singlecoil", "multicoil"}:
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.challenge = challenge
        self.transform = transform
        self.use_dataset_cache = use_dataset_cache
        self.sample_rate = sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.dataset_cache_file = dataset_cache_file
        self.num_cols = num_cols
        self.raw_sample_filter = raw_sample_filter
        self.sample_url = sample_url

        self.data_dir = self.root_dir / split
        self._slice_dataset: Any | None = None

        if self._has_local_samples():
            self._initialize_slice_dataset()
        elif auto_download:
            self.download()

    @property
    def slice_dataset(self) -> Any:
        self._ensure_slice_dataset()
        return self._slice_dataset

    def download(
        self,
        source: str | Path | None = None,
        destination: str | Path | None = None,
    ) -> Path:
        source_path = source or self.sample_url
        if source_path is None:
            raise ValueError("No source provided. Pass source=... or set sample_url.")

        target = Path(destination) if destination is not None else self.data_dir
        resolved = super().download(source=source_path, destination=target)
        self.data_dir = self._resolve_data_dir(resolved)
        self._initialize_slice_dataset()
        return resolved

    def get_sample_path(self, sample_id: str) -> Path:
        if sample_id.lower().endswith(self.supported_sample_extensions):
            return self.data_dir / sample_id
        return self.data_dir / f"{sample_id}.h5"

    def read_sample(self, sample_id: str, slice_index: int = 0) -> dict[str, Any]:
        wanted_id = Path(sample_id).stem
        for index, raw_sample in enumerate(self.slice_dataset.raw_samples):
            if raw_sample.fname.stem == wanted_id and raw_sample.slice_ind == slice_index:
                return self[index]

        raise FileNotFoundError(
            f"FastMRI sample {wanted_id!r} slice {slice_index} was not found in {self.data_dir}."
        )

    def to_numpy(self, sample: dict[str, Any], field: str = "target") -> Any:
        if field not in sample:
            raise KeyError(f"Sample does not contain field {field!r}")
        return np.asarray(sample[field])

    def sample_ids(self) -> list[str]:
        return list(dict.fromkeys(raw_sample.fname.stem for raw_sample in self.slice_dataset.raw_samples))

    def __len__(self) -> int:
        if self._slice_dataset is None and not self._has_local_samples():
            return 0
        return len(self.slice_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._normalize_slice_sample(self.slice_dataset[index])

    def _ensure_slice_dataset(self) -> None:
        if self._slice_dataset is not None:
            return
        if not self._has_local_samples():
            raise FileNotFoundError(
                f"No FastMRI volumes are available in {self.data_dir}. Call download() first "
                "or place .h5/.hdf5 files in that folder."
            )
        self._initialize_slice_dataset()

    def _initialize_slice_dataset(self) -> None:
        if SliceDataset is None:
            raise ImportError(
                "FastMRIDataset requires the fastmri package. Install CPU torch first and "
                "then install dependencies from requirements.txt."
            )

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._slice_dataset = SliceDataset(
            root=self.data_dir,
            challenge=self.challenge,
            transform=self.transform,
            use_dataset_cache=self.use_dataset_cache,
            sample_rate=self.sample_rate,
            volume_sample_rate=self.volume_sample_rate,
            dataset_cache_file=self.dataset_cache_file,
            num_cols=self.num_cols,
            raw_sample_filter=self.raw_sample_filter,
        )

    def _has_local_samples(self) -> bool:
        if not self.data_dir.exists():
            return False
        return any(self._is_volume_file(path) for path in self.data_dir.iterdir())

    def _resolve_data_dir(self, path: Path) -> Path:
        if path.is_file():
            if not self._is_volume_file(path):
                raise FileNotFoundError(f"FastMRI source file is not an HDF5 volume: {path}")
            return path.parent

        first_volume = next((candidate for candidate in sorted(path.rglob("*")) if self._is_volume_file(candidate)), None)
        if first_volume is not None:
            return first_volume.parent

        raise FileNotFoundError(f"No FastMRI .h5/.hdf5 files were found in: {path}")

    def _is_volume_file(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in self.supported_sample_extensions

    def _normalize_slice_sample(self, sample: tuple[Any, Any, Any, dict[str, Any], str, int]) -> dict[str, Any]:
        kspace, mask, target, attrs, filename, dataslice = sample
        return {
            "sample_id": Path(filename).stem,
            "filename": filename,
            "slice_index": int(dataslice),
            "kspace": None if kspace is None else np.asarray(kspace),
            "mask": None if mask is None else np.asarray(mask),
            "target": None if target is None else np.asarray(target),
            "metadata": dict(attrs),
        }

"""FastMRI dataset implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from .base import BaseDataset

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised via runtime guard.
    np = None

try:
    from fastmri.data import SliceDataset
except ImportError:  # pragma: no cover - exercised via runtime guard.
    SliceDataset = None


DEFAULT_FASTMRI_ROOT = Path.home() / ".cache" / "mri_recon" / "fastmri"
DEFAULT_FASTMRI_SAMPLE_ENV = "MRI_RECON_FASTMRI_SAMPLE_URL"
PACKAGED_SAMPLE_PATH = Path(__file__).with_name("fixtures") / "fastmri_sample_singlecoil.h5"


class FastMRIDataset(BaseDataset):
    """Wrapper around :class:`fastmri.data.SliceDataset`.

    The wrapper keeps a small repository-level API while delegating actual file
    indexing and slice loading to the upstream fastMRI dataset implementation.
    It also provides a default cache location and an optional built-in sample
    source so users can get started without manually wiring dataset paths.
    """

    sample_extension = ".h5"

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
        self.split = split
        self.challenge = challenge
        self.transform = transform
        self.use_dataset_cache = use_dataset_cache
        self.sample_rate = sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.dataset_cache_file = dataset_cache_file
        self.num_cols = num_cols
        self.raw_sample_filter = raw_sample_filter
        self.sample_url = (
            Path(sample_url).resolve().as_uri() if isinstance(sample_url, Path) else sample_url
        )
        self.data_dir = self.root_dir / self.split
        self._slice_dataset: Any | None = None

        if auto_download and not self._has_local_samples():
            self.download()
        elif self._has_local_samples():
            self._initialize_slice_dataset()

    @property
    def slice_dataset(self) -> Any:
        """Expose the underlying ``fastmri.data.SliceDataset`` instance."""

        self._ensure_slice_dataset()
        return self._slice_dataset

    @property
    def default_sample_source(self) -> str | None:
        """Return the built-in or environment-defined sample source."""

        env_source = os.environ.get(DEFAULT_FASTMRI_SAMPLE_ENV)
        if env_source:
            return env_source
        if self.sample_url:
            return self.sample_url
        if PACKAGED_SAMPLE_PATH.exists():
            return PACKAGED_SAMPLE_PATH.as_uri()
        return None

    def download(
        self,
        source: str | Path | None = None,
        destination: str | Path | None = None,
    ) -> Path:
        """Download or copy FastMRI data into the configured split directory."""

        actual_source = source or self.default_sample_source
        if actual_source is None:
            raise ValueError(
                "No FastMRI sample source is configured. Pass a source URL/path or set "
                f"{DEFAULT_FASTMRI_SAMPLE_ENV}."
            )

        target = Path(destination) if destination is not None else self.data_dir
        downloaded_path = super().download(source=actual_source, destination=target)
        self._initialize_slice_dataset()
        return downloaded_path

    def get_sample_path(self, sample_id: str) -> Path:
        """Return the HDF5 path for a FastMRI volume in the configured split."""

        sample_name = sample_id if sample_id.endswith(self.sample_extension) else (
            f"{sample_id}{self.sample_extension}"
        )
        return self.data_dir / sample_name

    def read_sample(self, sample_id: str, slice_index: int = 0) -> dict[str, Any]:
        """Read a single FastMRI slice via the upstream ``SliceDataset``."""

        normalized_sample_id = Path(sample_id).stem
        for index, raw_sample in enumerate(self.slice_dataset.raw_samples):
            if raw_sample.fname.stem == normalized_sample_id and raw_sample.slice_ind == slice_index:
                return self[index]

        raise FileNotFoundError(
            f"FastMRI sample {normalized_sample_id!r} slice {slice_index} was not found in "
            f"{self.data_dir}."
        )

    def to_numpy(self, sample: dict[str, Any], field: str = "target") -> Any:
        """Convert one field of a sample into a NumPy array."""

        if np is None:
            raise ImportError(
                "FastMRIDataset requires numpy. Install dependencies from "
                "requirements.txt before using this dataset."
            )
        if field not in sample:
            raise KeyError(f"Sample does not contain field {field!r}")
        return np.asarray(sample[field])

    def sample_ids(self) -> list[str]:
        """Return the available volume identifiers."""

        seen: list[str] = []
        for raw_sample in self.slice_dataset.raw_samples:
            sample_id = raw_sample.fname.stem
            if sample_id not in seen:
                seen.append(sample_id)
        return seen

    def __len__(self) -> int:
        if not self._has_local_samples() and self._slice_dataset is None:
            return 0
        return len(self.slice_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one slice sampled through the upstream ``SliceDataset``."""

        return self._normalize_slice_sample(self.slice_dataset[index])

    def _ensure_slice_dataset(self) -> None:
        """Initialize the upstream slice dataset when local data is available."""

        if self._slice_dataset is not None:
            return
        if not self._has_local_samples():
            raise FileNotFoundError(
                f"No FastMRI volumes are available in {self.data_dir}. "
                "Call download() first or place official .h5 files there."
            )
        self._initialize_slice_dataset()

    def _initialize_slice_dataset(self) -> None:
        """Create the underlying ``fastmri.data.SliceDataset`` instance."""

        if SliceDataset is None:
            raise ImportError(
                "FastMRIDataset requires the real fastmri package. Install CPU torch first "
                "and then install dependencies from requirements.txt."
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
        """Return whether the split directory currently contains HDF5 volumes."""

        return self.data_dir.exists() and any(
            path.suffix == self.sample_extension for path in self.data_dir.iterdir()
        )

    def _normalize_slice_sample(self, sample: tuple[Any, Any, Any, dict[str, Any], str, int]) -> dict[str, Any]:
        """Convert the upstream tuple return value into the package dictionary format."""

        kspace, mask, target, attrs, filename, dataslice = sample
        return {
            "sample_id": Path(filename).stem,
            "filename": filename,
            "slice_index": int(dataslice),
            "kspace": None if kspace is None else np.asarray(kspace) if np is not None else kspace,
            "mask": None if mask is None else np.asarray(mask) if np is not None else mask,
            "target": None if target is None else np.asarray(target) if np is not None else target,
            "metadata": dict(attrs),
        }

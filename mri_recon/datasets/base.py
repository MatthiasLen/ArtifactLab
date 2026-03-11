"""Shared dataset interfaces for MRI reconstruction datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from numbers import Real
from pathlib import Path
from statistics import fmean
import tarfile
import re
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname, urlretrieve
import zipfile


class BaseDataset(ABC):
    """Abstract interface for MRI datasets.

    Concrete datasets are expected to implement sample lookup and reading while
    reusing the generic download and normalization helpers defined here.
    """

    sample_extension = ""
    _ARCHIVE_SUFFIXES = (
        ".zip",
        ".tar",
        ".tar.gz",
        ".tgz",
        ".tar.bz2",
        ".tbz2",
        ".tar.xz",
        ".txz",
    )

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def download(self, source: str | Path, destination: str | Path | None = None) -> Path:
        """Download or resolve dataset content into the local dataset root.

        The method accepts either a local file/directory or a remote URL. Local
        paths are returned as-is to avoid copying large data. Remote sources are
        downloaded into *destination* and archives are extracted there.
        """

        target = Path(destination) if destination is not None else self.root_dir
        target.parent.mkdir(parents=True, exist_ok=True)

        source_text = str(source)
        local_source = self._resolve_local_source_path(source)
        if local_source is None:
            parsed = urlparse(source_text)
            filename = Path(parsed.path).name or "dataset.bin"
            target.mkdir(parents=True, exist_ok=True)
            destination_file = target / filename
            urlretrieve(source_text, destination_file)
            if self._is_archive_path(destination_file):
                return self._extract_archive(destination_file, target)
            return destination_file

        source_path = local_source
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset source does not exist: {source_path}")

        if source_path.is_dir():
            return source_path

        if self._is_archive_path(source_path):
            target.mkdir(parents=True, exist_ok=True)
            return self._extract_archive(source_path, target)

        return source_path

    def _resolve_local_source_path(self, source: str | Path) -> Path | None:
        if isinstance(source, Path):
            return source

        source_text = str(source)
        if re.match(r"^[a-zA-Z]:[\\/]", source_text):
            return Path(source_text)
        if source_text.startswith("\\\\"):
            return Path(source_text)

        parsed = urlparse(source_text)
        if parsed.scheme and parsed.scheme != "file":
            return None
        if parsed.scheme == "file":
            # Convert file:// URIs into a local path correctly on Windows and POSIX.
            uri_path = url2pathname(unquote(parsed.path))
            if parsed.netloc:
                uri_path = f"//{parsed.netloc}{uri_path}"
            return Path(uri_path)
        return Path(source)

    def _is_archive_path(self, path: Path) -> bool:
        lower_name = path.name.lower()
        return any(lower_name.endswith(suffix) for suffix in self._ARCHIVE_SUFFIXES)

    def _extract_archive(self, archive_path: Path, destination_dir: Path) -> Path:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(destination_dir)
            return destination_dir

        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as archive:
                archive.extractall(destination_dir, filter="data")
            return destination_dir

        raise ValueError(f"Unsupported archive format: {archive_path}")

    @abstractmethod
    def get_sample_path(self, sample_id: str) -> Path:
        """Return the on-disk path for a sample."""

    @abstractmethod
    def read_sample(self, sample_id: str, slice_index: int = 0) -> dict[str, Any]:
        """Read a sample into a standard dictionary representation."""

    def apply_normalization(
        self,
        sample: dict[str, Any],
        field: str = "kspace",
        method: str = "zscore",
    ) -> dict[str, Any]:
        """Return a copy of *sample* with normalized numeric values."""

        if field not in sample:
            raise KeyError(f"Sample does not contain field {field!r}")

        normalized_sample = deepcopy(sample)
        values = self._flatten_numeric_values(sample[field])
        if not values:
            normalized_sample[field] = sample[field]
            return normalized_sample

        if method == "zscore":
            mean = fmean(values)
            variance = fmean([(value - mean) ** 2 for value in values])
            scale = variance**0.5
            transform = (
                (lambda value: 0.0)
                if scale == 0
                else (lambda value: (value - mean) / scale)
            )
        elif method == "minmax":
            minimum = min(values)
            maximum = max(values)
            scale = maximum - minimum
            transform = (
                (lambda value: 0.0)
                if scale == 0
                else (lambda value: (value - minimum) / scale)
            )
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

        normalized_sample[field] = self._map_nested_numeric(sample[field], transform)
        return normalized_sample

    def _flatten_numeric_values(self, data: Any) -> list[float]:
        """Flatten nested lists and tuples into a flat list of floats."""

        if isinstance(data, Real):
            return [float(data)]
        if hasattr(data, "tolist") and not isinstance(data, (str, bytes)):
            return self._flatten_numeric_values(data.tolist())
        if isinstance(data, (list, tuple)):
            values: list[float] = []
            for item in data:
                values.extend(self._flatten_numeric_values(item))
            return values
        raise TypeError("Normalization expects numeric values stored in nested lists or tuples")

    def _map_nested_numeric(self, data: Any, transform_fn: Any) -> Any:
        """Apply *transform_fn* to all numeric values while preserving nesting."""

        if isinstance(data, Real):
            return transform_fn(float(data))
        if hasattr(data, "tolist") and not isinstance(data, (str, bytes)):
            try:
                import numpy as np
            except ImportError:  # pragma: no cover - numpy is optional in the base layer.
                return self._map_nested_numeric(data.tolist(), transform_fn)

            return np.asarray(self._map_nested_numeric(data.tolist(), transform_fn))
        if isinstance(data, list):
            return [self._map_nested_numeric(item, transform_fn) for item in data]
        if isinstance(data, tuple):
            return tuple(self._map_nested_numeric(item, transform_fn) for item in data)
        raise TypeError("Normalization expects numeric values stored in nested lists or tuples")

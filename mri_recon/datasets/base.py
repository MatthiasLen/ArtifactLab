"""Shared dataset interfaces for MRI reconstruction datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from numbers import Real
from pathlib import Path
import shutil
from statistics import fmean
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname, urlretrieve


class BaseDataset(ABC):
    """Abstract interface for MRI datasets.

    Concrete datasets are expected to implement sample lookup and reading while
    reusing the generic download and normalization helpers defined here.
    """

    sample_extension = ""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def download(self, source: str | Path, destination: str | Path | None = None) -> Path:
        """Download or copy dataset content into the local dataset root.

        The method accepts either a local file/directory or a remote URL. Local
        directories are copied recursively, while files are copied directly.
        """

        target = Path(destination) if destination is not None else self.root_dir
        target.parent.mkdir(parents=True, exist_ok=True)

        parsed = urlparse(str(source))
        if parsed.scheme and parsed.scheme != "file":
            filename = Path(parsed.path).name or "dataset.bin"
            target.mkdir(parents=True, exist_ok=True)
            destination_file = target / filename
            urlretrieve(str(source), destination_file)
            return destination_file

        if parsed.scheme == "file":
            # Convert file:// URIs into a local path correctly on Windows and POSIX.
            uri_path = url2pathname(unquote(parsed.path))
            if parsed.netloc:
                uri_path = f"//{parsed.netloc}{uri_path}"
            source_path = Path(uri_path)
        else:
            source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset source does not exist: {source_path}")

        if source_path.is_dir():
            shutil.copytree(source_path, target, dirs_exist_ok=True)
            return target

        target.mkdir(parents=True, exist_ok=True)
        destination_file = target / source_path.name
        shutil.copy2(source_path, destination_file)
        return destination_file

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

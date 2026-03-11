"""FastMRI dataset implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseDataset


class FastMRIDataset(BaseDataset):
    """Dataset wrapper for lightweight FastMRI-style JSON samples."""

    sample_extension = ".json"

    def __init__(self, root_dir: str | Path, split: str = "train") -> None:
        super().__init__(root_dir=root_dir)
        self.split = split

    def download(self, source: str | Path, destination: str | Path | None = None) -> Path:
        """Copy or download data for the configured split."""

        target = Path(destination) if destination is not None else self.root_dir / self.split
        return super().download(source=source, destination=target)

    def get_sample_path(self, sample_id: str) -> Path:
        """Return the JSON file path for a FastMRI sample."""

        return self.root_dir / self.split / f"{sample_id}{self.sample_extension}"

    def read_sample(self, sample_id: str) -> dict[str, Any]:
        """Read a FastMRI sample from disk.

        Samples are stored as compact JSON files in tests and lightweight
        development environments. The returned dictionary always includes
        ``sample_id``, ``kspace``, and ``metadata`` keys.
        """

        sample_path = self.get_sample_path(sample_id)
        if not sample_path.exists():
            raise FileNotFoundError(f"FastMRI sample does not exist: {sample_path}")

        with sample_path.open("r", encoding="utf-8") as handle:
            sample = json.load(handle)

        missing_keys = {"kspace", "metadata"} - sample.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"FastMRI sample is missing required keys: {missing}")

        sample.setdefault("sample_id", sample_id)
        return sample

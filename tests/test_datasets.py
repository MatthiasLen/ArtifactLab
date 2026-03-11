"""Tests for dataset interfaces."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mri_recon.datasets import BaseDataset, FastMRIDataset


DUMMY_KSPACE = [[1.0, 2.0], [3.0, 4.0]]
HAS_H5PY = importlib.util.find_spec("h5py") is not None

if HAS_H5PY:
    import h5py


class DatasetBaseTests(unittest.TestCase):
    """Validate generic dataset behaviour shared by all datasets."""

    def test_base_dataset_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BaseDataset("/tmp/base-dataset-test")  # type: ignore[abstract]

    def test_apply_normalization_returns_zscore_copy(self) -> None:
        class DummyDataset(BaseDataset):
            def get_sample_path(self, sample_id: str) -> Path:
                return Path(sample_id)

            def read_sample(self, sample_id: str, slice_index: int = 0) -> dict[str, object]:
                return {"sample_id": sample_id, "kspace": DUMMY_KSPACE}

        dataset = DummyDataset("/tmp/dummy-dataset")
        sample = {"kspace": DUMMY_KSPACE, "metadata": {"source": "dummy"}}

        normalized = dataset.apply_normalization(sample)

        self.assertEqual(sample["kspace"], DUMMY_KSPACE)
        flattened = [value for row in normalized["kspace"] for value in row]
        self.assertAlmostEqual(sum(flattened), 0.0, places=7)
        self.assertAlmostEqual(flattened[0], -1.3416407864998738)
        self.assertAlmostEqual(flattened[-1], 1.3416407864998738)


@unittest.skipUnless(HAS_H5PY, "h5py is required for FastMRI dataset tests")
class FastMRIDatasetTests(unittest.TestCase):
    """Validate FastMRI-specific dataset behaviour."""

    def test_download_read_and_normalize_fastmri_volume(self) -> None:
        with TemporaryDirectory() as source_directory, TemporaryDirectory() as root_directory:
            source_path = Path(source_directory) / "sample_volume.h5"
            with h5py.File(source_path, "w") as handle:
                handle.create_dataset(
                    "kspace",
                    data=[
                        [[[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]]],
                        [[[2 + 1j, 4 + 3j], [6 + 5j, 8 + 7j]]],
                    ],
                )
                handle.create_dataset(
                    "reconstruction_rss",
                    data=[
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[2.0, 3.0], [4.0, 5.0]],
                    ],
                )
                handle.create_dataset("mask", data=[1, 0])
                handle.attrs["acquisition"] = "AXT2"
                handle.attrs["patient_id"] = "patient-123"

            dataset = FastMRIDataset(root_directory, split="train", challenge="multicoil")
            downloaded_path = dataset.download(f"file://{source_path}")
            sample = dataset.read_sample("sample_volume", slice_index=1)
            normalized = dataset.apply_normalization(sample, field="target", method="minmax")

            self.assertEqual(downloaded_path, Path(root_directory) / "train" / "sample_volume.h5")
            self.assertTrue(downloaded_path.exists())
            self.assertEqual(sample["sample_id"], "sample_volume")
            self.assertEqual(sample["slice_index"], 1)
            self.assertEqual(sample["metadata"]["num_slices"], 2)
            self.assertEqual(sample["metadata"]["target_key"], "reconstruction_rss")
            self.assertEqual(sample["metadata"]["patient_id"], "patient-123")
            self.assertEqual(sample["kspace"][0][0][0], [2.0, 1.0])
            self.assertEqual(sample["target"], [[2.0, 3.0], [4.0, 5.0]])
            self.assertEqual(normalized["target"], [[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]])
            self.assertEqual(sample["target"], [[2.0, 3.0], [4.0, 5.0]])


if __name__ == "__main__":
    unittest.main()

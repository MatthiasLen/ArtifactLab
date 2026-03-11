"""Tests for dataset interfaces."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mri_recon.datasets import BaseDataset, FastMRIDataset


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "fastmri"


class DatasetBaseTests(unittest.TestCase):
    """Validate generic dataset behaviour shared by all datasets."""

    def test_base_dataset_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BaseDataset("/tmp/base-dataset-test")  # type: ignore[abstract]

    def test_apply_normalization_returns_zscore_copy(self) -> None:
        class DummyDataset(BaseDataset):
            def get_sample_path(self, sample_id: str) -> Path:
                return Path(sample_id)

            def read_sample(self, sample_id: str) -> dict[str, object]:
                return {"sample_id": sample_id, "kspace": [[1.0, 2.0], [3.0, 4.0]]}

        dataset = DummyDataset("/tmp/dummy-dataset")
        sample = {"kspace": [[1.0, 2.0], [3.0, 4.0]], "metadata": {"source": "dummy"}}

        normalized = dataset.apply_normalization(sample)

        self.assertEqual(sample["kspace"], [[1.0, 2.0], [3.0, 4.0]])
        flattened = [value for row in normalized["kspace"] for value in row]
        self.assertAlmostEqual(sum(flattened), 0.0, places=7)
        self.assertAlmostEqual(flattened[0], -1.3416407864998738)
        self.assertAlmostEqual(flattened[-1], 1.3416407864998738)


class FastMRIDatasetTests(unittest.TestCase):
    """Validate FastMRI-specific dataset behaviour."""

    def test_download_copies_fixture_split(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            dataset = FastMRIDataset(temporary_directory, split="train")

            downloaded_path = dataset.download(FIXTURE_ROOT / "train")
            sample_path = dataset.get_sample_path("sample_001")

            self.assertEqual(downloaded_path, Path(temporary_directory) / "train")
            self.assertEqual(sample_path, Path(temporary_directory) / "train" / "sample_001.json")
            self.assertTrue(sample_path.exists())

    def test_read_sample_and_minmax_normalization(self) -> None:
        dataset = FastMRIDataset(FIXTURE_ROOT, split="train")

        sample = dataset.read_sample("sample_001")
        normalized = dataset.apply_normalization(sample, method="minmax")

        self.assertEqual(sample["metadata"]["patient_id"], "patient-123")
        self.assertEqual(sample["sample_id"], "sample_001")
        self.assertEqual(normalized["kspace"], [[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]])
        self.assertEqual(sample["kspace"], [[0.0, 2.0], [4.0, 6.0]])


if __name__ == "__main__":
    unittest.main()

"""Tests for dataset interfaces."""

from __future__ import annotations

from contextlib import contextmanager
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from mri_recon.datasets import BaseDataset, FastMRIDataset


DUMMY_KSPACE = [[1.0, 2.0], [3.0, 4.0]]
HAS_H5PY = importlib.util.find_spec("h5py") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_FASTMRI = importlib.util.find_spec("fastmri") is not None

if HAS_H5PY:
    import h5py
if HAS_NUMPY:
    import numpy as np


def _ismrmrd_header(encoded_x: int, encoded_y: int, recon_x: int, recon_y: int) -> bytes:
    return f"""<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
  <encoding>
    <encodedSpace>
      <matrixSize>
        <x>{encoded_x}</x>
        <y>{encoded_y}</y>
        <z>1</z>
      </matrixSize>
    </encodedSpace>
    <reconSpace>
      <matrixSize>
        <x>{recon_x}</x>
        <y>{recon_y}</y>
        <z>1</z>
      </matrixSize>
    </reconSpace>
    <encodingLimits>
      <kspace_encoding_step_1>
        <center>{encoded_y // 2}</center>
        <maximum>{encoded_y - 1}</maximum>
      </kspace_encoding_step_1>
    </encodingLimits>
  </encoding>
</ismrmrdHeader>""".encode("utf-8")


def _create_fastmri_singlecoil_sample(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "kspace",
            data=np.asarray(
                [
                    [[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]],
                    [[2 + 1j, 3 + 2j], [4 + 3j, 5 + 4j]],
                ],
                dtype=np.complex64,
            ),
        )
        handle.create_dataset(
            "reconstruction_esc",
            data=np.asarray(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[2.0, 3.0], [4.0, 5.0]],
                ],
                dtype=np.float32,
            ),
        )
        handle.create_dataset("ismrmrd_header", data=_ismrmrd_header(2, 2, 2, 2))
        handle.attrs["acquisition"] = "CORPD_FBK"
        handle.attrs["max"] = 5.0


@contextmanager
def _serve_directory(directory: Path) -> str:
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


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


@unittest.skipUnless(HAS_H5PY and HAS_NUMPY and HAS_FASTMRI, "fastmri runtime is required")
class FastMRIDatasetTests(unittest.TestCase):
    """Validate FastMRI-specific dataset behaviour."""

    def test_downloads_sample_with_real_fastmri_and_converts_to_numpy(self) -> None:
        with TemporaryDirectory() as source_directory, TemporaryDirectory() as root_directory:
            source_path = Path(source_directory) / "sample_singlecoil.h5"
            _create_fastmri_singlecoil_sample(source_path)

            with _serve_directory(Path(source_directory)) as base_url:
                dataset = FastMRIDataset(
                    root_directory,
                    split="val",
                    challenge="singlecoil",
                    sample_url=f"{base_url}/sample_singlecoil.h5",
                    auto_download=True,
                )

                downloaded_path = dataset.get_sample_path("sample_singlecoil")
                sample = dataset.read_sample("sample_singlecoil", slice_index=1)
                matrix = dataset.to_numpy(sample, field="target")
                normalized = dataset.apply_normalization(sample, field="target", method="minmax")

            self.assertTrue(downloaded_path.exists())
            self.assertEqual(downloaded_path, Path(root_directory) / "val" / "sample_singlecoil.h5")
            self.assertGreater(len(dataset), 0)
            self.assertIn("sample_singlecoil", dataset.sample_ids())
            self.assertTrue(dataset.slice_dataset.__class__.__module__.startswith("fastmri"))
            self.assertEqual(sample["sample_id"], "sample_singlecoil")
            self.assertEqual(sample["slice_index"], 1)
            self.assertEqual(sample["metadata"]["acquisition"], "CORPD_FBK")
            self.assertIsInstance(matrix, np.ndarray)
            self.assertEqual(matrix.shape, (2, 2))
            self.assertAlmostEqual(float(matrix[0, 0]), 2.0)
            self.assertTrue(np.iscomplexobj(sample["kspace"]))
            self.assertTrue(np.array_equal(sample["target"], np.asarray([[2.0, 3.0], [4.0, 5.0]])))
            self.assertTrue(
                np.allclose(
                    normalized["target"],
                    np.asarray([[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]]),
                )
            )


if __name__ == "__main__":
    unittest.main()

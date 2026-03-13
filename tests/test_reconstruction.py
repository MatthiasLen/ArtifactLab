"""Tests for MRI reconstruction interfaces."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest

from mri_recon.datasets import FastMRIDataset
from mri_recon.datasets.fastmri import PACKAGED_SAMPLE_PATH
from mri_recon.reconstruction import (
    BaseReconstructor,
    ConjugateGradientReconstructor,
    DeepInverseRAMReconstructor,
    FISTAL1Reconstructor,
    LandweberReconstructor,
    POCSReconstructor,
    TVPDHGReconstructor,
    TikhonovReconstructor,
    ZeroFilledReconstructor,
)
from mri_recon.reconstruction.classic import _fft2c


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_FASTMRI = importlib.util.find_spec("fastmri") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DEEPINV = importlib.util.find_spec("deepinv") is not None

if HAS_NUMPY:
    import numpy as np


class ReconstructionBaseTests(unittest.TestCase):
    """Validate generic behaviour shared by reconstructors."""

    def test_base_reconstructor_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BaseReconstructor()  # type: ignore[abstract]


@unittest.skipUnless(HAS_NUMPY, "numpy runtime is required")
class ClassicReconstructionTests(unittest.TestCase):
    """Validate classic MRI reconstructors on synthetic data."""

    def test_zero_filled_reconstruction_recovers_fully_sampled_image(self) -> None:
        image = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        sample = {"kspace": _fft2c(image), "mask": np.ones_like(image, dtype=bool), "target": image}

        reconstructor = ZeroFilledReconstructor()
        reconstructed = reconstructor.apply_reconstruction(sample)

        self.assertTrue(np.allclose(reconstructed, image))
        self.assertTrue(np.array_equal(reconstructor.get_target(sample), image))
        self.assertTrue(np.allclose(reconstructor(sample), image))

    def test_landweber_reconstruction_recovers_fully_sampled_image(self) -> None:
        image = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        sample = {"kspace": _fft2c(image), "mask": np.ones_like(image, dtype=bool)}

        reconstructed = LandweberReconstructor(num_iterations=5).apply_reconstruction(sample)

        self.assertTrue(np.allclose(reconstructed, image))

    def test_conjugate_gradient_reconstruction_recovers_fully_sampled_image(self) -> None:
        image = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sample = {"kspace": _fft2c(image), "mask": np.ones_like(image, dtype=bool)}

        reconstructed = ConjugateGradientReconstructor(num_iterations=10).apply_reconstruction(
            sample
        )

        self.assertTrue(np.allclose(reconstructed, image, atol=1e-6))

    def test_tikhonov_reconstruction_matches_zero_filled_when_lambda_is_zero(self) -> None:
        image = np.asarray([[1.0, 2.0], [0.5, 0.25]], dtype=np.float32)
        sample = {"kspace": _fft2c(image), "mask": np.ones_like(image, dtype=bool)}

        reconstructed = TikhonovReconstructor(l2_weight=0.0).apply_reconstruction(sample)

        self.assertTrue(np.allclose(reconstructed, image, atol=1e-6))


@unittest.skipUnless(HAS_NUMPY, "numpy runtime is required")
class UndersampledReconstructionTests(unittest.TestCase):
    """Validate undersampled-specific reconstruction methods."""

    def test_pocs_reconstruction_runs_on_undersampled_data(self) -> None:
        image = np.asarray([[1.0, 0.0], [0.5, 2.0]], dtype=np.float32)
        mask = np.asarray([[1, 0], [0, 1]], dtype=bool)
        sample = {"kspace": _fft2c(image), "mask": mask}

        reconstructed = POCSReconstructor(
            num_iterations=10,
        ).apply_reconstruction(sample)

        self.assertEqual(reconstructed.shape, image.shape)
        self.assertTrue(np.isfinite(reconstructed).all())

    def test_fista_l1_reconstruction_runs_on_undersampled_data(self) -> None:
        image = np.asarray([[1.0, 0.0], [0.5, 2.0]], dtype=np.float32)
        mask = np.asarray([[1, 0], [0, 1]], dtype=bool)
        sample = {"kspace": _fft2c(image), "mask": mask}

        reconstructed = FISTAL1Reconstructor(
            num_iterations=10,
            l1_weight=1e-2,
        ).apply_reconstruction(sample)

        self.assertEqual(reconstructed.shape, image.shape)
        self.assertTrue(np.isfinite(reconstructed).all())

    def test_fista_l1_reconstruction_is_not_all_zero_for_small_scale_data(self) -> None:
        image = np.asarray([[1.0, 0.0], [0.5, 2.0]], dtype=np.float32) * 1e-4
        mask = np.asarray([[1, 0], [0, 1]], dtype=bool)
        sample = {"kspace": _fft2c(image), "mask": mask}

        reconstructed = FISTAL1Reconstructor(
            num_iterations=15,
            l1_weight=1e-3,
            step_size=1.0,
        ).apply_reconstruction(sample)

        self.assertGreater(float(np.max(np.abs(reconstructed))), 0.0)

    def test_tv_pdhg_reconstruction_runs_on_undersampled_data(self) -> None:
        image = np.asarray([[1.0, 0.0], [0.5, 2.0]], dtype=np.float32)
        mask = np.asarray([[1, 0], [0, 1]], dtype=bool)
        sample = {"kspace": _fft2c(image), "mask": mask}

        reconstructed = TVPDHGReconstructor(
            num_iterations=20,
            tv_weight=1e-3,
        ).apply_reconstruction(sample)

        self.assertEqual(reconstructed.shape, image.shape)
        self.assertTrue(np.isfinite(reconstructed).all())

    def test_tv_pdhg_matches_zero_filled_for_fully_sampled_data(self) -> None:
        image = np.asarray([[1.0, 2.0], [0.5, 0.25]], dtype=np.float32)
        full_mask = np.ones_like(image, dtype=bool)
        sample = {"kspace": _fft2c(image), "mask": full_mask}

        zero_filled = ZeroFilledReconstructor().apply_reconstruction(sample)
        tv_reconstructed = TVPDHGReconstructor(
            num_iterations=40,
            tv_weight=1e-5,
            tau=0.2,
            sigma=0.2,
        ).apply_reconstruction(sample)

        relative_error = np.linalg.norm(tv_reconstructed - zero_filled) / (
            np.linalg.norm(zero_filled) + 1e-12
        )
        self.assertLess(float(relative_error), 0.05)


class _FakeModel:
    def __init__(self, gain: float = 1.0, **kwargs) -> None:  # noqa: ANN003 - test double
        del kwargs
        self.gain = gain
        self.eval_called = False

    def eval(self) -> "_FakeModel":
        self.eval_called = True
        return self

    def __call__(self, input_data, **kwargs):  # noqa: ANN001 - test double
        del kwargs
        return input_data * self.gain


class _FakePhysicsModel(_FakeModel):
    def __call__(self, input_data, physics=None, **kwargs):  # noqa: ANN001 - test double
        del kwargs
        if physics is None:
            raise TypeError("physics is required")
        return input_data * physics


class DeepInverseReconstructionTests(unittest.TestCase):
    """Validate DeepInverse RAM wrapper without requiring the real dependency."""

    @unittest.skipUnless(HAS_NUMPY, "numpy runtime is required")
    def test_ram_reconstructor_apply_reconstruction_uses_injected_model(self) -> None:
        model = _FakePhysicsModel()
        sample = {
            "kspace": np.asarray([[1.0, 2.0]], dtype=np.float32),
            "mask": None,
        }

        reconstructed = DeepInverseRAMReconstructor(
            model=model,
            physics=3.0,
        ).apply_reconstruction(sample)

        self.assertTrue(model.eval_called)
        reconstructed_array = np.asarray(reconstructed)
        self.assertTrue(
            np.allclose(
                reconstructed_array,
                np.asarray(sample["kspace"], dtype=np.float32) * 3.0,
            )
        )

    def test_ram_reconstructor_build_model_uses_pretrained_by_default(self) -> None:
        captured_kwargs: dict[str, object] = {}

        class _RAMFactoryModel(_FakeModel):
            def __init__(self, **kwargs) -> None:  # noqa: ANN003 - test double
                captured_kwargs.update(kwargs)
                super().__init__(**kwargs)

        fake_module = ModuleType("deepinv")
        fake_module.models = SimpleNamespace(RAM=_RAMFactoryModel)
        original_module = sys.modules.get("deepinv")

        try:
            sys.modules["deepinv"] = fake_module
            DeepInverseRAMReconstructor().build_model()
            self.assertEqual(captured_kwargs.get("pretrained"), True)
        finally:
            if original_module is None:
                sys.modules.pop("deepinv", None)
            else:
                sys.modules["deepinv"] = original_module



@unittest.skipUnless(HAS_NUMPY and HAS_FASTMRI, "fastmri runtime is required")
class FastMRIReconstructionIntegrationTests(unittest.TestCase):
    """Run a real reconstructor on a real FastMRI sample."""

    def test_zero_filled_reconstruction_runs_on_packaged_fastmri_sample(self) -> None:
        with TemporaryDirectory() as root_directory:
            dataset = FastMRIDataset(
                root_directory,
                split="val",
                challenge="singlecoil",
                sample_url=PACKAGED_SAMPLE_PATH,
                auto_download=True,
            )

            sample = dataset.read_sample("fastmri_sample_singlecoil", slice_index=0)
            reconstructed = ZeroFilledReconstructor().apply_reconstruction(sample)

            self.assertEqual(reconstructed.shape, sample["target"].shape)
            self.assertTrue(np.isfinite(reconstructed).all())
            self.assertGreater(float(np.max(reconstructed)), 0.0)

    @unittest.skipUnless(HAS_NUMPY and HAS_FASTMRI and HAS_TORCH and HAS_DEEPINV, "full deepinverse runtime is required")
    def test_deepinverse_ram_physics_builds_on_packaged_fastmri_sample(self) -> None:
        with TemporaryDirectory() as root_directory:
            dataset = FastMRIDataset(
                root_directory,
                split="val",
                challenge="singlecoil",
                sample_url=PACKAGED_SAMPLE_PATH,
                auto_download=True,
            )

            sample = dataset.read_sample("fastmri_sample_singlecoil", slice_index=0)
            physics = DeepInverseRAMReconstructor.build_mri_physics(sample)
            self.assertIsNotNone(physics)


if __name__ == "__main__":
    unittest.main()

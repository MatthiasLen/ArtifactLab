"""Tests for image metric interfaces."""

from __future__ import annotations

import importlib.util
import math
import unittest

from mri_recon.metrics import (
    BaseMetric,
    EntropyMetric,
    L1Metric,
    LPIPSMetric,
    MSEMetric,
    NMSEMetric,
    PSNRMetric,
    RMSEMetric,
    SSIMMetric,
)


HAS_NUMPY = importlib.util.find_spec("numpy") is not None

if HAS_NUMPY:
    import numpy as np


class MetricBaseTests(unittest.TestCase):
    """Validate generic behaviour shared by image metrics."""

    def test_base_metric_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BaseMetric()  # type: ignore[abstract]


@unittest.skipUnless(HAS_NUMPY, "numpy runtime is required")
class ReferenceMetricTests(unittest.TestCase):
    """Validate reference metrics on compact example arrays."""

    def setUp(self) -> None:
        self.reference = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        self.prediction = np.asarray([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32)

    def test_l1_mse_rmse_and_nmse_match_expected_values(self) -> None:
        self.assertAlmostEqual(L1Metric().apply_metric(self.prediction, self.reference), 0.5)
        self.assertAlmostEqual(MSEMetric().apply_metric(self.prediction, self.reference), 0.5)
        self.assertAlmostEqual(RMSEMetric().apply_metric(self.prediction, self.reference), math.sqrt(0.5))
        self.assertAlmostEqual(NMSEMetric().apply_metric(self.prediction, self.reference), 1.0 / 7.0)

    def test_reference_metrics_accept_sample_dictionaries(self) -> None:
        sample = {"prediction": self.prediction, "target": self.reference}
        self.assertAlmostEqual(L1Metric()(sample), 0.5)

    def test_reference_metric_requires_reference_image(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference image"):
            L1Metric().apply_metric(self.prediction)

    def test_reference_metric_rejects_none_prediction(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be None"):
            L1Metric().apply_metric(None, self.reference)

    def test_reference_metric_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            SSIMMetric().apply_metric(self.prediction, self.reference[:, :1])

    def test_reference_metric_rejects_empty_and_non_finite_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            PSNRMetric().apply_metric(np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "finite values"):
            MSEMetric().apply_metric(
                np.asarray([[np.nan, 0.0]], dtype=np.float32),
                np.asarray([[0.0, 0.0]], dtype=np.float32),
            )

    def test_psnr_is_infinite_and_ssim_is_one_for_identical_images(self) -> None:
        self.assertTrue(math.isinf(PSNRMetric().apply_metric(self.reference, self.reference)))
        self.assertAlmostEqual(SSIMMetric().apply_metric(self.reference, self.reference), 1.0)

    def test_ssim_decreases_for_different_images(self) -> None:
        self.assertLess(SSIMMetric().apply_metric(self.prediction, self.reference), 1.0)

    def test_lpips_supports_injected_backend(self) -> None:
        captured_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def fake_backend(prediction: np.ndarray, reference: np.ndarray) -> float:
            captured_shapes.append((prediction.shape, reference.shape))
            return float(np.mean(np.abs(prediction - reference)))

        metric = LPIPSMetric(backend=fake_backend)
        value = metric.apply_metric(self.prediction, self.reference)

        self.assertGreaterEqual(value, 0.0)
        self.assertEqual(captured_shapes, [((1, 3, 2, 2), (1, 3, 2, 2))])

    def test_lpips_raises_without_backend_or_optional_dependency(self) -> None:
        with self.assertRaises(ImportError):
            LPIPSMetric().apply_metric(self.prediction, self.reference)

    def test_lpips_rejects_invalid_tensor_rank(self) -> None:
        metric = LPIPSMetric(backend=lambda prediction, reference: 0.0)
        with self.assertRaisesRegex(ValueError, "2D, 3D or 4D"):
            metric.apply_metric(
                np.zeros((1, 1, 1, 1, 1), dtype=np.float32),
                np.zeros((1, 1, 1, 1, 1), dtype=np.float32),
            )

    def test_lpips_rejects_invalid_channel_configuration(self) -> None:
        metric = LPIPSMetric(backend=lambda prediction, reference: 0.0)
        with self.assertRaisesRegex(ValueError, "one or three channels"):
            metric.apply_metric(
                np.zeros((2, 2, 2), dtype=np.float32),
                np.zeros((2, 2, 2), dtype=np.float32),
            )


@unittest.skipUnless(HAS_NUMPY, "numpy runtime is required")
class NonReferenceMetricTests(unittest.TestCase):
    """Validate non-reference metrics on compact example arrays."""

    def test_entropy_is_zero_for_constant_images(self) -> None:
        image = np.zeros((4, 4), dtype=np.float32)
        self.assertEqual(EntropyMetric().apply_metric(image), 0.0)

    def test_entropy_is_positive_for_non_constant_images(self) -> None:
        image = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        self.assertGreater(EntropyMetric(num_bins=2).apply_metric(image), 0.0)

    def test_entropy_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            EntropyMetric(num_bins=1)

"""Tests for image metric interfaces."""

from __future__ import annotations

import importlib.util
import math
import unittest
from unittest.mock import patch

from mri_recon.metrics import (
    BaseMetric,
    BlurEffectMetric,
    EntropyMetric,
    GMSDMetric,
    L1Metric,
    LPIPSMetric,
    MSEMetric,
    NMSEMetric,
    PSNRMetric,
    RMSContrastMetric,
    RMSEMetric,
    SREMetric,
    SSIMMetric,
    TenengradMetric,
    UQIMetric,
)
from mri_recon.metrics.nonreference import BlurEffectMetric as BlurEffectMetricModuleImport
from mri_recon.metrics.reference import UQIMetric as UQIMetricModuleImport


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
        self.assertAlmostEqual(UQIMetric().apply_metric(self.reference, self.reference), 1.0)
        self.assertEqual(GMSDMetric().apply_metric(self.reference, self.reference), 0.0)
        self.assertTrue(math.isinf(SREMetric().apply_metric(self.reference, self.reference)))

    def test_ssim_decreases_for_different_images(self) -> None:
        self.assertLess(SSIMMetric().apply_metric(self.prediction, self.reference), 1.0)
        self.assertLess(UQIMetric().apply_metric(self.prediction, self.reference), 1.0)
        self.assertGreater(GMSDMetric().apply_metric(self.prediction, self.reference), 0.0)

    def test_reference_module_exports_are_available(self) -> None:
        self.assertIs(UQIMetricModuleImport, UQIMetric)
        self.assertAlmostEqual(UQIMetricModuleImport().apply_metric(self.reference, self.reference), 1.0)

    def test_sre_matches_expected_value(self) -> None:
        # mean(reference**2) == 3.5 and MSE(prediction, reference) == 0.5.
        expected = 10.0 * math.log10(3.5 / 0.5)
        self.assertAlmostEqual(SREMetric().apply_metric(self.prediction, self.reference), expected)

    def test_gmsd_rejects_one_dimensional_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two dimensions"):
            GMSDMetric().apply_metric(
                np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
            )

    def test_lpips_prepares_inputs_for_backend(self) -> None:
        captured_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def fake_backend(prediction: np.ndarray, reference: np.ndarray) -> float:
            captured_shapes.append((prediction.shape, reference.shape))
            difference = (prediction - reference).detach().cpu().numpy()
            return float(np.mean(np.abs(difference)))

        with patch("torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity", return_value=fake_backend):
            metric = LPIPSMetric()
            value = metric.apply_metric(self.prediction / 3.0, self.reference / 3.0)

        self.assertGreaterEqual(value, 0.0)
        self.assertEqual(captured_shapes, [((1, 3, 2, 2), (1, 3, 2, 2))])

    def test_lpips_initializes_metric_once_in_constructor(self) -> None:
        with patch("torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity", return_value=lambda prediction, reference: 0.123) as constructor:
            metric = LPIPSMetric()
            first = metric.apply_metric(self.prediction / 3.0, self.reference / 3.0)
            second = metric.apply_metric(self.prediction / 3.0, self.reference / 3.0)

        self.assertEqual(first, 0.123)
        self.assertEqual(second, 0.123)
        self.assertEqual(constructor.call_count, 1)

    def test_lpips_rejects_invalid_tensor_rank(self) -> None:
        with patch("torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity", return_value=lambda prediction, reference: 0.0):
            metric = LPIPSMetric()
            with self.assertRaisesRegex(ValueError, "2D, 3D or 4D"):
                metric.apply_metric(
                    np.zeros((1, 1, 1, 1, 1), dtype=np.float32),
                    np.zeros((1, 1, 1, 1, 1), dtype=np.float32),
                )

    def test_lpips_rejects_invalid_channel_configuration(self) -> None:
        with patch("torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity", return_value=lambda prediction, reference: 0.0):
            metric = LPIPSMetric()
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

    def test_nonreference_module_exports_are_available(self) -> None:
        self.assertIs(BlurEffectMetricModuleImport, BlurEffectMetric)

    def test_blur_related_metrics_distinguish_sharp_and_blurred_images(self) -> None:
        sharp = np.asarray(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        blurred = np.asarray(
            [
                [0.25, 0.50, 0.50, 0.50, 0.25],
                [0.25, 0.50, 0.50, 0.50, 0.25],
                [0.25, 0.50, 0.50, 0.50, 0.25],
                [0.25, 0.50, 0.50, 0.50, 0.25],
                [0.25, 0.50, 0.50, 0.50, 0.25],
            ],
            dtype=np.float32,
        )

        self.assertGreater(BlurEffectMetric().apply_metric(blurred), BlurEffectMetric().apply_metric(sharp))
        self.assertGreater(TenengradMetric().apply_metric(sharp), TenengradMetric().apply_metric(blurred))

    def test_rms_contrast_reflects_contrast_changes(self) -> None:
        low_contrast = np.asarray([[0.45, 0.55], [0.50, 0.50]], dtype=np.float32)
        high_contrast = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

        self.assertGreater(
            RMSContrastMetric().apply_metric(high_contrast),
            RMSContrastMetric().apply_metric(low_contrast),
        )

    def test_blur_effect_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "odd integer of at least 3"):
            BlurEffectMetric(kernel_size=2)

    def test_blur_effect_rejects_one_dimensional_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two dimensions"):
            BlurEffectMetric().apply_metric(np.asarray([0.0, 1.0, 0.0], dtype=np.float32))

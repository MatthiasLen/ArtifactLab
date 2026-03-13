"""Non-reference image quality metrics."""

from __future__ import annotations

from math import sqrt
from typing import Any

from .base import BaseMetric, _require_numpy, gradient_magnitude, require_spatial_image

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised via runtime guard.
    np = None


def _convolve_same(values: Any, kernel: Any) -> Any:
    _require_numpy()
    return np.convolve(values, kernel, mode="same")


def _uniform_blur(array: Any, kernel_size: int) -> Any:
    _require_numpy()
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    blurred = np.apply_along_axis(_convolve_same, -1, array, kernel)
    return np.apply_along_axis(_convolve_same, -2, blurred, kernel)


class EntropyMetric(BaseMetric):
    """Shannon entropy of a single image."""

    def __init__(self, num_bins: int = 256, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        self.num_bins = num_bins

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, _ = self.prepare_inputs(
            prediction,
            reference,
            require_reference=False,
        )
        flattened = prediction_array.reshape(-1)
        minimum = float(np.min(flattened))
        maximum = float(np.max(flattened))
        if minimum == maximum:
            return 0.0
        histogram, _ = np.histogram(flattened, bins=self.num_bins, range=(minimum, maximum))
        probabilities = histogram.astype(np.float64) / float(np.sum(histogram))
        probabilities = probabilities[probabilities > 0]
        return float(-np.sum(probabilities * np.log2(probabilities)))


class BlurEffectMetric(BaseMetric):
    """Approximate blur score based on edge preservation after re-blurring."""

    def __init__(self, kernel_size: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least 3")
        self.kernel_size = kernel_size

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, _ = self.prepare_inputs(prediction, reference, require_reference=False)
        require_spatial_image(prediction_array, "BlurEffectMetric")

        source = np.abs(prediction_array)
        blurred = _uniform_blur(source, self.kernel_size)
        axis_scores: list[float] = []
        for axis in (-1, -2):
            original_edges = np.abs(np.diff(source, axis=axis))
            blurred_edges = np.abs(np.diff(blurred, axis=axis))
            edge_energy = float(np.sum(original_edges))
            if edge_energy == 0.0:
                axis_scores.append(1.0)
                continue
            preserved_sharpness = float(
                np.sum(np.maximum(original_edges - blurred_edges, 0.0)) / edge_energy
            )
            axis_scores.append(1.0 - preserved_sharpness)
        return float(np.mean(axis_scores))


class TenengradMetric(BaseMetric):
    """Gradient-energy focus measure, larger values indicate sharper images."""

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, _ = self.prepare_inputs(prediction, reference, require_reference=False)
        require_spatial_image(prediction_array, "TenengradMetric")
        gradients = gradient_magnitude(prediction_array)
        return float(np.mean(gradients * gradients))


class RMSContrastMetric(BaseMetric):
    """Root-mean-square contrast of a single image."""

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, _ = self.prepare_inputs(prediction, reference, require_reference=False)
        centered = prediction_array - float(np.mean(prediction_array))
        return sqrt(float(np.mean(centered * centered)))

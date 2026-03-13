"""Reference image quality metrics."""

from __future__ import annotations

from math import inf, log10, sqrt
from typing import Any

import numpy as np

from .base import BaseMetric, gradient_magnitude, require_spatial_image


DEFAULT_GMSD_STABLE_CONSTANT = 0.0026


def _mean_squared_error(prediction: Any, reference: Any) -> float:
    difference = prediction - reference
    return float(np.mean(difference * difference))


def _resolve_data_range(reference: Any, configured_range: float | None) -> float:
    if configured_range is not None:
        if configured_range <= 0:
            raise ValueError("data_range must be strictly positive")
        return float(configured_range)

    reference_min = float(np.min(reference))
    reference_max = float(np.max(reference))
    dynamic_range = reference_max - reference_min
    if dynamic_range > 0:
        return dynamic_range

    maximum_magnitude = float(np.max(np.abs(reference)))
    return maximum_magnitude if maximum_magnitude > 0 else 1.0

class L1Metric(BaseMetric):
    """Mean absolute error between a prediction and a reference image."""

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, reference_array = self.prepare_inputs(prediction, reference)
        return float(np.mean(np.abs(prediction_array - reference_array)))


class MSEMetric(BaseMetric):
    """Mean squared error between a prediction and a reference image."""

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, reference_array = self.prepare_inputs(prediction, reference)
        return _mean_squared_error(prediction_array, reference_array)


class RMSEMetric(BaseMetric):
    """Root mean squared error between a prediction and a reference image."""

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, reference_array = self.prepare_inputs(prediction, reference)
        return sqrt(_mean_squared_error(prediction_array, reference_array))


class NMSEMetric(BaseMetric):
    """Normalized mean squared error between a prediction and a reference image."""

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, reference_array = self.prepare_inputs(prediction, reference)
        numerator = float(np.sum((prediction_array - reference_array) ** 2))
        denominator = float(np.sum(reference_array**2))
        if denominator == 0.0:
            return 0.0 if numerator == 0.0 else inf
        return numerator / denominator


class PSNRMetric(BaseMetric):
    """Peak signal-to-noise ratio in decibels."""

    def __init__(self, data_range: float | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.data_range = data_range

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, reference_array = self.prepare_inputs(prediction, reference)
        mse = _mean_squared_error(prediction_array, reference_array)
        if mse == 0.0:
            return inf
        data_range = _resolve_data_range(reference_array, self.data_range)
        return 20.0 * log10(data_range) - 10.0 * log10(mse)


class SSIMMetric(BaseMetric):
    """Global structural similarity index."""

    def __init__(
        self,
        data_range: float | None = None,
        k1: float = 0.01,
        k2: float = 0.03,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if k1 <= 0 or k2 <= 0:
            raise ValueError("k1 and k2 must be strictly positive")
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, reference_array = self.prepare_inputs(prediction, reference)
        data_range = _resolve_data_range(reference_array, self.data_range)

        mu_prediction = float(np.mean(prediction_array))
        mu_reference = float(np.mean(reference_array))
        sigma_prediction = float(np.var(prediction_array))
        sigma_reference = float(np.var(reference_array))
        sigma_cross = float(
            np.mean((prediction_array - mu_prediction) * (reference_array - mu_reference))
        )

        c1 = (self.k1 * data_range) ** 2
        c2 = (self.k2 * data_range) ** 2
        numerator = (2.0 * mu_prediction * mu_reference + c1) * (2.0 * sigma_cross + c2)
        denominator = (
            (mu_prediction**2 + mu_reference**2 + c1)
            * (sigma_prediction + sigma_reference + c2)
        )
        return 1.0 if denominator == 0.0 else float(numerator / denominator)


class UQIMetric(BaseMetric):
    """Universal image quality index."""

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, reference_array = self.prepare_inputs(prediction, reference)

        mu_prediction = float(np.mean(prediction_array))
        mu_reference = float(np.mean(reference_array))
        variance_prediction = float(np.var(prediction_array))
        variance_reference = float(np.var(reference_array))
        covariance = float(
            np.mean((prediction_array - mu_prediction) * (reference_array - mu_reference))
        )

        denominator = (
            (variance_prediction + variance_reference)
            * (mu_prediction * mu_prediction + mu_reference * mu_reference)
        )
        if denominator == 0.0:
            return 1.0 if np.array_equal(prediction_array, reference_array) else 0.0
        return float((4.0 * covariance * mu_prediction * mu_reference) / denominator)


class SREMetric(BaseMetric):
    """Signal-to-reconstruction error in decibels."""

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, reference_array = self.prepare_inputs(prediction, reference)
        mse = _mean_squared_error(prediction_array, reference_array)
        reference_power = float(np.mean(reference_array**2))
        if mse == 0.0:
            return inf
        if reference_power == 0.0:
            return -inf
        return 10.0 * log10(reference_power / mse)


class GMSDMetric(BaseMetric):
    """Gradient magnitude similarity deviation."""

    def __init__(
        self,
        data_range: float | None = None,
        stable_constant: float = DEFAULT_GMSD_STABLE_CONSTANT,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if stable_constant <= 0:
            raise ValueError("stable_constant must be strictly positive")
        self.data_range = data_range
        self.stable_constant = stable_constant

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, reference_array = self.prepare_inputs(prediction, reference)
        require_spatial_image(prediction_array, "GMSDMetric")
        data_range = _resolve_data_range(reference_array, self.data_range)

        prediction_gradient = gradient_magnitude(prediction_array)
        reference_gradient = gradient_magnitude(reference_array)
        constant = (self.stable_constant * data_range) ** 2
        similarity = (2.0 * prediction_gradient * reference_gradient + constant) / (
            prediction_gradient * prediction_gradient
            + reference_gradient * reference_gradient
            + constant
        )
        return float(np.std(similarity))


class LPIPSMetric(BaseMetric):
    """Learned perceptual image patch similarity via torchmetrics.

    Inputs are converted to NCHW tensors with 3 channels and normalized to
    ``[-1, 1]``.
    """

    def __init__(
        self,
        net_type: str = "alex",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        import torch
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.net_type = net_type
        self._torch = torch
        self._metric = LearnedPerceptualImagePatchSimilarity(
            net_type=self.net_type,
            normalize=False,
        )

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, reference_array = self.prepare_inputs(prediction, reference)
        prediction_batch = self._prepare_lpips_array(prediction_array)
        reference_batch = self._prepare_lpips_array(reference_array)
        value = self._metric(
            self._torch.as_tensor(prediction_batch, dtype=self._torch.float32),
            self._torch.as_tensor(reference_batch, dtype=self._torch.float32),
        )
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    def _prepare_lpips_array(self, image: Any) -> Any:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = array[None, :, :]
        elif array.ndim == 3:
            if array.shape[0] not in (1, 3):
                if array.shape[-1] not in (1, 3):
                    raise ValueError("LPIPS expects one or three channels")
                array = np.moveaxis(array, -1, 0)
        elif array.ndim == 4:
            if array.shape[1] not in (1, 3):
                if array.shape[-1] not in (1, 3):
                    raise ValueError("LPIPS expects one or three channels")
                array = np.moveaxis(array, -1, 1)
        else:
            raise ValueError("LPIPS expects 2D, 3D or 4D image tensors")

        if array.ndim == 3:
            if array.shape[0] not in (1, 3):
                raise ValueError("LPIPS expects one or three channels")
            array = array[None, :, :, :]
        elif array.shape[1] not in (1, 3):
            raise ValueError("LPIPS expects one or three channels")

        if array.shape[1] == 1:
            array = np.repeat(array, 3, axis=1)

        minimum = float(np.min(array))
        maximum = float(np.max(array))
        # torchmetrics LPIPS with normalize=False expects [-1, 1].
        # Accept common [0, 1] inputs and map them to the expected range,
        # but reject any other range to avoid silently changing contrast.
        if 0.0 <= minimum and maximum <= 1.0:
            array = 2.0 * array - 1.0
        elif minimum < -1.0 or maximum > 1.0:
            raise ValueError(
                "LPIPS expects values in [0, 1] or [-1, 1]; "
                f"got range [{minimum}, {maximum}]"
            )
        return array


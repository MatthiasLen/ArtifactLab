"""Reference image quality metrics."""

from __future__ import annotations

from importlib import import_module
from math import inf, log10, sqrt
from typing import Any, Callable

from .base import BaseMetric, _require_numpy

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised via runtime guard.
    np = None


def _mean_squared_error(prediction: Any, reference: Any) -> float:
    _require_numpy()
    difference = prediction - reference
    return float(np.mean(difference * difference))


def _resolve_data_range(reference: Any, configured_range: float | None) -> float:
    _require_numpy()
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


def _require_spatial_image(array: Any, metric_name: str) -> None:
    if array.ndim < 2:
        raise ValueError(f"{metric_name} expects image inputs with at least two dimensions")


def _gradient_magnitude(array: Any) -> Any:
    _require_numpy()
    gradient_x = np.diff(array, axis=-1, append=array[..., -1:])
    gradient_y = np.diff(array, axis=-2, append=array[..., -1:, :])
    return np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)


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

    def __init__(self, data_range: float | None = None, stable_constant: float = 0.0026, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if stable_constant <= 0:
            raise ValueError("stable_constant must be strictly positive")
        self.data_range = data_range
        self.stable_constant = stable_constant

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, reference_array = self.prepare_inputs(prediction, reference)
        _require_spatial_image(prediction_array, "GMSDMetric")
        data_range = _resolve_data_range(reference_array, self.data_range)

        prediction_gradient = _gradient_magnitude(prediction_array)
        reference_gradient = _gradient_magnitude(reference_array)
        constant = (self.stable_constant * data_range) ** 2
        similarity = (2.0 * prediction_gradient * reference_gradient + constant) / (
            prediction_gradient * prediction_gradient
            + reference_gradient * reference_gradient
            + constant
        )
        return float(np.std(similarity))


class LPIPSMetric(BaseMetric):
    """Learned perceptual image patch similarity with optional backend injection.

    Inputs are reshaped into batched channel-first tensors and normalized to the
    ``[-1, 1]`` range expected by common LPIPS backends.
    """

    def __init__(
        self,
        net_type: str = "alex",
        backend: Callable[[Any, Any], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.net_type = net_type
        self._backend = backend

    def apply_metric(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        del kwargs
        prediction_array, reference_array = self.prepare_inputs(prediction, reference)
        backend = self._backend or self._resolve_backend()
        prediction_batch = self._prepare_lpips_array(prediction_array)
        reference_batch = self._prepare_lpips_array(reference_array)
        return float(backend(prediction_batch, reference_batch))

    def _prepare_lpips_array(self, image: Any) -> Any:
        _require_numpy()
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
        if minimum < -1.0 or maximum > 1.0:
            # LPIPS backends expect image tensors normalized to [-1, 1]. For
            # constant images outside that range, returning zeros preserves the
            # "no contrast" signal without introducing artificial differences.
            if maximum == minimum:
                array = np.zeros_like(array)
            else:
                array = 2.0 * (array - minimum) / (maximum - minimum) - 1.0
        return array

    def _resolve_backend(self) -> Callable[[Any, Any], float]:
        try:
            import torch

            torchmetrics_module = None
            for module_name in ("torchmetrics.image.lpip", "torchmetrics.image.lpips"):
                try:
                    torchmetrics_module = import_module(module_name)
                    break
                except ImportError:
                    continue
            if torchmetrics_module is None:
                raise ImportError
            learned_metric = getattr(
                torchmetrics_module,
                "LearnedPerceptualImagePatchSimilarity",
            )
            metric = learned_metric(net_type=self.net_type, normalize=True)

            def _torchmetrics_backend(prediction: Any, reference: Any) -> float:
                return float(
                    metric(
                        torch.as_tensor(prediction, dtype=torch.float32),
                        torch.as_tensor(reference, dtype=torch.float32),
                    )
                    .detach()
                    .cpu()
                    .item()
                )

            return _torchmetrics_backend
        except ImportError:
            pass

        try:
            import lpips
            import torch

            metric = lpips.LPIPS(net=self.net_type)

            def _lpips_backend(prediction: Any, reference: Any) -> float:
                return float(
                    metric(
                        torch.as_tensor(prediction, dtype=torch.float32),
                        torch.as_tensor(reference, dtype=torch.float32),
                    )
                    .detach()
                    .cpu()
                    .mean()
                    .item()
                )

            return _lpips_backend
        except ImportError as error:
            raise ImportError(
                "LPIPSMetric requires torchmetrics or lpips with torch installed, "
                "or an explicit backend callable"
            ) from error

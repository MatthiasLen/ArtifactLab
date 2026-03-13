"""Shared interfaces for image quality metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised via runtime guard.
    np = None


def _require_numpy() -> None:
    if np is None:  # pragma: no cover - numpy is a project dependency.
        raise ImportError("Image metrics require numpy to be installed")


def require_spatial_image(array: Any, metric_name: str) -> None:
    """Require image-like inputs with explicit spatial dimensions."""

    if array.ndim < 2:
        raise ValueError(f"{metric_name} expects image inputs with at least two dimensions")


def gradient_magnitude(array: Any) -> Any:
    """Compute a simple finite-difference gradient magnitude map."""

    _require_numpy()
    gradient_x = np.diff(array, axis=-1, append=array[..., -1:])
    gradient_y = np.diff(array, axis=-2, append=array[..., -1:, :])
    return np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)


class BaseMetric(ABC):
    """Compact base interface for reference and non-reference image metrics."""

    def __init__(
        self,
        prediction_field: str = "prediction",
        reference_field: str = "target",
    ) -> None:
        self.prediction_field = prediction_field
        self.reference_field = reference_field

    def __call__(self, prediction: Any, reference: Any | None = None, **kwargs: Any) -> float:
        """Alias :meth:`apply_metric` for compact call-site syntax."""

        return self.apply_metric(prediction, reference, **kwargs)

    @abstractmethod
    def apply_metric(
        self,
        prediction: Any,
        reference: Any | None = None,
        **kwargs: Any,
    ) -> float:
        """Compute the metric value."""

    def prepare_inputs(
        self,
        prediction: Any,
        reference: Any | None = None,
        *,
        require_reference: bool = True,
    ) -> tuple[Any, Any | None]:
        """Resolve sample dictionaries, convert arrays and validate inputs."""

        _require_numpy()
        if isinstance(prediction, dict):
            sample = prediction
            prediction = self._require_field(sample, self.prediction_field)
            if reference is None and self.reference_field in sample:
                reference = sample[self.reference_field]

        if isinstance(reference, dict):
            reference = self._require_field(reference, self.reference_field)

        if prediction is None:
            raise ValueError("Metric input prediction must not be None")
        prediction_array = self.to_numpy(prediction)
        reference_array = self.to_numpy(reference) if reference is not None else None

        self._validate_array(prediction_array, "prediction")

        if require_reference:
            if reference_array is None:
                raise ValueError("Reference metric requires a reference image")
            self._validate_array(reference_array, "reference")
            self.ensure_same_shape(prediction_array, reference_array)
        elif reference_array is not None:
            self._validate_array(reference_array, "reference")
            self.ensure_same_shape(prediction_array, reference_array)

        return prediction_array, reference_array

    def to_numpy(self, data: Any) -> Any:
        """Convert tensors or array-like inputs into finite NumPy arrays."""

        if data is None:
            return data
        _require_numpy()
        if hasattr(data, "detach") and hasattr(data, "cpu") and hasattr(data, "numpy"):
            data = data.detach().cpu().numpy()
        array = np.asarray(data)
        if np.iscomplexobj(array):
            array = np.abs(array)
        return np.asarray(array, dtype=np.float32)

    def ensure_same_shape(self, prediction: Any, reference: Any) -> None:
        """Raise when prediction and reference shapes differ."""

        if prediction.shape != reference.shape:
            raise ValueError(
                "Prediction and reference must share the same shape, "
                f"got {prediction.shape} and {reference.shape}"
            )

    def _require_field(self, sample: dict[str, Any], field: str) -> Any:
        if field not in sample:
            raise KeyError(f"Sample does not contain field {field!r}")
        return sample[field]

    def _validate_array(self, array: Any, name: str) -> None:
        if getattr(array, "ndim", 0) == 0:
            raise ValueError(f"Metric input {name} must be at least one-dimensional")
        if array.size == 0:
            raise ValueError(f"Metric input {name} must not be empty")
        if np is not None and not np.isfinite(array).all():
            raise ValueError(f"Metric input {name} must contain only finite values")

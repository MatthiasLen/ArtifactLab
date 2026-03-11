"""Wrappers around selected DeepInverse reconstruction algorithms."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from .base import BaseReconstructor

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised via runtime guard.
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - exercised via runtime guard.
    torch = None


class DeepInverseReconstructor(BaseReconstructor):
    """Wrap a small set of DeepInverse MRI reconstructor implementations.

    The class currently supports three well-known algorithms exposed by the
    DeepInverse toolbox: ``VarNet``, ``MoDL`` and the generic optimization
    builder ``deepinv.optim.optim_builder``.
    """

    _ALGORITHM_REGISTRY = {
        "varnet": ("models", "VarNet"),
        "modl": ("models", "MoDL"),
        "optim_builder": ("optim", "optim_builder"),
    }

    def __init__(
        self,
        algorithm: str = "varnet",
        physics: Any | None = None,
        model: Any | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.algorithm = algorithm
        self.physics = physics
        self.model = model
        self.model_kwargs = model_kwargs or {}

    @classmethod
    def available_algorithms(cls) -> tuple[str, ...]:
        """Return the supported DeepInverse algorithm identifiers."""

        return ("varnet", "modl", "optim_builder")

    def build_model(self) -> Any:
        """Construct and cache the selected DeepInverse model."""

        if self.model is not None:
            return self.model

        deepinv_module = self._import_deepinv()
        registry_entry = self._ALGORITHM_REGISTRY.get(self.algorithm.lower())
        if registry_entry is None:
            available = ", ".join(self.available_algorithms())
            raise ValueError(
                f"Unsupported DeepInverse algorithm {self.algorithm!r}. Choose from {available}."
            )
        module_name, symbol_name = registry_entry

        try:
            source_module = getattr(deepinv_module, module_name)
            model_class = getattr(source_module, symbol_name)
        except AttributeError as error:
            raise AttributeError(
                f"deepinv.{module_name} does not expose the required symbol {symbol_name!r}."
            ) from error

        self.model = model_class(**self.model_kwargs)
        return self.model

    def apply_reconstruction(
        self,
        sample: dict[str, Any],
        physics: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply the selected DeepInverse model to the provided k-space data."""

        selected_physics = self.physics if physics is None else physics
        model = self.build_model()
        if hasattr(model, "eval"):
            model.eval()

        measurement = self._to_model_input(self.get_kspace(sample))
        mask = self.get_mask(sample)
        if mask is not None:
            mask = self._to_model_input(mask)

        errors: list[str] = []
        for call in self._candidate_calls(model, measurement, mask, selected_physics, kwargs):
            try:
                result = call()
                return self.to_numpy(self._unwrap_output(result))
            except TypeError as error:
                errors.append(str(error))
                continue

        details = "; ".join(filter(None, errors)) or "no compatible call signature matched"
        raise TypeError(
            "Unable to call the selected DeepInverse model with the provided sample and "
            f"options: {details}."
        )

    def _candidate_calls(
        self,
        model: Any,
        measurement: Any,
        mask: Any,
        physics: Any | None,
        extra_kwargs: dict[str, Any],
    ) -> list[Any]:
        calls = []
        keyword_args = dict(extra_kwargs)
        if mask is not None:
            keyword_args["mask"] = mask

        if physics is not None:
            calls.append(lambda: model(measurement, physics=physics, **keyword_args))
            calls.append(lambda: model(y=measurement, physics=physics, **keyword_args))
        calls.append(lambda: model(measurement, **keyword_args))
        calls.append(lambda: model(y=measurement, **keyword_args))
        return calls

    def _to_model_input(self, data: Any) -> Any:
        if torch is not None:
            if isinstance(data, torch.Tensor):
                return data
            return torch.as_tensor(data)
        if np is None:
            return data
        return np.asarray(data)

    def _unwrap_output(self, result: Any) -> Any:
        if isinstance(result, dict):
            for key in ("reconstruction", "output", "x"):
                if key in result:
                    return result[key]
        if isinstance(result, (list, tuple)) and result:
            return result[0]
        return result

    def _import_deepinv(self) -> ModuleType:
        try:
            return importlib.import_module("deepinv")
        except ImportError as error:
            raise ImportError(
                "DeepInverseReconstructor requires the optional deepinv package to "
                "build models. Install deepinv or inject a pre-built model instance."
            ) from error

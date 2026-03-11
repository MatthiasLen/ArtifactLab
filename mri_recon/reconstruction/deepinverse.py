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


class DeepInverseRAMReconstructor(BaseReconstructor):
    """Compact wrapper around DeepInverse RAM for MRI reconstruction.

    This class is intentionally independent from ``DeepInverseReconstructor`` to
    provide a clean, RAM-only implementation while preserving the legacy
    multi-algorithm wrapper for backward compatibility.
    """

    def __init__(
        self,
        physics: Any | None = None,
        model: Any | None = None,
        model_kwargs: dict[str, Any] | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize a RAM reconstructor.

        Args:
            physics: Optional DeepInverse MRI physics operator. If omitted,
                physics is inferred from each sample.
            model: Optional pre-built RAM model instance.
            model_kwargs: Optional keyword arguments used to build RAM.
            device: Torch device identifier used for physics/model tensors.

        Returns:
            None.
        """

        super().__init__()
        self.physics = physics
        self.model = model
        self.model_kwargs = model_kwargs or {}
        self.device = device

    def build_model(self) -> Any:
        """Build and cache the DeepInverse RAM model.

        Args:
            None.

        Returns:
            The instantiated RAM model.
        """

        if self.model is not None:
            return self.model

        deepinv_module = self._import_deepinv_module()
        try:
            ram_class = getattr(deepinv_module.models, "RAM")
        except AttributeError as error:
            raise AttributeError(
                "deepinv.models does not expose the required symbol 'RAM'."
            ) from error

        # Default to pretrained RAM unless explicitly overridden by caller.
        combined_kwargs = {"pretrained": True}
        combined_kwargs.update(self.model_kwargs)
        self.model = ram_class(**combined_kwargs)
        return self.model

    @classmethod
    def build_mri_physics(
        cls,
        sample: dict[str, Any],
        device: str = "cpu",
    ) -> Any:
        """Build a DeepInverse MRI physics operator from one sample.

        Args:
            sample: Dataset sample containing ``kspace`` and optional ``mask``
                and ``target`` fields.
            device: Torch device identifier used by DeepInverse physics.

        Returns:
            A configured ``deepinv.physics.MRI`` operator.
        """

        if torch is None:
            raise ImportError(
                "DeepInverse RAM reconstruction requires torch. Install CPU "
                "torch first and then install dependencies from requirements.txt."
            )

        deepinv_module = cls._import_deepinv_module()
        measurement = sample.get("kspace")
        if measurement is None:
            raise KeyError("Sample does not contain field 'kspace'")

        img_shape = cls._infer_img_size(sample)
        mask = cls._prepare_mri_mask(sample.get("mask"), img_shape, device)
        return deepinv_module.physics.MRI(mask=mask, img_size=img_shape, device=device)

    def apply_reconstruction(
        self,
        sample: dict[str, Any],
        physics: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Reconstruct one image with DeepInverse RAM.

        Args:
            sample: Dataset sample containing at least ``kspace``.
            physics: Optional physics override. If omitted, uses constructor
                physics or builds physics from the sample.
            **kwargs: Extra keyword arguments forwarded to RAM forward call.

        Returns:
            NumPy-converted RAM reconstruction output.
        """

        model = self.build_model()
        if hasattr(model, "eval"):
            model.eval()

        selected_physics = physics or self.physics
        if selected_physics is None:
            selected_physics = self.build_mri_physics(sample, device=self.device)

        measurement = self._prepare_measurement(self.get_kspace(sample))

        # RAM supports both positional and named measurement signatures. Try
        # both to remain compatible across DeepInverse releases.
        calls = [
            lambda: model(measurement, selected_physics, **kwargs),
            lambda: model(y=measurement, physics=selected_physics, **kwargs),
        ]
        errors: list[str] = []
        for call in calls:
            try:
                result = call()
                return self.to_numpy(self._unwrap_output(result))
            except TypeError as error:
                errors.append(str(error))

        details = "; ".join(filter(None, errors)) or "no compatible call signature matched"
        raise TypeError(
            "Unable to call DeepInverse RAM with the provided sample/options: "
            f"{details}."
        )

    def to_magnitude_image(self, reconstruction: Any) -> Any:
        """Convert RAM output into magnitude image representation.

        Args:
            reconstruction: Raw model output, tensor or array-like.

        Returns:
            Magnitude image as a NumPy array when conversion is possible.
        """

        array = self.to_numpy(reconstruction)
        if array is None or np is None:
            return array
        if array.ndim == 4 and array.shape[1] == 2:
            return np.abs(array[:, 0] + 1j * array[:, 1])
        if array.ndim == 3 and array.shape[0] == 2:
            return np.abs(array[0] + 1j * array[1])
        if np.iscomplexobj(array):
            return np.abs(array)
        return array

    def _prepare_measurement(self, data: Any) -> Any:
        """Convert sample k-space to RAM-compatible tensor shape.

        Args:
            data: K-space data from dataset sample.

        Returns:
            A torch tensor in ``(B, 2, H, W)`` format for complex MRI data,
            or the original array-like object when torch is unavailable.
        """

        measurement = self._to_model_input(data)
        if torch is None or not isinstance(measurement, torch.Tensor):
            return measurement

        # DeepInverse MRI physics expects real-valued tensors with explicit
        # real/imaginary channels rather than native complex tensors.
        if measurement.is_complex():
            measurement = torch.view_as_real(measurement)
            if measurement.ndim == 3:
                measurement = measurement.permute(2, 0, 1)
            elif measurement.ndim == 4:
                measurement = measurement.permute(0, 3, 1, 2)
            measurement = measurement.float()

        # Add batch dimension for single-slice tensors.
        if measurement.ndim == 3:
            measurement = measurement.unsqueeze(0)
        return measurement

    def _to_model_input(self, data: Any) -> Any:
        """Convert incoming data to model input format.

        Args:
            data: Tensor/array-like input.

        Returns:
            Torch tensor when torch is available, otherwise NumPy array.
        """

        if torch is not None:
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            return torch.as_tensor(data, device=self.device)
        if np is None:
            return data
        return np.asarray(data)

    @staticmethod
    def _unwrap_output(result: Any) -> Any:
        """Extract reconstruction payload from common model return formats.

        Args:
            result: Model return value.

        Returns:
            Tensor/array payload representing reconstructed image.
        """

        if isinstance(result, dict):
            for key in ("reconstruction", "output", "x"):
                if key in result:
                    return result[key]
        if isinstance(result, (list, tuple)) and result:
            return result[0]
        return result

    @staticmethod
    def _import_deepinv_module() -> ModuleType:
        """Import DeepInverse and raise a clear optional-dependency error.

        Args:
            None.

        Returns:
            Imported ``deepinv`` module.
        """

        try:
            return importlib.import_module("deepinv")
        except ImportError as error:
            raise ImportError(
                "DeepInverseRAMReconstructor requires the optional deepinv "
                "package. Install deepinv or inject a pre-built RAM model."
            ) from error

    @staticmethod
    def _infer_img_size(sample: dict[str, Any]) -> tuple[int, int]:
        """Infer image size from target or k-space fields.

        Args:
            sample: Input sample dictionary.

        Returns:
            ``(height, width)`` image shape inferred from sample content.
        """

        target = sample.get("target")
        if target is not None:
            target_shape = getattr(target, "shape", None)
            if target_shape is not None and len(target_shape) >= 2:
                return int(target_shape[-2]), int(target_shape[-1])
        measurement = sample.get("kspace")
        measurement_shape = getattr(measurement, "shape", None)
        if measurement_shape is None or len(measurement_shape) < 2:
            raise ValueError("Unable to infer MRI image size from the provided sample.")
        return int(measurement_shape[-2]), int(measurement_shape[-1])

    @staticmethod
    def _prepare_mri_mask(mask: Any, img_shape: tuple[int, int], device: str) -> Any:
        """Prepare mask tensor for DeepInverse MRI physics.

        Args:
            mask: Optional mask from sample.
            img_shape: Inferred image shape ``(H, W)``.
            device: Torch device identifier.

        Returns:
            Float mask tensor with shape ``(1, 1, H, W)``.
        """

        if torch is None:
            raise ImportError(
                "DeepInverse MRI physics creation requires torch. Install CPU "
                "torch first and then install dependencies from requirements.txt."
            )
        if mask is None:
            return torch.ones((1, 1, *img_shape), dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=device)
        while mask_tensor.ndim < 4:
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_tensor


class DeepInverseReconstructor(BaseReconstructor):
    """Wrap a small set of DeepInverse MRI reconstructor implementations.

    The class currently supports three well-known algorithms exposed by the
    DeepInverse toolbox: the pretrained foundation model ``RAM``, ``VarNet``,
    ``MoDL`` and the generic optimization builder ``deepinv.optim.optim_builder``.
    It also exposes the most relevant documented pretrained DeepInverse models
    for reconstruction workflows via :meth:`available_pretrained_models` and
    :meth:`load_pretrained_model`.
    """

    _ALGORITHM_REGISTRY = {
        "ram": ("models", "RAM"),
        "varnet": ("models", "VarNet"),
        "modl": ("models", "MoDL"),
        "optim_builder": ("optim", "optim_builder"),
    }
    _PRETRAINED_MODEL_REGISTRY = {
        "ram": ("models", "RAM", {"pretrained": True}),
        "drunet": ("models", "DRUNet", {"pretrained": "download"}),
        "dncnn": ("models", "DnCNN", {"pretrained": "download"}),
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

        return ("ram", "varnet", "modl", "optim_builder")

    @classmethod
    def available_pretrained_models(cls) -> tuple[str, ...]:
        """Return the documented pretrained DeepInverse model identifiers.

        ``ram`` is a direct pretrained reconstructor, while ``drunet`` and
        ``dncnn`` are pretrained denoiser backbones commonly used inside
        iterative DeepInverse reconstruction pipelines.
        """

        return ("ram", "drunet", "dncnn")

    @classmethod
    def load_pretrained_model(cls, model_name: str, **model_kwargs: Any) -> Any:
        """Load a documented pretrained DeepInverse model or backbone."""

        deepinv_module = cls._import_deepinv_module()
        registry_entry = cls._PRETRAINED_MODEL_REGISTRY.get(model_name.lower())
        if registry_entry is None:
            available = ", ".join(cls.available_pretrained_models())
            raise ValueError(
                f"Unsupported pretrained DeepInverse model {model_name!r}. "
                f"Choose from {available}."
            )
        module_name, symbol_name, default_kwargs = registry_entry
        try:
            source_module = getattr(deepinv_module, module_name)
            model_class = getattr(source_module, symbol_name)
        except AttributeError as error:
            raise AttributeError(
                f"deepinv.{module_name} does not expose the required symbol {symbol_name!r}."
            ) from error

        combined_kwargs = dict(default_kwargs)
        combined_kwargs.update(model_kwargs)
        return model_class(**combined_kwargs)

    @classmethod
    def build_mri_physics(
        cls,
        sample: dict[str, Any],
        device: str = "cpu",
    ) -> Any:
        """Build a simple DeepInverse MRI physics operator for a FastMRI sample."""

        if torch is None:
            raise ImportError(
                "DeepInverse MRI physics creation requires torch. Install CPU torch first "
                "and then install dependencies from requirements.txt."
            )

        deepinv_module = cls._import_deepinv_module()
        target = sample.get("target")
        measurement = sample.get("kspace")
        if measurement is None:
            raise KeyError("Sample does not contain field 'kspace'")

        img_shape = cls._infer_img_size(target=target, measurement=measurement)
        mask = cls._prepare_mri_mask(sample.get("mask"), img_shape, device=device)
        return deepinv_module.physics.MRI(mask=mask, img_size=img_shape, device=device)

    def build_model(self) -> Any:
        """Construct and cache the selected DeepInverse model."""

        if self.model is not None:
            return self.model

        deepinv_module = self._import_deepinv_module()
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

        if self.algorithm.lower() == "ram":
            combined_kwargs = {"pretrained": False}
            combined_kwargs.update(self.model_kwargs)
            self.model = model_class(**combined_kwargs)
        else:
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

        measurement = self._prepare_measurement(self.get_kspace(sample), physics=selected_physics)
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
        keyword_variants = [dict(extra_kwargs)]
        if mask is not None:
            with_mask = dict(extra_kwargs)
            with_mask["mask"] = mask
            # Try explicit-mask variants first, then fallback variants without mask.
            keyword_variants = [with_mask, dict(extra_kwargs)]

        # for compatibility with a range of DeepInverse model signatures, try multiple plausible
        # combinations of positional and keyword arguments. The physics argument is only included
        # when it appears to be an MRI physics object, as some models may not expect or support it.
        # TODO: consider a more robust way to detect this...
        for keyword_args in keyword_variants:
            if physics is not None:
                calls.append(lambda keyword_args=keyword_args: model(measurement, physics=physics, **keyword_args))
                calls.append(lambda keyword_args=keyword_args: model(y=measurement, physics=physics, **keyword_args))
            calls.append(lambda keyword_args=keyword_args: model(measurement, **keyword_args))
            calls.append(lambda keyword_args=keyword_args: model(y=measurement, **keyword_args))
        return calls

    def _to_model_input(self, data: Any) -> Any:
        if torch is not None:
            if isinstance(data, torch.Tensor):
                return data
            return torch.as_tensor(data)
        if np is None:
            return data
        return np.asarray(data)

    def to_magnitude_image(self, reconstruction: Any) -> Any:
        """Convert a DeepInverse reconstruction into a magnitude image array."""

        array = self.to_numpy(reconstruction)
        if array is None or np is None:
            return array
        if array.ndim == 4 and array.shape[1] == 2:
            return np.abs(array[:, 0] + 1j * array[:, 1])
        if array.ndim == 3 and array.shape[0] == 2:
            return np.abs(array[0] + 1j * array[1])
        if np.iscomplexobj(array):
            return np.abs(array)
        return array

    def _prepare_measurement(self, data: Any, physics: Any | None) -> Any:
        measurement = self._to_model_input(data)
        if torch is None or not isinstance(measurement, torch.Tensor):
            return measurement
        if not self._is_mri_physics(physics):
            return measurement
        if measurement.is_complex():
            measurement = torch.view_as_real(measurement)
            if measurement.ndim == 3:
                measurement = measurement.permute(2, 0, 1)
            elif measurement.ndim == 4:
                measurement = measurement.permute(0, 3, 1, 2)
            measurement = measurement.float()
        if measurement.ndim == 3:
            measurement = measurement.unsqueeze(0)
        return measurement

    def _unwrap_output(self, result: Any) -> Any:
        if isinstance(result, dict):
            for key in ("reconstruction", "output", "x"):
                if key in result:
                    return result[key]
        if isinstance(result, (list, tuple)) and result:
            return result[0]
        return result

    @staticmethod
    def _import_deepinv_module() -> ModuleType:
        try:
            return importlib.import_module("deepinv")
        except ImportError as error:
            raise ImportError(
                "DeepInverseReconstructor requires the optional deepinv package to "
                "build models. Install deepinv or inject a pre-built model instance."
            ) from error

    @staticmethod
    def _is_mri_physics(physics: Any | None) -> bool:
        return physics is not None and physics.__class__.__name__ == "MRI"

    @staticmethod
    def _infer_img_size(target: Any, measurement: Any) -> tuple[int, int]:
        if target is not None:
            target_shape = getattr(target, "shape", None)
            if target_shape is not None and len(target_shape) >= 2:
                return int(target_shape[-2]), int(target_shape[-1])
        measurement_shape = getattr(measurement, "shape", None)
        if measurement_shape is None or len(measurement_shape) < 2:
            raise ValueError("Unable to infer MRI image size from the provided sample.")
        return int(measurement_shape[-2]), int(measurement_shape[-1])

    @staticmethod
    def _prepare_mri_mask(mask: Any, img_shape: tuple[int, int], device: str) -> Any:
        if torch is None:
            raise ImportError(
                "DeepInverse MRI physics creation requires torch. Install CPU torch first "
                "and then install dependencies from requirements.txt."
            )
        if mask is None:
            return torch.ones((1, 1, *img_shape), dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=device)
        while mask_tensor.ndim < 4:
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_tensor

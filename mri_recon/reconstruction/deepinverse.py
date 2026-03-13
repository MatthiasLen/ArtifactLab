"""Wrappers around selected DeepInverse reconstruction algorithms."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import numpy as np

from .base import BaseReconstructor

try:
    import torch
except ImportError:  # pragma: no cover - exercised via runtime guard.
    torch = None


class DeepInverseRAMReconstructor(BaseReconstructor):
    """Compact wrapper around DeepInverse RAM for MRI reconstruction.

    This class provides a focused RAM-only integration.
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
        """Infer MRI physics image size from sample fields.

        Args:
            sample: Input sample dictionary.

        Returns:
            ``(height, width)`` image shape inferred from sample content.
        """

        measurement = sample.get("kspace")
        measurement_shape = getattr(measurement, "shape", None)
        if measurement_shape is None or len(measurement_shape) < 2:
            raise ValueError("Unable to infer MRI image size from the provided sample.")
        # k-space defines the FFT grid used by the physics model. Some datasets
        # expose cropped targets that do not match this grid.
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

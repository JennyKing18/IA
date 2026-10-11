"""Analizar modelos A y B."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class LayerInfo:
    name: str
    type: str
    input_shape: list[int] | None
    output_shape: list[int] | None
    parameters: int
    operations: int | None
    onnx_status: str


SUPPORTED_ONNX_LAYERS = {
    "Conv1d",
    "Conv2d",
    "Linear",
    "ReLU",
    "BatchNorm1d",
    "BatchNorm2d",
    "MaxPool1d",
    "MaxPool2d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "Flatten",
    "Dropout",
}


def _parameter_count(layer: nn.Module) -> int:
    return sum(parameter.numel() for parameter in layer.parameters(recurse=False))


def _operation_count(layer: nn.Module, output: torch.Tensor | tuple[torch.Tensor, ...]) -> int | None:
    output_tensor = output[0] if isinstance(output, tuple) else output
    output_elements = output_tensor.numel()

    if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
        kernel_elements = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) and len(layer.kernel_size) == 1 else int(torch.tensor(layer.kernel_size).prod().item())
        return output_elements * (layer.in_channels // layer.groups) * kernel_elements * 2
    if isinstance(layer, nn.Linear):
        return output_elements * layer.in_features * 2
    if isinstance(layer, (nn.ReLU, nn.MaxPool1d, nn.MaxPool2d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)):
        return output_elements
    return None


def analyze_model(model: nn.Module, input_shape: tuple[int, ...] = (1, 1, 40, 101)) -> list[LayerInfo]:
    """Ejecuta un ejemplo y devuelve informacion de las capas con parametros propios."""
    records: list[LayerInfo] = []
    hooks: list[Any] = []

    def capture(name: str, layer: nn.Module):
        def hook(_layer: nn.Module, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            input_tensor = inputs[0]
            output_tensor = output[0] if isinstance(output, tuple) else output
            records.append(LayerInfo(
                name=name,
                type=type(layer).__name__,
                input_shape=list(input_tensor.shape) if isinstance(input_tensor, torch.Tensor) else None,
                output_shape=list(output_tensor.shape) if isinstance(output_tensor, torch.Tensor) else None,
                parameters=_parameter_count(layer),
                operations=_operation_count(layer, output) if isinstance(output_tensor, torch.Tensor) else None,
                onnx_status="compatible" if type(layer).__name__ in SUPPORTED_ONNX_LAYERS else "revisar",
            ))
        return hook

    for name, layer in model.named_modules():
        if name and not list(layer.children()):
            hooks.append(layer.register_forward_hook(capture(name, layer)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape, dtype=torch.float32))
    if was_training:
        model.train()
    for hook in hooks:
        hook.remove()
    return records


def summarize_model(model: nn.Module, input_shape: tuple[int, ...] = (1, 1, 40, 101)) -> dict[str, Any]:
    layers = analyze_model(model, input_shape)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "input_shape": list(input_shape),
        "output_shape": layers[-1].output_shape if layers else None,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "total_operations": sum(layer.operations or 0 for layer in layers),
        "onnx_review_required": [layer.name for layer in layers if layer.onnx_status != "compatible"],
        "layers": [asdict(layer) for layer in layers],
    }

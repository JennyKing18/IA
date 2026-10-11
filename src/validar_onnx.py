"""Exporta y valida los modelos A y B contra ONNX Runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.modelos import ModeloA, ModeloB

INPUT_SHAPE = (1, 1, 40, 101)
OUTPUT_SHAPE = (1, 10)
DEFAULT_OPSET = 17


def export_and_validate(name: str, model: torch.nn.Module, output_dir: Path, opset: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.onnx"
    model.eval()
    sample = torch.randn(INPUT_SHAPE, dtype=torch.float32)

    torch.onnx.export(
        model,
        sample,
        path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=False,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    with torch.no_grad():
        torch_output = model(sample).cpu().numpy()
    onnx_output = session.run(["logits"], {input_info.name: sample.numpy()})[0]
    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    mean_abs_diff = float(np.mean(np.abs(torch_output - onnx_output)))

    result = {
        "model": name,
        "file": str(path),
        "opset": opset,
        "input_name": input_info.name,
        "input_shape": input_info.shape,
        "output_name": output_info.name,
        "output_shape": output_info.shape,
        "expected_input_shape": list(INPUT_SHAPE),
        "expected_output_shape": list(OUTPUT_SHAPE),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "file_size_bytes": path.stat().st_size,
        "valid": (
            input_info.name == "input"
            and output_info.name == "logits"
            and input_info.shape == list(INPUT_SHAPE)
            and output_info.shape == list(OUTPUT_SHAPE)
            and max_abs_diff <= 1e-5
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("resultados/onnx_fase2"))
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    args = parser.parse_args()

    results = [
        export_and_validate("modelo_a", ModeloA(), args.output, args.opset),
        export_and_validate("modelo_b", ModeloB(), args.output, args.opset),
    ]
    (args.output / "validacion.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(
            f"{result['model']}: valid={result['valid']}, "
            f"input={result['input_shape']}, output={result['output_shape']}, "
            f"max_abs_diff={result['max_abs_diff']:.3g}, "
            f"size={result['file_size_bytes']} bytes"
        )
    if not all(result["valid"] for result in results):
        raise SystemExit("La validacion ONNX fallo")
    print(f"OK: {len(results)} modelos validados con opset {args.opset}")


if __name__ == "__main__":
    main()

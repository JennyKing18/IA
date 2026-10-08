"""Valida que las referencias guardadas coincidan con el preprocesamiento actual."""
import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import F_MAX, F_MIN, HOP_LENGTH, LOG_EPS, N_FFT, N_MELS, NUM_SAMPLES, SAMPLE_RATE, WIN_LENGTH
from src.generar_referencias import log_mel


def load_wav(path: Path) -> torch.Tensor:
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError(f"Formato inesperado en {path}")
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    return torch.from_numpy(samples.astype(np.float32) / 32768.0)


def fix_length(audio: torch.Tensor) -> torch.Tensor:
    if audio.numel() >= NUM_SAMPLES:
        return audio[:NUM_SAMPLES]
    return F.pad(audio, (0, NUM_SAMPLES - audio.numel()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, default=Path("resultados/referencias_preprocesamiento"))
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    contract = json.loads((args.references / "contrato.json").read_text(encoding="utf-8"))
    commands = contract["commands"]
    failures = []

    for command in commands:
        expected = np.load(args.references / f"{command}_logmel.npy")
        actual = log_mel(fix_length(load_wav(args.references / f"{command}.wav"))).numpy()
        max_abs_diff = float(np.max(np.abs(actual - expected)))
        mean_abs_diff = float(np.mean(np.abs(actual - expected)))
        valid = expected.shape == (N_MELS, NUM_SAMPLES // HOP_LENGTH + 1) and expected.dtype == np.float32 and np.isfinite(expected).all() and max_abs_diff <= args.atol
        print(f"{command}: max_abs_diff={max_abs_diff:.3g}, mean_abs_diff={mean_abs_diff:.3g}, valid={valid}")
        if not valid:
            failures.append(command)

    if failures:
        raise SystemExit(f"Referencias inválidas: {', '.join(failures)}")
    print(f"OK: {len(commands)} referencias validadas con tolerancia {args.atol:g}")


if __name__ == "__main__":
    main()

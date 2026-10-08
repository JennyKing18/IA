"""Genera audios y tensores de referencia para validar el preprocesamiento."""
import argparse
import json
import math
import shutil
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import F_MAX, F_MIN, HOP_LENGTH, LOG_EPS, N_FFT, N_MELS, NUM_SAMPLES, SAMPLE_RATE, WIN_LENGTH

REFERENCE_COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


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


def hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank() -> torch.Tensor:
    frequencies = torch.linspace(0, SAMPLE_RATE // 2, N_FFT // 2 + 1, dtype=torch.float64)
    points = mel_to_hz(torch.linspace(hz_to_mel(torch.tensor(F_MIN)), hz_to_mel(torch.tensor(F_MAX)), N_MELS + 2, dtype=torch.float64))
    differences = points[1:] - points[:-1]
    slopes = points[None, :] - frequencies[:, None]
    down = -slopes[:, :-2] / differences[:-1]
    up = slopes[:, 2:] / differences[1:]
    return torch.clamp(torch.minimum(down, up), min=0.0).float()


def log_mel(audio: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(WIN_LENGTH)
    spectrum = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    mel = mel_filterbank().T @ (spectrum.abs() ** 2)
    return torch.log(mel + LOG_EPS).float()


def find_reference(dataset_dir: Path, command: str) -> Path:
    candidates = sorted((dataset_dir / command).glob("*.wav"))
    if not candidates:
        raise FileNotFoundError(f"No hay WAV para el comando {command}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("resultados/referencias_preprocesamiento"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    contract = {
        "sample_rate": SAMPLE_RATE,
        "num_samples": NUM_SAMPLES,
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "window": "hann",
        "center": True,
        "pad_mode": "reflect",
        "n_mels": N_MELS,
        "f_min": F_MIN,
        "f_max": F_MAX,
        "mel_scale": "htk",
        "log_eps": LOG_EPS,
        "normalization": "none",
        "output_shape": [1, 1, N_MELS, NUM_SAMPLES // HOP_LENGTH + 1],
        "commands": REFERENCE_COMMANDS,
    }
    (args.output / "contrato.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    for command in REFERENCE_COMMANDS:
        source = find_reference(args.dataset, command)
        destination = args.output / f"{command}.wav"
        shutil.copy2(source, destination)
        tensor = log_mel(fix_length(load_wav(source)))
        np.save(args.output / f"{command}_logmel.npy", tensor.numpy())
        metadata = {
            "command": command,
            "source": str(source),
            "shape": list(tensor.shape),
            "dtype": str(tensor.numpy().dtype),
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "mean": float(tensor.mean()),
            "std": float(tensor.std()),
        }
        (args.output / f"{command}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"{command}: {tuple(tensor.shape)} -> {destination.name}, {command}_logmel.npy")


if __name__ == "__main__":
    main()

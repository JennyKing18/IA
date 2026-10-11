"""Modelos A y B de la Fase 2.
"""
from __future__ import annotations

import torch
from torch import nn

from src.config import COMMANDS, F_MAX, F_MIN, HOP_LENGTH, N_FFT, N_MELS, NUM_SAMPLES, NORMALIZATION_MEAN, NORMALIZATION_STD, SAMPLE_RATE, WIN_LENGTH

N_FRAMES = NUM_SAMPLES // HOP_LENGTH + 1
INPUT_SHAPE = (1, N_MELS, N_FRAMES)
NUM_CLASSES = len(COMMANDS)


class ModeloA(nn.Module):
    """LeNet-5 adaptada a espectrogramas log-mel."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        media: float = NORMALIZATION_MEAN,
        desv: float = NORMALIZATION_STD,
    ) -> None:
        super().__init__()
        self.register_buffer("media", torch.tensor(float(media)))
        self.register_buffer("desv", torch.tensor(float(desv)))

        alto = ((N_MELS - 4) // 2 - 4) // 2
        ancho = ((N_FRAMES - 4) // 2 - 4) // 2
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=(alto, ancho)),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.media) / self.desv
        return self.classifier(self.features(x))


class ConvBNAct(nn.Sequential):
    """Conv2d sin sesgo, BatchNorm2d y ReLU6."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 3, stride: int = 1, groups: int = 1) -> None:
        super().__init__(
            nn.Conv2d(c_in, c_out, kernel, stride, kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """Bloque MobileNetV2 con expansion, depthwise, proyeccion y atajo opcional."""

    def __init__(self, c_in: int, c_out: int, stride: int, expand: int) -> None:
        super().__init__()
        hidden = c_in * expand
        layers: list[nn.Module] = []
        if expand != 1:
            layers.append(ConvBNAct(c_in, hidden, kernel=1))
        layers.append(ConvBNAct(hidden, hidden, kernel=3, stride=stride, groups=hidden))
        layers.extend([nn.Conv2d(hidden, c_out, 1, bias=False), nn.BatchNorm2d(c_out)])
        self.block = nn.Sequential(*layers)
        self.use_res = stride == 1 and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block(x)
        return x + output if self.use_res else output


STEM_CHANNELS = 16
SETTINGS = [(1, 16, 1, 1), (4, 24, 2, 2), (4, 32, 2, 2), (4, 64, 1, 1)]
HEAD_CHANNELS = 128


class ModeloB(nn.Module):
    """Version reducida de MobileNetV2 para espectrogramas."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.2,
        media: float = NORMALIZATION_MEAN,
        desv: float = NORMALIZATION_STD,
    ) -> None:
        super().__init__()
        self.register_buffer("media", torch.tensor(float(media)))
        self.register_buffer("desv", torch.tensor(float(desv)))

        channels = STEM_CHANNELS
        layers: list[nn.Module] = [ConvBNAct(1, channels, kernel=3, stride=2)]
        for expansion, output_channels, repetitions, stride in SETTINGS:
            for repetition in range(repetitions):
                layers.append(InvertedResidual(channels, output_channels, stride if repetition == 0 else 1, expansion))
                channels = output_channels
        layers.append(ConvBNAct(channels, HEAD_CHANNELS, kernel=1))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(HEAD_CHANNELS, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing() and (x.dim() != 4 or tuple(x.shape[1:]) != INPUT_SHAPE):
            raise ValueError(f"Se esperaba (N, 1, {N_MELS}, {N_FRAMES}), llego {tuple(x.shape)}")
        x = (x - self.media) / self.desv
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))

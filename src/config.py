"""Constantes compartidas del proyecto.

Los valores de AUDIO son PROVISIONALES hasta que el equipo cierre el contrato
de entrada (ver docs/contrato_entrada.md). Cualquier cambio aquí debe
reflejarse también en la app móvil.
"""
import os
import random

COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABEL_TO_IDX = {c: i for i, c in enumerate(COMMANDS)}

# --- Contrato de entrada (PROVISIONAL) ---
SAMPLE_RATE = 16_000   # Hz
CLIP_SECONDS = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
N_FFT = 480            # 30 ms
WIN_LENGTH = 480       # 30 ms
HOP_LENGTH = 160       # 10 ms
N_MELS = 40
F_MIN = 20.0
F_MAX = 8_000.0
LOG_EPS = 1e-6

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fija todas las semillas para que los resultados sean reproducibles."""
    # Importados aquí para que COMMANDS y las constantes se puedan usar sin torch instalado.
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

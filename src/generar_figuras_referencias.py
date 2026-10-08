"""Genera una figura log-mel de referencia para cada comando."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, default=Path("resultados/referencias_preprocesamiento"))
    parser.add_argument("--output", type=Path, default=Path("resultados/ejemplos_espectrogramas"))
    args = parser.parse_args()

    commands = json.loads((args.references / "contrato.json").read_text(encoding="utf-8"))["commands"]
    args.output.mkdir(parents=True, exist_ok=True)
    for command in commands:
        tensor = np.load(args.references / f"{command}_logmel.npy")
        figure, axis = plt.subplots(figsize=(5, 3))
        image = axis.imshow(tensor, origin="lower", aspect="auto", cmap="magma")
        axis.set_title(command)
        axis.set_xlabel("tiempo (frames de 10 ms)")
        axis.set_ylabel("banda mel")
        figure.colorbar(image, ax=axis, label="log energía")
        figure.tight_layout()
        figure.savefig(args.output / f"{command}.png", dpi=150)
        plt.close(figure)
        print(f"{command}: {tuple(tensor.shape)} -> {command}.png")


if __name__ == "__main__":
    main()

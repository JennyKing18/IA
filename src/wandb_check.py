"""Prueba de conexión con Weights & Biases.

Sube un run corto con métricas inventadas y una matriz de confusión, para
confirmar que la clave y el equipo están bien configurados. No entrena nada.

Uso (desde la raíz del repo):
    Escritorio:  py -m src.wandb_check
    Colab:       después de la celda de login de 00_setup.ipynb,
                 from src.wandb_check import run_check; run_check()
"""
import argparse
import math
import os
import random

import wandb

from src.config import COMMANDS

PROJECT = "comandos-de-voz"


def run_check(entity=None, project=PROJECT, epochs=5):
    entity = entity or os.environ.get("WANDB_ENTITY")
    rng = random.Random(0)

    run = wandb.init(
        project=project,
        entity=entity,
        name=f"prueba-conexion-{rng.randint(1000, 9999)}",
        group="prueba-conexion",
        tags=["prueba"],
        config={"modelo": "ninguno", "datos": "inventados", "epocas": epochs},
    )

    for epoch in range(1, epochs + 1):
        wandb.log({
            "epoch": epoch,
            "train/loss": 2.3 * math.exp(-0.5 * epoch) + rng.uniform(0, 0.05),
            "val/loss": 2.3 * math.exp(-0.4 * epoch) + rng.uniform(0, 0.08),
            "val/accuracy": 1 - 0.9 * math.exp(-0.5 * epoch),
        })

    # Matriz de confusión con predicciones inventadas (80 % correctas).
    n_classes = len(COMMANDS)
    y_true = [rng.randrange(n_classes) for _ in range(300)]
    y_pred = [t if rng.random() < 0.8 else rng.randrange(n_classes) for t in y_true]
    wandb.log({"val/confusion_matrix": wandb.plot.confusion_matrix(
        y_true=y_true, preds=y_pred, class_names=COMMANDS)})

    run.summary["resultado"] = "conexion OK"
    url = run.url
    run.finish()

    print("\nListo. Abre este enlace y revisa que aparezcan las gráficas y la matriz de confusión:")
    print(url or "(modo offline: no se subió nada)")
    return url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", default=None, help="Equipo de W&B (por defecto: WANDB_ENTITY o tu cuenta personal)")
    args = parser.parse_args()
    run_check(entity=args.entity)

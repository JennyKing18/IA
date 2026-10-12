"""Entrenamiento de la Fase 3, configuración 3.

Ejecuta las 4 corridas que corresponden a la Persona 3:
- A_crudo_config3
- A_aumentado_config3
- B_crudo_config3
- B_aumentado_config3

Uso:
    py -m src.fase3_config3 --dataset-root "C:/ruta/con/raw_y_augmented"
    py -m src.fase3_config3 --raw-train ".../raw_train.pt" --raw-val ".../raw_val.pt" --aug-train ".../aug_train.pt" --aug-val ".../aug_val.pt"
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import COMMANDS, SEED, NORMALIZATION_MEAN, NORMALIZATION_STD
from src.modelos import ModeloA, ModeloB


def load_split(path: str | Path):
    """Carga un split con formato tolerante.

    Acepta:
    - tupla (X, y)
    - dict con claves X/y
    - dict con claves train/val/test
    """
    full_path = Path(path)
    if not full_path.exists():
        raise FileNotFoundError(f"No existe el archivo del split: {full_path}")

    obj = torch.load(full_path, map_location="cpu", weights_only=False)

    if isinstance(obj, tuple) and len(obj) == 2:
        return obj[0], obj[1]

    if isinstance(obj, dict):
        if "X" in obj and "y" in obj:
            return obj["X"], obj["y"]
        if "train" in obj and isinstance(obj["train"], (tuple, list)) and len(obj["train"]) == 2:
            return obj["train"][0], obj["train"][1]
        if "X_train" in obj and "y_train" in obj:
            return obj["X_train"], obj["y_train"]

    raise TypeError(f"Formato inesperado para {full_path}: {type(obj)}")


def load_dataset_pair(dataset_root: str | Path, raw_name: str = "raw", augmented_name: str = "aug"):
    root = Path(dataset_root)
    candidates = [
        root / f"{raw_name}_train.pt",
        root / f"{raw_name}_val.pt",
        root / f"{augmented_name}_train.pt",
        root / f"{augmented_name}_val.pt",
    ]
    for path in candidates:
        if not path.exists():
            continue

    raw_train = root / f"{raw_name}_train.pt"
    raw_val = root / f"{raw_name}_val.pt"
    aug_train = root / f"{augmented_name}_train.pt"
    aug_val = root / f"{augmented_name}_val.pt"

    return {
        "raw": {
            "train": load_split(raw_train),
            "val": load_split(raw_val),
        },
        "aug": {
            "train": load_split(aug_train),
            "val": load_split(aug_val),
        },
    }


def make_loader(X, y, batch_size, shuffle, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)


def confusion_matrix(y_true, y_pred, n_clases=len(COMMANDS)):
    cm = torch.zeros((n_clases, n_clases), dtype=torch.int64)
    for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
        cm[yt, yp] += 1
    return cm


def metrics_from_confusion(cm):
    cm = cm.numpy()
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "f1_macro": float(f1.mean()),
        "f1_por_clase": {c: float(v) for c, v in zip(COMMANDS, f1)},
    }


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    perdida, n_total, ys, ps = 0.0, 0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        salida = model(xb.unsqueeze(1))
        if criterion is not None:
            perdida += criterion(salida, yb).item() * len(yb)
        n_total += len(yb)
        ys.append(yb.cpu())
        ps.append(salida.argmax(1).cpu())
    y_true, y_pred = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    cm = confusion_matrix(torch.from_numpy(y_true), torch.from_numpy(y_pred))
    res = metrics_from_confusion(cm)
    res.update(loss=perdida / max(n_total, 1) if criterion is not None else float("nan"), y_true=y_true, y_pred=y_pred)
    return res


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    perdida, aciertos, n_total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        salida = model(xb.unsqueeze(1))
        loss = criterion(salida, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        perdida += loss.item() * len(yb)
        aciertos += (salida.argmax(1) == yb).sum().item()
        n_total += len(yb)
    return {"loss": perdida / n_total, "accuracy": aciertos / n_total}


def _guardar_atomico(obj, ruta):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, ruta)


def fit(model, X_train, y_train, X_val, y_val, *, epochs, lr, batch_size, weight_decay=0.0,
        optimizer_name="adam", seed=SEED, device=None, ckpt_dir=None, run_name="modeloB",
        run=None, resume=False, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    else:
        raise ValueError(f"optimizador desconocido: {optimizer_name}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = make_loader(X_train, y_train, batch_size, True, seed)
    val_loader = make_loader(X_val, y_val, batch_size * 2, False)

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]}
    best_f1, best_epoch, inicio = -1.0, 0, 1
    last_path = Path(ckpt_dir) / f"{run_name}_last.pt" if ckpt_dir else None
    best_path = Path(ckpt_dir) / f"{run_name}_best.pt" if ckpt_dir else None

    if resume and last_path is not None and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        hist, best_f1, best_epoch, inicio = ck["hist"], ck["best_f1"], ck["best_epoch"], ck["epoch"] + 1
        if verbose:
            print(f"Reanudando desde la época {inicio}")

    for epoch in range(inicio, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        for k, v in zip(hist, [tr["loss"], tr["accuracy"], va["loss"], va["accuracy"], va["f1_macro"]]):
            hist[k].append(v)

        if run is not None:
            run.log({
                "epoch": epoch,
                "train/loss": tr["loss"],
                "train/accuracy": tr["accuracy"],
                "val/loss": va["loss"],
                "val/accuracy": va["accuracy"],
                "val/f1_macro": va["f1_macro"],
                "lr": scheduler.get_last_lr()[0],
            })

        if va["f1_macro"] > best_f1:
            best_f1, best_epoch = va["f1_macro"], epoch
            if best_path is not None:
                _guardar_atomico({"model": model.state_dict(), "epoch": epoch, "val_f1": best_f1}, best_path)

        if last_path is not None:
            _guardar_atomico({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "hist": hist,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "epoch": epoch,
            }, last_path)

        if verbose:
            print(f"ép {epoch:3d}/{epochs} | train {tr['loss']:.3f}/{tr['accuracy']:.3f} | val {va['loss']:.3f}/{va['accuracy']:.3f} | F1 {va['f1_macro']:.3f}")

    if best_path is not None and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])

    if run is not None:
        va = evaluate(model, val_loader, device, criterion)
        run.log({
            "val/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=va["y_true"].tolist(),
                preds=va["y_pred"].tolist(),
                class_names=COMMANDS,
            )
        })
        run.summary["mejor_val_f1"] = best_f1
        run.summary["mejor_epoca"] = best_epoch

    return {"hist": hist, "best_f1": best_f1, "best_epoch": best_epoch}


def run_single_training(model_name: str, dataset_name: str, X_train, y_train, X_val, y_val, *, run_name: str, entity: str | None, project: str = "comandos-de-voz"):
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group="fase3-config3",
        tags=["fase3", dataset_name, model_name, "config3"],
        config={
            "modelo": model_name,
            "dataset": dataset_name,
            "optimizer": "adam",
            "lr": 3e-4,
            "batch_size": 64,
            "weight_decay": 1e-4,
            "epochs": 50,
            "dropout": 0.10 if model_name == "B" else 0.0,
            "seed": SEED,
        },
    )

    model = ModeloA(media=NORMALIZATION_MEAN, desv=NORMALIZATION_STD) if model_name == "A" else ModeloB(media=NORMALIZATION_MEAN, desv=NORMALIZATION_STD, dropout=0.10)
    fit(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=50,
        lr=3e-4,
        batch_size=64,
        weight_decay=1e-4,
        optimizer_name="adam",
        seed=SEED,
        ckpt_dir="./checkpoints",
        run_name=run_name,
        run=run,
        verbose=True,
    )
    run.finish()


def main():
    parser = argparse.ArgumentParser(description="Entrena la config 3 de la Fase 3 para los 4 modelos asignados.")
    parser.add_argument("--dataset-root", type=str, default="./resultados", help="Carpeta donde están los splits crudos y aumentados (por ejemplo raw_train.pt, raw_val.pt, aug_train.pt, aug_val.pt)")
    parser.add_argument("--raw-train", type=str, default=None, help="Ruta a raw_train.pt")
    parser.add_argument("--raw-val", type=str, default=None, help="Ruta a raw_val.pt")
    parser.add_argument("--aug-train", type=str, default=None, help="Ruta a aug_train.pt")
    parser.add_argument("--aug-val", type=str, default=None, help="Ruta a aug_val.pt")
    parser.add_argument("--entity", type=str, default=None, help="Equipo de W&B; usa WANDB_ENTITY si no se pasa")
    parser.add_argument("--project", type=str, default="comandos-de-voz", help="Proyecto de W&B")
    args = parser.parse_args()

    entity = args.entity or os.environ.get("WANDB_ENTITY")

    if args.raw_train and args.raw_val and args.aug_train and args.aug_val:
        raw_train, raw_y_train = load_split(args.raw_train)
        raw_val, raw_y_val = load_split(args.raw_val)
        aug_train, aug_y_train = load_split(args.aug_train)
        aug_val, aug_y_val = load_split(args.aug_val)
        raw_sets = {"train": (raw_train, raw_y_train), "val": (raw_val, raw_y_val)}
        aug_sets = {"train": (aug_train, aug_y_train), "val": (aug_val, aug_y_val)}
    else:
        dataset_root = Path(args.dataset_root)
        if not dataset_root.exists():
            raise FileNotFoundError(f"No existe la carpeta de datasets: {dataset_root}")

        dataset_bundle = load_dataset_pair(dataset_root)
        raw_sets = dataset_bundle["raw"]
        aug_sets = dataset_bundle["aug"]

    # Se valida que existan los archivos y se cargan con el formato real.
    for key, value in {"raw": raw_sets, "aug": aug_sets}.items():
        if not all(item in value for item in ["train", "val"]):
            raise ValueError(f"Faltan los splits de {key} en el dataset: {value}")

    lista = [
        ("A", "crudo", "A_crudo_config3", raw_sets["train"], raw_sets["val"]),
        ("A", "aumentado", "A_aumentado_config3", aug_sets["train"], aug_sets["val"]),
        ("B", "crudo", "B_crudo_config3", raw_sets["train"], raw_sets["val"]),
        ("B", "aumentado", "B_aumentado_config3", aug_sets["train"], aug_sets["val"]),
    ]

    for model_name, dataset_name, run_name, train_bundle, val_bundle in lista:
        X_train, y_train = train_bundle
        X_val, y_val = val_bundle
        run_single_training(
            model_name=model_name,
            dataset_name=dataset_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            run_name=run_name,
            entity=entity,
            project=args.project,
        )

    print("\nListo. Corridas de la Fase 3 config 3 completadas:")
    for _, _, run_name, _, _ in lista:
        print(f"- {run_name}")


if __name__ == "__main__":
    main()

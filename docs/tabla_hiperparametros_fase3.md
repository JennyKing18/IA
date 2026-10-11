# Tabla compartida de hiperparametros

- Se ejecutan 12 corridas: 2 modelos x 2 datasets x 3 configuraciones.
- MEDIA = -6.5107 y DESV = 4.8010 se mantienen iguales en los dos modelos y datasets.
- El optimizador `sgd` usa momentum `0.9` y `nesterov=True`.
- La semilla base es `42`.

## Configuraciones

| Configuracion | Optimizador (`fit`) | Learning rate | Batch de entrenamiento | Weight decay | Epocas | Dropout Modelo B |
|---|---|---:|---:|---:|---:|---:|
| 1 | `adaptación automática de la tasa de aprendizaje` | `0.001` | 64 | `0` | 35 | `0.20` |
| 2 | `sgd` + Nesterov | `0.01` | 128 | `0.0001` | 45 | `0.30` |
| 3 | `adaptación automática de la tasa de aprendizaje` | `0.0003` | 64 | `0.0001` | 50 | `0.10` |

Puse esas épocas pero no se si son suficientes para que converja el modelo B, tal vez hay que meterle más.

## Corridas asignadas

| ID | Modelo | Dataset | Configuracion | Responsable | Nombre de corrida |
|---|---|---|---:|---|---|
| 1 | A | crudo | 1 | Jenny | `A_crudo_config1` |
| 2 | A | aumentado | 1 | Jenny | `A_aumentado_config1` |
| 3 | A | crudo | 2 | Jenny | `A_crudo_config2` |
| 4 | A | aumentado | 2 | Jenny | `A_aumentado_config2` |
| 5 | B | crudo | 1 | Kendall | `B_crudo_config1` |
| 6 | B | aumentado | 1 | Kendall | `B_aumentado_config1` |
| 7 | B | crudo | 2 | Kendall | `B_crudo_config2` |
| 8 | B | aumentado | 2 | Kendall | `B_aumentado_config2` |
| 9 | A | crudo | 3 | Isaac | `A_crudo_config3` |
| 10 | A | aumentado | 3 | Isaac | `A_aumentado_config3` |
| 11 | B | crudo | 3 | Isaac | `B_crudo_config3` |
| 12 | B | aumentado | 3 | Isaac | `B_aumentado_config3` |

## Como ejecutar una corrida

La llamada conceptual para cada corrida es:

```python
resultado = fit(
    modelo,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=config["epochs"],
    lr=config["lr"],
    batch_size=config["batch_size"],
    weight_decay=config["weight_decay"],
    optimizer_name=config["optimizer"],
    seed=42,
    ckpt_dir=CKPT_DIR,
    run_name="A_crudo_config1",
    run=wandb_run,
)
```

Para Modelo B, construir antes el modelo con dropout de la configuración:

```python
modelo = ModeloB(media=MEDIA, desv=DESV, dropout=config["dropout"])
```

Para Modelo A:

```python
modelo = ModeloA(media=MEDIA, desv=DESV)
```
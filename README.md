# Proyecto I: Reconocimiento de comandos de voz

Curso de Inteligencia Artificial, Instituto Tecnológico de Costa Rica.

Reconocimiento de 10 comandos (yes, no, up, down, left, right, on, off, stop, go) de Speech Commands v0.02 con dos modelos de PyTorch, exportados a ONNX y usados en la app móvil *Smart Home Voice Navigator*.

## Estructura

| Carpeta | Contenido |
|---|---|
| `src/` | Código reutilizable (`config.py`, preprocesamiento, dataset, modelos, entrenamiento). Los notebooks lo importan. |
| `notebooks/` | Notebooks por fase. `00_setup.ipynb` prepara el entorno. |
| `docs/` | Plan del proyecto y contrato de entrada del modelo. |
| `app/` | App móvil (Expo / React Native, pendiente de confirmar). |
| `informe/` | Fuente LaTeX del informe (plantilla IEEE). |

El dataset, los checkpoints y los `.onnx` viven en la carpeta compartida de Google Drive `ProyectoIA_ComandosDeVoz`, **no** en git.

## Primeros pasos

1. Clonar el repo (o abrir `notebooks/00_setup.ipynb` en Colab).
2. Instalar PyTorch desde pytorch.org y luego `pip install -r requirements.txt`.
3. Editar solo la celda de configuración de `00_setup.ipynb` y ejecutarlo completo.

## Reglas de trabajo con git

- Trabajar en ramas o en copias personales de los notebooks; los `.ipynb` se fusionan mal.
- `git pull` antes de `git push`.
- Todo cambio lo revisa alguien que no lo escribió.

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

1. **Drive (una vez):** en *Compartidos conmigo*, clic derecho en `ProyectoIA_ComandosDeVoz` → *Organizar → Agregar acceso directo* → *Mi unidad*. Así la ruta es la misma para las tres.
2. **W&B (una vez):** en Colab, icono de llave 🔑 → secreto `WANDB_API_KEY` (clave de https://wandb.ai/authorize) → activar *Acceso al notebook*.
3. **Escritorio:** clonar el repo, instalar PyTorch desde pytorch.org y luego `pip install -r requirements.txt`. Si Drive no está en `G:\Mi unidad` ni en `G:\My Drive`, definir la variable de entorno `DRIVE_ROOT`.
4. Abrir `notebooks/00_setup.ipynb` y ejecutarlo completo. No hay que editar nada.

## Reglas de trabajo con git

- **El código se escribe en VS Code y se sube desde la computadora.** Colab solo baja el código y entrena; no se hace push desde Colab.
- Trabajar en ramas o en copias personales de los notebooks; los `.ipynb` se fusionan mal.
- `git pull` antes de `git push`.
- Todo cambio lo revisa alguien que no lo escribió.

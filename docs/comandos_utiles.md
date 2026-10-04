# Comandos útiles

En Windows usar `py` en lugar de `python` si la terminal no lo reconoce.

## Git

```bash
git clone https://github.com/JennyKing18/IA.git   # primera vez
git pull                                          # antes de empezar a trabajar
git checkout -b fase1-preprocesamiento            # rama nueva
git status                                        # ver qué cambió
git add -A                                        # preparar todo
git commit -m "mensaje"                           # guardar
git push -u origin HEAD                           # subir la rama
git checkout main                                 # volver a main
```

## Python / entorno

```bash
py -m pip install -r requirements.txt   # librerías del proyecto
py -m pip install torch torchaudio      # PyTorch (CPU)
py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Weights & Biases

```bash
py -m wandb login            # una vez, pega tu clave de wandb.ai/authorize
py -m src.wandb_check        # prueba de conexión (desde la raíz del repo)
```

Colab: 🔑 Secrets → `WANDB_API_KEY` → activar "Acceso al notebook".

## Colab

```python
from google.colab import drive; drive.mount("/content/drive")   # montar Drive
!git -C /content/IA pull                                        # actualizar el código
!nvidia-smi                                                     # ver la GPU
```

Abrir notebook desde GitHub: Archivo → Abrir notebook → GitHub → `JennyKing18/IA`.
GPU: Entorno de ejecución → Cambiar tipo de entorno → T4 GPU.

# Comandos útiles

En Windows usar `py` en lugar de `python` si la terminal no lo reconoce.


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

Abrir notebook desde GitHub: Archivo → Abrir notebook → GitHub → `user/IA`.
GPU: Entorno de ejecución → Cambiar tipo de entorno → T4 GPU.

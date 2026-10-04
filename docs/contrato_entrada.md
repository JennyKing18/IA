# Contrato de entrada del modelo (BORRADOR)

> Estado: **propuesta, no acordada**. Las tres personas deben aprobarlo en la Fase 0.
> La app móvil debe reproducir estos pasos exactamente. Los valores viven en `src/config.py`.

| Paso | Valor propuesto | Notas |
|---|---|---|
| Formato de audio | WAV PCM 16 bits, mono | Igual que Speech Commands |
| Frecuencia de muestreo | 16 000 Hz | Si el micrófono graba a otra, remuestrear a 16 kHz |
| Duración | 1 s = 16 000 muestras | Rellenar con ceros al final si es más corto; recortar si es más largo |
| Normalización de onda | float32 en [-1, 1] | int16 / 32768 |
| Ventana | Hann, 480 muestras (30 ms) | `n_fft = 480` |
| Salto (hop) | 160 muestras (10 ms) | |
| Espectro | Potencia (`|STFT|^2`), `center=True`, relleno reflect | Confirmar que TypeScript replica el relleno |
| Banco mel | 40 bandas, 20–8000 Hz, escala HTK, sin normalización | |
| Logaritmo | `log(mel + 1e-6)` | |
| Normalización final | Por definir: media/desv. global del train o por ejemplo | Si es global, guardar los valores en el contrato |
| Forma del tensor | `[1, 1, 40, 101]` float32 (batch, canal, mel, frames) | 101 = 1 + 16000/160 con `center=True` |
| Salida | `[1, 10]` logits en el orden de `COMMANDS` | yes, no, up, down, left, right, on, off, stop, go |

## Preguntas abiertas
- ¿El espectrograma va dentro del grafo ONNX (simplifica la app) o se reimplementa en TypeScript? Consultar al profesor.
- ¿Clase extra de silencio/desconocido? Cambiaría la salida a `[1, 11]` o `[1, 12]`.

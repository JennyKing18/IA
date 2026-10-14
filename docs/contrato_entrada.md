# Contrato de entrada del modelo (DEFINITIVO)

## Por qué existe este contrato

El modelo no recibe sonido, sino un espectrograma: una imagen que muestra qué frecuencias suenan en cada momento. El modelo aprende a reconocer imágenes construidas de una forma exacta. Si la app construye la imagen de otra forma (otro tamaño, otra escala, otra normalización), el modelo falla en el celular aunque tenga buena exactitud en el notebook. Por eso la receta se fija una vez y no se cambia.

## Valores

| Paso | Valor | Notas |
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
| Normalización final | `(x - MEDIA) / DESV` con `MEDIA = -6.5107` y `DESV = 4.8010` (globales, calculadas solo con el train crudo) | Se aplica dentro del modelo como primera operación: viaja en el `.onnx` y la app NO la reimplementa. La app entrega el log-mel sin normalizar |
| Forma del tensor | `[1, 1, 40, 101]` float32 (batch, canal, mel, frames) | 101 = 1 + 16000/160 con `center=True` |
| Salida | `[1, 10]` logits en el orden de `COMMANDS` | yes, no, up, down, left, right, on, off, stop, go |

## Justificación

**Formato y frecuencia de muestreo (16 kHz, mono, 16 bits).** Es el formato en que viene grabado Speech Commands [1], así que no hace falta convertir el dataset. Por el teorema de muestreo, 16 kHz representa frecuencias de hasta 8 kHz, y la mayor parte de la información que permite distinguir palabras está por debajo de ese límite. Es la frecuencia habitual en reconocimiento de voz de banda ancha.

**Duración (1 s).** Los clips del dataset duran 1 segundo como máximo [1], y un comando corto ("yes", "stop") cabe entero. Los clips más cortos se rellenan con ceros para que todas las entradas tengan el mismo tamaño. En la app, el micrófono se analiza en ventanas de 1 segundo por la misma razón.

**Normalización de la onda a [-1, 1].** Dividir las muestras int16 entre 32768 deja la señal en una escala fija e independiente del formato de grabación. Es lo que hacen torchaudio y la mayoría de librerías de audio, y es fácil de repetir en TypeScript.

**Ventana de 30 ms con salto de 10 ms.** La voz cambia continuamente, pero en tramos de unos 20–30 ms se puede considerar casi estable, y en esos tramos tiene sentido calcular un espectro. Un salto de 10 ms hace que los tramos se solapen, así no se pierden los cambios rápidos entre sonidos. La combinación de una ventana de 25–30 ms con un salto de 10 ms es la configuración clásica en reconocimiento de voz; Sainath y Parada [2] usan 25 ms con salto de 10 ms para detección de palabras clave. Se usa la ventana de Hann para reducir la fuga espectral en los bordes de cada tramo.

**Escala mel.** El oído humano distingue bien diferencias pequeñas en frecuencias graves y mal en las agudas [3]. El banco de filtros mel reparte las bandas de la misma forma, más densas en los graves, y concentra la representación en la información útil para entender palabras. Las representaciones basadas en mel son el estándar en reconocimiento de voz desde Davis y Mermelstein [4].

**Logaritmo.** La percepción del volumen es aproximadamente logarítmica. Además, la energía del espectro varía en varios órdenes de magnitud: sin logaritmo, los sonidos fuertes dominarían la imagen y los débiles serían casi invisibles para la red. El término `1e-6` evita calcular `log(0)` en tramos en silencio. El log-mel es también la entrada sobre la que se define SpecAugment [5], la técnica de aumentación elegida, de modo que el contrato y la aumentación son compatibles.

**Normalización final.** Se estandariza con una media y una desviación globales, calculadas una sola vez con el conjunto de entrenamiento crudo (30 769 espectrogramas de 40 × 101) y usadas igual para el dataset crudo y el aumentado, para que la comparación sea justa. No se calculan por ejemplo para que el resultado no dependa de la duración ni del volumen de cada clip, y para que el modelo exportado se comporte igual en el notebook y en el celular. Las constantes están en el modelo como *buffers* de PyTorch, así que quedan dentro del archivo `.onnx`.

**40 bandas mel.** Es el número de bandas que usan trabajos de detección de palabras clave pensados para dispositivos con recursos limitados [2]. Da suficiente resolución para distinguir 10 comandos y mantiene la imagen pequeña, lo que reduce el cálculo y la latencia en el celular. La alternativa común es 64 bandas: más detalle a cambio de una entrada más grande.

**Forma del tensor `[1, 1, 40, 101]`.** Es consecuencia directa de los valores anteriores: un ejemplo por lote, un canal (como una imagen en escala de grises), 40 bandas y 101 columnas de tiempo (16 000 / 160 = 100 saltos, más uno por el relleno de `center=True`). Una entrada de 40 × 101 es de un orden de magnitud cercano a la entrada de 32 × 32 para la que se diseñó LeNet-5 [6], así que el Modelo A requiere pocos cambios respecto a la arquitectura original.

**Salida de 10 logits.** Una salida por comando, en el orden fijo de `COMMANDS`. La app aplica softmax para obtener probabilidades y puede descartar predicciones con poca confianza.

## Qué está fijo y qué es elección del equipo

- **Fijado por el dataset:** 16 kHz, mono, 1 segundo.
- **Elección del equipo (valores estándar, pero podrían ser otros):** 30 ms de ventana, 40 bandas y estandarización global dentro del modelo. Lo importante es decidirlos una vez, justificarlos y no cambiarlos después.

## Modelo exportado (Fase 5)

| Campo | Valor |
|---|---|
| Modelo | `B_aumentado_config1`, mejor época 32 (F1 macro: 0.971 val, 0.974 test) |
| Archivos | `models/modelo_final.onnx` (grafo, 86 KB) y `models/modelo_final.onnx.data` (pesos, 190 KB). Deben estar **juntos en la misma carpeta**: el `.onnx` busca los pesos por nombre |
| Opset / IR | 18 / 10 (torch 2.11, onnx 1.23.2, onnxruntime 1.31.0) |
| Entrada | `input`, `float32`, `[1, 1, 40, 101]`: log-mel **sin normalizar** |
| Salida | `logits`, `float32`, `[1, 10]`, orden `yes, no, up, down, left, right, on, off, stop, go` |
| Operadores | Add, Clip, Conv, Div, Gemm, ReduceMean, Reshape, Sub (todos estándar) |
| sha256 del `.onnx` | `655f887f6e2e563756df879eea9d1fd6a67965e6151cca6e8b5613a12e33a8e2` |
| Validación | ONNX Runtime contra PyTorch en los 4 074 audios de test: 100 % de predicciones iguales, diferencia máxima de logits 3.1e-05 (`models/validacion_independiente.json`) |

Para pasar de logits a comando: `softmax` → índice del mayor valor → `COMMANDS[índice]`. La probabilidad del máximo sirve como confianza.

## Receta exacta para la app

La app tiene que construir el mismo tensor que el notebook (sección 1.4). Paso a paso:

1. **Audio:** muestras int16 mono a 16 kHz → `x[n] = int16 / 32768` (float32).
2. **Largo fijo:** quedarse con las primeras 16 000 muestras; si hay menos, rellenar con ceros **al final**.
3. **Relleno reflect:** agregar 240 muestras (`n_fft / 2`) a cada lado, reflejando sin repetir el borde: a la izquierda `x[240], …, x[1]` y a la derecha `x[15998], …, x[15759]`. Largo resultante: 16 480.
4. **Tramos:** 101 tramos de 480 muestras; el tramo `t` empieza en `t · 160` (t = 0 … 100).
5. **Ventana de Hann periódica:** `w[n] = 0.5 − 0.5 · cos(2πn / 480)`, n = 0 … 479. Es la versión *periódica* (divide entre 480, no entre 479).
6. **Espectro de potencia:** FFT real de cada tramo ventaneado → 241 coeficientes `X[k]` (k = 0 … 240); `P[k] = Re² + Im²`. Sin escalar.
7. **Banco mel (HTK, 40 filtros, sin normalizar):**
   - `mel(f) = 2595 · log10(1 + f / 700)` y su inversa `f(m) = 700 · (10^(m / 2595) − 1)`.
   - 42 puntos equiespaciados en mel entre `mel(20)` y `mel(8000)`, convertidos a Hz: `f₀ … f₄₁`.
   - Frecuencia de cada coeficiente: `fₖ = k · 8000 / 240` (k = 0 … 240).
   - Filtro `m` (m = 0 … 39): `peso[k][m] = max(0, min((fₖ − f_m) / (f_{m+1} − f_m), (f_{m+2} − fₖ) / (f_{m+2} − f_{m+1})))`.
   - `mel[m][t] = Σₖ peso[k][m] · P[k][t]`.
8. **Logaritmo natural:** `logmel[m][t] = ln(mel[m][t] + 1e-6)`.
9. **Tensor:** `Float32Array` de 4 040 valores en orden `[banda][tramo]` (índice `m · 101 + t`) con forma `[1, 1, 40, 101]`. **No** normalizar: `(x − MEDIA) / DESV` ya está dentro del modelo.

**Cómo comprobarlo:** `models/referencias_app.json` trae, para 10 audios del dataset (uno por comando), el log-mel exacto que calcula Python y los logits del ONNX. Si la app calcula esos mismos audios y la diferencia con el log-mel de referencia es menor que 1e-3, el preprocesamiento es idéntico.

## Decisiones cerradas

- **Espectrograma fuera del ONNX.** El modelo recibe el log-mel y la app lo calcula con la receta de arriba. Si el profesor permite meterlo en el grafo, la app se simplifica, pero el modelo actual no cambia.
- **Sin clase de silencio/desconocido.** La salida queda en 10 comandos (`[1, 10]`). Para no reaccionar a ruido, la app puede descartar predicciones con baja confianza (umbral sobre la probabilidad del softmax).

## Referencias

[1] P. Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition," arXiv:1804.03209, 2018.

[2] T. N. Sainath y C. Parada, "Convolutional Neural Networks for Small-footprint Keyword Spotting," en *Proc. Interspeech*, 2015.

[3] S. S. Stevens, J. Volkmann y E. B. Newman, "A Scale for the Measurement of the Psychological Magnitude Pitch," *J. Acoust. Soc. Am.*, vol. 8, no. 3, pp. 185–190, 1937.

[4] S. Davis y P. Mermelstein, "Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 28, no. 4, pp. 357–366, 1980.

[5] D. S. Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," en *Proc. Interspeech*, 2019.

[6] Y. LeCun, L. Bottou, Y. Bengio y P. Haffner, "Gradient-Based Learning Applied to Document Recognition," *Proc. IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998.
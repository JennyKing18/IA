# Contrato de entrada del modelo (DEFINITIVO)

## Por qué existe este contrato

El modelo no recibe sonido, sino un espectrograma: una imagen que muestra qué frecuencias suenan en cada momento. El modelo aprende a reconocer imágenes construidas de una forma exacta. Si la app construye la imagen de otra forma (otro tamaño, otra escala, otra normalización), el modelo falla en el celular aunque tenga buena exactitud en el notebook. Por eso la receta se fija una vez y no se cambia.

## Valores

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
| Normalización final | Ninguna | El modelo recibe directamente el log-mel calculado en el paso anterior |
| Forma del tensor | `[1, 1, 40, 101]` float32 (batch, canal, mel, frames) | 101 = 1 + 16000/160 con `center=True` |
| Salida | `[1, 10]` logits en el orden de `COMMANDS` | yes, no, up, down, left, right, on, off, stop, go |

## Justificación

**Formato y frecuencia de muestreo (16 kHz, mono, 16 bits).** Es el formato en que viene grabado Speech Commands [1], así que no hace falta convertir el dataset. Por el teorema de muestreo, 16 kHz representa frecuencias de hasta 8 kHz, y la mayor parte de la información que permite distinguir palabras está por debajo de ese límite. Es la frecuencia habitual en reconocimiento de voz de banda ancha.

**Duración (1 s).** Los clips del dataset duran 1 segundo como máximo [1], y un comando corto ("yes", "stop") cabe entero. Los clips más cortos se rellenan con ceros para que todas las entradas tengan el mismo tamaño. En la app, el micrófono se analiza en ventanas de 1 segundo por la misma razón.

**Normalización de la onda a [-1, 1].** Dividir las muestras int16 entre 32768 deja la señal en una escala fija e independiente del formato de grabación. Es lo que hacen torchaudio y la mayoría de librerías de audio, y es fácil de repetir en TypeScript.

**Ventana de 30 ms con salto de 10 ms.** La voz cambia continuamente, pero en tramos de unos 20–30 ms se puede considerar casi estable, y en esos tramos tiene sentido calcular un espectro. Un salto de 10 ms hace que los tramos se solapen, así no se pierden los cambios rápidos entre sonidos. La combinación de una ventana de 25–30 ms con un salto de 10 ms es la configuración clásica en reconocimiento de voz; Sainath y Parada [2] usan 25 ms con salto de 10 ms para detección de palabras clave. Se usa la ventana de Hann para reducir la fuga espectral en los bordes de cada tramo.

**Escala mel.** El oído humano distingue bien diferencias pequeñas en frecuencias graves y mal en las agudas [3]. El banco de filtros mel reparte las bandas de la misma forma, más densas en los graves, y concentra la representación en la información útil para entender palabras. Las representaciones basadas en mel son el estándar en reconocimiento de voz desde Davis y Mermelstein [4].

**Logaritmo.** La percepción del volumen es aproximadamente logarítmica. Además, la energía del espectro varía en varios órdenes de magnitud: sin logaritmo, los sonidos fuertes dominarían la imagen y los débiles serían casi invisibles para la red. El término `1e-6` evita calcular `log(0)` en tramos en silencio. El log-mel es también la entrada sobre la que se define SpecAugment [5], la técnica de aumentación elegida, de modo que el contrato y la aumentación son compatibles.

**40 bandas mel.** Es el número de bandas que usan trabajos de detección de palabras clave pensados para dispositivos con recursos limitados [2]. Da suficiente resolución para distinguir 10 comandos y mantiene la imagen pequeña, lo que reduce el cálculo y la latencia en el celular. La alternativa común es 64 bandas: más detalle a cambio de una entrada más grande.

**Forma del tensor `[1, 1, 40, 101]`.** Es consecuencia directa de los valores anteriores: un ejemplo por lote, un canal (como una imagen en escala de grises), 40 bandas y 101 columnas de tiempo (16 000 / 160 = 100 saltos, más uno por el relleno de `center=True`). Una entrada de 40 × 101 es de un orden de magnitud cercano a la entrada de 32 × 32 para la que se diseñó LeNet-5 [6], así que el Modelo A requiere pocos cambios respecto a la arquitectura original.

**Salida de 10 logits.** Una salida por comando, en el orden fijo de `COMMANDS`. La app aplica softmax para obtener probabilidades y puede descartar predicciones con poca confianza.

## Qué está fijo y qué es elección del equipo

- **Fijado por el dataset:** 16 kHz, mono, 1 segundo.
- **Elección del equipo (valores estándar, pero podrían ser otros):** 30 ms de ventana, 40 bandas y ausencia de normalización final. Lo importante es decidirlos una vez, justificarlos y no cambiarlos después.

## Preguntas abiertas

- ¿El espectrograma va dentro del grafo ONNX (simplifica la app) o se reimplementa en TypeScript? Consultar al profesor.
- ¿Clase extra de silencio/desconocido? Cambiaría la salida a `[1, 11]` o `[1, 12]`.

## Referencias

[1] P. Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition," arXiv:1804.03209, 2018.

[2] T. N. Sainath y C. Parada, "Convolutional Neural Networks for Small-footprint Keyword Spotting," en *Proc. Interspeech*, 2015.

[3] S. S. Stevens, J. Volkmann y E. B. Newman, "A Scale for the Measurement of the Psychological Magnitude Pitch," *J. Acoust. Soc. Am.*, vol. 8, no. 3, pp. 185–190, 1937.

[4] S. Davis y P. Mermelstein, "Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences," *IEEE Trans. Acoust., Speech, Signal Process.*, vol. 28, no. 4, pp. 357–366, 1980.

[5] D. S. Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," en *Proc. Interspeech*, 2019.

[6] Y. LeCun, L. Bottou, Y. Bengio y P. Haffner, "Gradient-Based Learning Applied to Document Recognition," *Proc. IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998.

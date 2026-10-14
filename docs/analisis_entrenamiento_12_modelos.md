# Análisis de los 12 modelos y selección del modelo final

## 1. Protocolo

Se entrenaron 12 modelos: 2 arquitecturas (A: LeNet-5, B: MobileNetV2 reducido) × 2 datasets de entrenamiento (crudo; aumentado con SpecAugment) × 3 configuraciones de hiperparámetros. Todos usan la misma semilla (42), la misma normalización (MEDIA = −6.5107, DESV = 4.8010), entropía cruzada y una tasa de aprendizaje con decaimiento coseno hasta 0. Validación y test son siempre los conjuntos crudos de las listas oficiales (3 703 y 4 074 audios).

| Config. | Optimizador | LR inicial | Batch | Weight decay | Épocas | Dropout (solo B) |
|---|---|---:|---:|---:|---:|---:|
| 1 | Adam | 0.001 | 64 | 0 | 35 | 0.20 |
| 2 | SGD + Nesterov (momentum 0.9) | 0.01 | 128 | 0.0001 | 45 | 0.30 |
| 3 | Adam | 0.0003 | 64 | 0.0001 | 50 | 0.10 |

**Regla de selección:**

1. En cada entrenamiento se guarda la época con mayor F1 macro de validación (`_best.pt`).
2. De cada combinación modelo × dataset pasa a test solo la configuración con mayor F1 de validación (4 candidatos).
3. Cada candidato se evalúa en test una sola vez.
4. El modelo final es el candidato con mayor F1 de validación; en empate, el de menor tamaño y luego el más rápido.

Test no participa en ninguna decisión: si se usara para elegir, su resultado dejaría de ser una estimación independiente del desempeño real.

## 2. Resultados de validación (12 modelos)

| Modelo | Dataset | Config. | Mejor F1 val | Mejor época | Época del mínimo de val loss | Val loss mín. → final | Brecha acc. train − val en la mejor época |
|---|---|---:|---:|---:|---:|---|---:|
| B | aumentado | 1 | **0.9707** | 32 | 15 | 0.110 → 0.130 | +2.4 % |
| B | aumentado | 2 | 0.9674 | 33 | 29 | 0.111 → 0.121 | +1.5 % |
| B | crudo | 1 | 0.9637 | 35 | 8 | 0.133 → 0.171 | +3.5 % |
| B | crudo | 2 | 0.9607 | 36 | 19 | 0.129 → 0.139 | +3.3 % |
| B | aumentado | 3 | 0.9603 | 21 | 11 | 0.129 → 0.170 | +1.8 % |
| B | crudo | 3 | 0.9550 | 22 | 19 | 0.145 → 0.175 | +3.1 % |
| A | aumentado | 2 | 0.9025 | 21 | 8 | 0.341 → 0.438 | +9.7 % |
| A | crudo | 2 | 0.8900 | 42 | 10 | 0.397 → 0.454 | +11.1 % |
| A | aumentado | 3 | 0.8691 | 47 | 8 | 0.447 → 0.689 | +13.2 % |
| A | aumentado | 1 | 0.8667 | 6 | 6 | 0.411 → 0.957 | +0.1 % |
| A | crudo | 3 | 0.8631 | 38 | 11 | 0.460 → 0.645 | +13.7 % |
| A | crudo | 1 | 0.8546 | 29 | 8 | 0.495 → 0.814 | +14.5 % |

Los 6 Modelos B superan a los 6 Modelos A: el peor B (0.955) está 5 puntos por encima del mejor A (0.902). Datos completos y curvas en W&B (proyecto `comandos-de-voz`, grupos `fase3-config1/2/3`) y en `resultados/fase4/tabla_12_modelos.csv`.

## 3. Candidatos, test y modelo final

| Candidato | F1 train | F1 val | F1 test | Val → test | Parámetros | Tamaño (fp32) | ms / predicción* |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B_aumentado_config1** | 0.999 | **0.971** | **0.974** | +0.3 | 50 410 | 245 KB | 4.6 |
| B_crudo_config1 | 0.999 | 0.964 | 0.966 | +0.2 | 50 410 | 245 KB | 4.5 |
| A_aumentado_config2 | 1.000 | 0.902 | 0.887 | −1.6 | 309 386 | 1.2 MB | 1.5 |
| A_crudo_config2 | 1.000 | 0.890 | 0.869 | −2.1 | 309 386 | 1.2 MB | 1.6 |

\* Mediana de 300 predicciones de un espectrograma, CPU de escritorio con 1 hilo. Train, val y test se midieron con el mismo checkpoint `_best.pt`.

**Modelo final: `B_aumentado_config1`.** Tiene el mayor F1 de validación, sin empate (0.33 puntos sobre el segundo), y en test obtiene 0.974, un resultado igual al de validación. Que el puntaje no baje de val a test indica que la selección no se ajustó al conjunto de validación por casualidad.

## 4. Por qué gana el Modelo B

### 4.1 En qué aspectos gana

| Aspecto | Modelo A (mejor) | Modelo B (final) |
|---|---|---|
| F1 en test | 0.887 | 0.974 (+8.7 puntos) |
| Errores en test (de 4 074) | 464 | 105 (4.4 veces menos) |
| Pérdida al pasar de val a test | −1.6 puntos | +0.3 puntos |
| Brecha train − val en la mejor época | 10–15 % | 1.5–3.5 % |
| Peor comando en test (F1) | *no*, 0.81 | *off*, 0.963 |
| Parámetros / tamaño | 309 386 / 1.2 MB | 50 410 / 245 KB (6 veces menos) |
| Tiempo por predicción | 1.5 ms | 4.6 ms |

B gana en todo salvo en velocidad. Tiene más operaciones (6.44 M MACs frente a 2.31 M) y sus convoluciones depthwise aprovechan mal la CPU. Aun así, 4.6 ms es menos del 0.5 % de la duración del audio (1 s), así que la diferencia no importa para la aplicación.

### 4.2 Causas

Las diferencias vienen de la arquitectura, no de los hiperparámetros: B gana con las 3 configuraciones.

- **Invariancia a la posición.** En un clip de 1 s, la palabra puede empezar en cualquier momento. B termina con un *promedio global*, que resume cada mapa de características sin importar dónde apareció el patrón. En A, la capa C5 usa un kernel de 7 × 22 que cubre todo el mapa de S4, así que funciona como una capa densa con un peso distinto para cada posición. Para reconocer una palabra al inicio y al final del clip, A tiene que aprender pesos separados para cada caso.
- **Dónde están los parámetros.** C5 tiene 295 800 de los 309 386 parámetros de A (95.6 %). Casi toda la capacidad del modelo está en esa capa dependiente de la posición, y es ahí donde memoriza el entrenamiento. Los 50 410 parámetros de B están repartidos en filtros pequeños que se reutilizan en todo el espectrograma.
- **Regularización y entrenamiento estable.** B usa BatchNorm después de cada convolución y dropout antes de la capa final. A no tiene ninguno de los dos, y usa tanh, que se satura con valores grandes y frena el gradiente.
- **Profundidad.** B encadena 19 convoluciones, con bloques residuales invertidos y atajos. Al final, cada posición ve 59 × 59 (todas las bandas y más de la mitad del clip), y el promedio global junta todo el clip. A solo tiene dos convoluciones de 5 × 5 antes de C5, así que sus filtros ven parches de 16 × 16 y el resto lo resuelve C5 de forma densa.

## 5. Qué pasa con el Modelo A

**Sobreajuste en las 6 corridas.** La accuracy de train llega a 1.000 y la loss de train baja a 0.0002–0.0095. La val loss toca su mínimo entre las épocas 6 y 11 y después sube (hasta 0.96). El F1 de validación se estanca en 0.85–0.90 y, en test, pierde otros 1.6–2.1 puntos.

**Qué configuración lo controla mejor:**

- **Config. 2 (SGD) es la mejor para A**, y le saca 2.5–3.5 puntos a las otras. Es la que más controla el sobreajuste: su val loss solo sube de 0.34 a 0.44, frente a 0.41 → 0.96 con la config. 1. Ayudan el weight decay (1e-4 en SGD es una penalización L2 directa), el batch de 128 y la menor velocidad de memorización de SGD frente a Adam.
- **Config. 1 (Adam, lr 0.001) es la peor.** Con el dataset aumentado, su mejor época fue la **6 de 35**. A partir de ahí el modelo memorizó el entrenamiento y la validación solo empeoró. La brecha de +0.1 % de esa fila no significa que esté bien: el modelo se eligió antes de que empezara a sobreajustar.

**Comandos que confunde** (A_aumentado_config2, test):

| Real → predicho | Audios |
|---|---:|
| go → no | 51 |
| no → go | 38 |
| down → no | 29 |
| off → up | 22 |
| off → on | 19 |
| left → yes | 15 |

Son palabras de una sílaba que comparten la vocal (*go*/*no*/*down*: "o"; *off*/*on*/*up*: vocal corta inicial) y se distinguen por una consonante corta y de poca energía (/g/ frente a /n/, /f/ frente a /n/). En el espectrograma la vocal domina, y un modelo que no generaliza bien termina decidiendo por ella.

## 6. Por qué la configuración 3 no llega

La config. 3 nunca fue la mejor en ninguna combinación: queda segunda o tercera en las cuatro, aunque tiene la mayor cantidad de épocas (50). Hay tres razones:

1. **Tasa de aprendizaje baja con decaimiento coseno.** Empieza en 0.0003, entre 3 y 33 veces menos que las otras, y el coseno la sigue bajando hasta 0. A las 10 épocas su F1 es parecido al de las demás (B: 0.950–0.955; A: 0.851–0.857), pero en las épocas siguientes avanza muy poco: en las corridas B la mejor época es la 21–22 y no mejora en las 28 restantes. Más épocas no compensan un paso pequeño que encima se sigue achicando.
2. **Menos regularización implícita.** Con un paso grande, el ruido del gradiente evita que el modelo se acomode en mínimos muy ajustados al entrenamiento, que generalizan peor. Con lr 0.0003 ese efecto casi desaparece. Por eso la config. 3 no reduce la brecha train − val: 13 % en A, igual que la config. 1, y 1.8–3.1 % en B, igual que las demás.
3. **Weight decay débil con Adam.** En `torch.optim.Adam`, el weight decay se suma al gradiente y después Adam lo escala por coordenada, así que la penalización real es menor que en SGD (Loshchilov y Hutter, 2019). La versión desacoplada, AdamW, la aplicaría completa. Por eso, en A, la config. 3 sobreajusta casi igual que la 1: su val loss sube de 0.45 a 0.69.

En A, la config. 3 supera por poco a la config. 1 (0.863 frente a 0.855 en crudo), solo porque sobreajusta más despacio. En B queda última porque B no tiene un problema de sobreajuste que una tasa baja pueda corregir, y en cambio sí sufre la convergencia más lenta.

## 7. Efecto de la aumentación

| Modelo | Config. | F1 val crudo | F1 val aumentado | Diferencia |
|---|---:|---:|---:|---:|
| A | 1 | 0.8546 | 0.8667 | +1.21 |
| A | 2 | 0.8900 | 0.9025 | +1.25 |
| A | 3 | 0.8631 | 0.8691 | +0.60 |
| B | 1 | 0.9637 | 0.9707 | +0.70 |
| B | 2 | 0.9607 | 0.9674 | +0.67 |
| B | 3 | 0.9550 | 0.9603 | +0.53 |

La aumentación mejora las 6 comparaciones, y en test se confirma: +1.7 puntos en A y +0.8 en B. Las máscaras de frecuencia y de tiempo obligan al modelo a no depender de una sola banda ni de un solo instante, y eso funciona como regularización. Ayuda más a A porque A es el que más sobreajusta.

En las corridas aumentadas, durante las primeras épocas la métrica de validación supera a la de entrenamiento (en B_aumentado_config1, hasta la época ~13). Esto es esperado por tres motivos: el train aumentado es más difícil (la mitad tiene zonas tapadas), la métrica de train se promedia mientras los pesos cambian, y el dropout solo actúa en train. Con el modelo congelado de la mejor época, train queda por encima de val, como corresponde (tabla de la sección 3).

## 8. Resultados por comando (test)

| Comando | A_crudo_config2 | A_aumentado_config2 | B_crudo_config1 | **B_aumentado_config1** |
|---|---:|---:|---:|---:|
| yes | 0.935 | 0.940 | 0.988 | 0.989 |
| no | 0.811 | 0.810 | 0.957 | 0.968 |
| up | 0.846 | 0.874 | 0.950 | 0.967 |
| down | 0.822 | 0.860 | 0.959 | 0.963 |
| left | 0.881 | 0.901 | 0.976 | 0.981 |
| right | 0.923 | 0.945 | 0.986 | 0.987 |
| on | 0.865 | 0.900 | 0.954 | 0.966 |
| off | 0.870 | 0.883 | 0.948 | 0.963 |
| stop | 0.945 | 0.945 | 0.985 | 0.988 |
| go | 0.795 | 0.807 | 0.955 | 0.970 |

- **Los comandos fáciles son los mismos para todos:** *yes*, *stop*, *right* y *left*. Tienen consonantes fuertes (/s/, /t/, /r/, /l/) o dos sílabas claras, y su patrón en el espectrograma no se parece al de los otros.
- **Los difíciles también se repiten:** *go*, *no*, *down*, *off*, *on* y *up*, los grupos fonéticos de la sección 5.
- **El modelo final mantiene todos los comandos entre 0.963 y 0.989.** Sus confusiones más frecuentes son *down → no* (8), *no → go*, *up → off*, *on → off* y *off → on* (6 cada una). Son los mismos pares que confunde A, pero entre 3 y 12 veces menos.

## 9. Conclusiones

1. **El modelo final es `B_aumentado_config1`**, con F1 macro de 0.971 en validación y 0.974 en test, 50 410 parámetros, 245 KB y 4.6 ms por predicción.
2. **B supera a A en las 12 corridas por la arquitectura.** El promedio global le da invariancia a la posición de la palabra, tiene BatchNorm y dropout, y sus filtros pequeños se reutilizan en todo el espectrograma. En cambio, el 95.6 % de los parámetros de A está en una capa densa que depende de la posición y memoriza el entrenamiento.
3. **A sobreajusta en todas sus configuraciones.** SGD con weight decay (config. 2) es lo que mejor lo controla.
4. **La config. 3 no llega** porque su tasa de aprendizaje baja, que además decae con el coseno, avanza muy poco después de la época ~20, pierde la regularización implícita de un paso grande y tiene un weight decay debilitado por Adam. Más épocas no compensan esto.
5. **La aumentación con SpecAugment mejora todas las combinaciones**, en validación y en test, y ayuda más al modelo que más sobreajusta.
6. **Los errores restantes se concentran en palabras de una sílaba con la misma vocal**: *go*/*no*/*down* y *off*/*on*/*up*.

## 10. Limitaciones

- **Una sola semilla.** Las diferencias entre configuraciones del Modelo B (0.3–0.7 puntos, unos 12–26 audios de validación) pueden estar dentro de la variación entre semillas. La diferencia entre A y B (7–9 puntos) no.
- **Velocidad medida en PC.** El tiempo de predicción se midió en una CPU de escritorio. En el celular, con ONNX Runtime, será distinto, y se mide en las Fases 5 y 6.
- **Dónde están las figuras.** Las matrices de confusión de test, las gráficas train/val/test y el F1 por comando de los 4 candidatos están en `resultados/fase4/` y en W&B, en la corrida de cada candidato (`test/confusion_matrix`).

# Análisis de la corrida de los 12 modelos

## 1. Objetivo

La Fase 3 compara dos arquitecturas, Modelo A y Modelo B, usando datos crudos y datos aumentados. Cada combinación se entrena con tres configuraciones de hiperparámetros y se selecciona el mejor modelo usando exclusivamente el F1 macro sobre el conjunto de validación.

## 2. Configuraciones ejecutadas

| Parámetro | Valor |
|---|---:|
| Configuración 1 | Adam, learning rate 0.001, batch 64, weight decay 0, 35 épocas |
| Configuración 2 | SGD + Nesterov, learning rate 0.01, batch 128, weight decay 0.0001, 45 épocas |
| Configuración 3 | Adam, learning rate 0.0003, batch 64, weight decay 0.0001, 50 épocas |
| Dropout Modelo A | 0.00 |
| Dropout Modelo B | 0.20 (config. 1), 0.30 (config. 2), 0.10 (config. 3) |
| Semilla | 42 |
| Criterio de selección | Mejor F1 macro de validación |

## 3. Resultados

| Run | Modelo | Dataset | Mejor época | Mejor F1 validación | Accuracy validación final | Loss validación final |
|---|---|---|---:|---:|---:|---:|
| `A_crudo_config1` | A | Crudo | 29 | 0.8546 | 0.8534 | 0.8136 |
| `A_aumentado_config1` | A | Aumentado | 6 | 0.8667 | 0.8563 | 0.9572 |
| `A_crudo_config2` | A | Crudo | 42 | 0.8900 | 0.8882 | 0.4539 |
| `A_aumentado_config2` | A | Aumentado | 21 | 0.9025 | 0.9020 | 0.4383 |
| `A_crudo_config3` | A | Crudo | 38 | 0.8631 | 0.8601 | 0.6451 |
| `A_aumentado_config3` | A | Aumentado | 47 | 0.8691 | 0.8682 | 0.6888 |
| `B_crudo_config1` | B | Crudo | 36 | 0.9607 | 0.9584 | 0.1394 |
| `B_aumentado_config1` | B | Aumentado | 32 | 0.9707 | 0.9706 | 0.1304 |
| `B_crudo_config2` | B | Crudo | 35 | 0.9637 | 0.9635 | 0.1712 |
| `B_aumentado_config2` | B | Aumentado | 33 | 0.9674 | 0.9657 | 0.1213 |
| `B_crudo_config3` | B | Crudo | 22 | 0.9550 | 0.9506 | 0.1747 |
| `B_aumentado_config3` | B | Aumentado | 21 | 0.9603 | 0.9517 | 0.1701 |

### Métricas finales adicionales

| Run | Accuracy entrenamiento final | Loss entrenamiento final | F1 macro mostrado al final |
|---|---:|---:|---:|
| `A_crudo_config1` | 1.0000 | 0.0005 | 0.8533 |
| `A_aumentado_config1` | 1.0000 | 0.0002 | 0.8566 |
| `A_crudo_config2` | 0.9999 | 0.0095 | 0.8888 |
| `A_aumentado_config2` | 1.0000 | 0.0051 | 0.9025 |
| `A_crudo_config3` | 1.0000 | 0.0036 | 0.8608 |
| `A_aumentado_config3` | 1.0000 | 0.0021 | 0.8688 |
| `B_crudo_config1` | 0.9946 | 0.0280 | 0.9585 |
| `B_aumentado_config1` | 0.9953 | 0.0179 | 0.9706 |
| `B_crudo_config2` | 0.9985 | 0.0085 | 0.9637 |
| `B_aumentado_config2` | 0.9871 | 0.0476 | 0.9658 |
| `B_crudo_config3` | 0.9981 | 0.0139 | 0.9506 |
| `B_aumentado_config3` | 0.9961 | 0.0209 | 0.9518 |

## 4. Análisis del efecto de la aumentación

La aumentación mejoró el F1 macro de validación en las seis comparaciones crudo/aumentado:

| Modelo | Configuración | F1 crudo | F1 aumentado | Diferencia |
|---|---:|---:|---:|---:|
| A | 1 | 0.8546 | 0.8667 | +0.0121 |
| A | 2 | 0.8900 | 0.9025 | +0.0125 |
| A | 3 | 0.8631 | 0.8691 | +0.0060 |
| B | 3 | 0.9550 | 0.9603 | +0.0052 |

En el Modelo B, la aumentación también mejoró las tres configuraciones:

| Modelo | Configuración | F1 crudo | F1 aumentado | Diferencia |
|---|---:|---:|---:|---:|
| B | 1 | 0.9607 | 0.9707 | +0.0100 |
| B | 2 | 0.9637 | 0.9674 | +0.0037 |
| B | 3 | 0.9550 | 0.9603 | +0.0052 |

La mejora fue consistente y más grande en el Modelo A para las configuraciones 1 y 2. Esto indica que las máscaras de SpecAugment ayudaron a la generalización, aunque el efecto exacto debe interpretarse junto con las curvas de cada entrenamiento.

En el Modelo A, la configuración 2 fue superior a las configuraciones 1 y 3 tanto con datos crudos como aumentados:

- Crudo: `0.8900` frente a `0.8546` (config. 1) y `0.8631` (config. 3).
- Aumentado: `0.9025` frente a `0.8667` (config. 1) y `0.8691` (config. 3).

En el Modelo B, la configuración 2 fue la mejor con datos crudos, mientras que la configuración 1 fue la mejor con datos aumentados:

- Crudo: `B_crudo_config2` obtuvo `0.9637`, frente a `0.9607` de la configuración 1 y `0.9550` de la configuración 3.
- Aumentado: `B_aumentado_config1` obtuvo `0.9707`, frente a `0.9674` de la configuración 2 y `0.9603` de la configuración 3.

El Modelo B obtuvo un F1 superior al Modelo A en las dos condiciones y con las mejores configuraciones de cada modelo:

- Con datos crudos: `0.9637` (`B_crudo_config2`) frente a `0.8900` (`A_crudo_config2`).
- Con datos aumentados: `0.9707` (`B_aumentado_config1`) frente a `0.9025` (`A_aumentado_config2`).

El mejor modelo global es `B_aumentado_config1`, con F1 macro de validación `0.9707` en la época 32.

## 5. Análisis de las curvas

### Modelo A

Las corridas del Modelo A muestran señales de overfitting:

- La accuracy de entrenamiento se acerca a 1.0.
- La loss de entrenamiento continúa disminuyendo hasta valores cercanos a cero.
- La loss de validación baja al principio y luego aumenta.
- La accuracy y el F1 de validación se estabilizan antes de terminar las 50 épocas.

### Modelo B

Las corridas del Modelo B muestran un comportamiento más estable que las del Modelo A:

- La accuracy de validación se mantiene entre aproximadamente 0.95 y 0.97.
- El F1 macro alcanza valores altos y se estabiliza antes del final del entrenamiento.
- Las pérdidas de entrenamiento y validación disminuyen de forma consistente.
- Aunque existe una brecha entre entrenamiento y validación, es menor que en el Modelo A, especialmente en las corridas aumentadas de las configuraciones 1 y 2.

No se observa un overfitting muy marcado en el Modelo B. La configuración 3 muestra una brecha algo mayor que las configuraciones 1 y 2, pero su resultado de validación es inferior.

## 6. Selección definitiva por combinación

Esta selección puede hacerse usando el mayor `val/f1_macro` de cada combinación.

| Combinación | Mejor run disponible | F1 validación |
|---|---|---:|
| Modelo A + crudo | `A_crudo_config2` | 0.8900 |
| Modelo A + aumentado | `A_aumentado_config2` | 0.9025 |
| Modelo B + crudo | `B_crudo_config2` | 0.9637 |
| Modelo B + aumentado | `B_aumentado_config1` | 0.9707 |

Estos cuatro modelos son los candidatos seleccionados para la Fase 4. La evaluación con el conjunto de prueba debe hacerse una sola vez por candidato y después de congelar esta selección.

## 7. Resumen completo y trazabilidad

La tabla completa de las 12 corridas queda así:

| Modelo | Dataset | Configuración | Mejor F1 validación | Mejor época | Responsable
|---|---|---:|---:|---:|---|
| A | Crudo | 1 | 0.8546 | 29 | Jenny | 
| A | Aumentado | 1 | 0.8667 | 6 | Jenny | 
| A | Crudo | 2 | 0.8900 | 42 | Jenny | 
| A | Aumentado | 2 | 0.9025 | 21 | Jenny |
| B | Crudo | 1 | 0.9607 | 36 | Kendall | 
| B | Aumentado | 1 | 0.9707 | 32 | Kendall | 
| B | Crudo | 2 | 0.9637 | 35 | Kendall | 
| B | Aumentado | 2 | 0.9674 | 33 | Kendall | 
| A | Crudo | 3 | 0.8631 | 38 | Isaac | 
| A | Aumentado | 3 | 0.8691 | 47 | Isaac | 
| B | Crudo | 3 | 0.9550 | 22 | Isaac |
| B | Aumentado | 3 | 0.9603 | 21 | Isaac | 

## 8. Conclusiones

1. La aumentación mejoró el F1 macro de validación en las seis comparaciones crudo/aumentado.
2. La configuración 2 fue la mejor para el Modelo A, tanto con datos crudos como aumentados.
3. En el Modelo B, la configuración 2 fue mejor con datos crudos y la configuración 1 fue mejor con datos aumentados.
4. El Modelo A presentó overfitting más evidente, el Modelo B mostró curvas más estables.
# Diagramas de arquitecturas - Fase 2

## Modelo A - LeNet-5 adaptado

```mermaid
flowchart TD
    A[Entrada 1x1x40x101] --> B[Conv2d 1->6, kernel 5x5]
    B --> C[Tanh]
    C --> D[AvgPool2d 2x2, stride 2\n6x18x48]
    D --> E[Conv2d 6->16, kernel 5x5]
    E --> F[Tanh]
    F --> G[AvgPool2d 2x2, stride 2\n16x7x22]
    G --> H[Conv2d 16->120, kernel 7x22]
    H --> I[Tanh\n120x1x1]
    I --> J[Flatten 120]
    J --> K[Linear 120->84]
    K --> L[Tanh]
    L --> M[Linear 84->10]
    M --> N[Logits 1x10]
```

Resumen:

- Parametros: `309,386`.
- MACs aproximadas: `2,308,800`.
- Estandarizacion: `(x - (-6.5107)) / 4.8010` dentro del modelo.

## Modelo B - MobileNetV2 reducido

```mermaid
flowchart TD
    A[Entrada 1x1x40x101] --> B[Stem ConvBNReLU6\n1->16, kernel 3, stride 2]
    B --> C[InvertedResidual t=1\n16->16, 1 bloque]
    C --> D[InvertedResidual t=4\n16->24, 2 bloques, stride 2]
    D --> E[InvertedResidual t=4\n24->32, 2 bloques, stride 2]
    E --> F[InvertedResidual t=4\n32->64, 1 bloque]
    F --> G[Head ConvBNReLU6\n64->128, kernel 1]
    G --> H[AdaptiveAvgPool2d 1x1]
    H --> I[Flatten 128]
    I --> J[Dropout 0.2]
    J --> K[Linear 128->10]
    K --> L[Logits 1x10]
```

Resumen:
- Parametros: `50,410`.
- MACs aproximadas: `6,440,800`.
- Estandarizacion: `(x - (-6.5107)) / 4.8010` dentro del modelo.

## Validacion ONNX

- Modelo A: entrada `[1, 1, 40, 101]`, salida `[1, 10]`, diferencia maxima `1.15e-7`.
- Modelo B: entrada `[1, 1, 40, 101]`, salida `[1, 10]`, diferencia maxima `1.09e-11`.
- Opset: `17`; ejecucion validada con ONNX Runtime en CPU.

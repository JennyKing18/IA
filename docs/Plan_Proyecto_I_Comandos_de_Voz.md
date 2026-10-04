# Plan del Proyecto I: Reconocimiento de comandos de voz

**Qué se construye:** un sistema que reconoce 10 comandos de voz (yes, no, up, down, left, right, on, off, stop, go). Se entrena con dos modelos de PyTorch, se exporta al formato ONNX y se usa dentro de una aplicación móvil de casa inteligente que funciona con voz y sin internet.

**Personas del equipo:**
- Persona 1: ____________________
- Persona 2: ____________________
- Persona 3: ____________________

---

# Reglas para que las tres personas participen en TODAS las fases

1. **Cada fase se trabaja entre las tres personas.** En cada fase las tres tienen una tarea sobre el mismo resultado: una lo construye, otra lo revisa y otra lo prueba o lo documenta.
2. **Los papeles rotan en cada fase.** La persona que lidera una fase no lidera la siguiente, así las tres construyen de todo.
3. **Cada fase termina con una sesión de explicación** (de 20 a 30 minutos, por llamada o en persona). Quien lideró explica qué hizo y por qué, y las otras dos hacen preguntas. No se pasa a la siguiente fase hasta que las tres puedan explicar el resultado.
4. **Todas ejecutan todo.** Al terminar cada fase, cada persona corre el notebook completo de esa fase en su propia sesión de Google Colab y confirma que obtiene los mismos resultados.
5. **Ningún cambio de código se acepta sin que lo revise otra persona.** Quien revisa tiene que entender el cambio, no solo mirarlo.
6. **En el informe, cada persona escribe sobre fases que NO lideró.** Así queda comprobado que entiende lo que hicieron las otras.
7. **Reunión corta de seguimiento dos o tres veces por semana** (15 minutos): qué se hizo, qué sigue, qué está bloqueado.

---

# Fase 0: Arranque
**Lidera: Persona 3**

**Resultado esperado:** repositorio, herramientas y decisiones iniciales listos, con el contrato de entrada del modelo escrito.

- **Persona 3 (lidera):** hacer una prueba temprana con Expo. Crear una aplicación mínima que cargue un archivo `.onnx` de juguete y ejecute una predicción en un celular real. Se hace al inicio para descubrir problemas a tiempo. Tener en cuenta (verificar en la documentación actual):
  - La librería `onnxruntime-react-native` no funciona con Expo Go; hace falta una versión de desarrollo compilada (development build).
  - Compilar para iPhone desde Windows o Linux requiere el servicio de compilación en la nube de Expo y una cuenta de Apple. Si nadie tiene Mac, Android es la ruta más sencilla.
  - Los módulos de audio de Expo graban a archivo y no entregan el sonido en bruto en tiempo real; probablemente se necesite un módulo adicional de transmisión de audio.
- **Persona 1:** crear el repositorio compartido y el notebook con secciones ordenadas. Subir el dataset Speech Commands v0.02 a Google Drive para usarlo en Google Colab. Enviar al profesor estas preguntas: (1) si se puede usar Expo con React Native, (2) si se puede agregar una clase extra de "silencio o palabra desconocida", (3) si los cambios a LeNet-5 más allá de tamaño de entrada, canales e hiperparámetros necesitan aprobación, (4) si el espectrograma puede ir dentro del modelo exportado, (5) cuál es la fecha de entrega.
- **Persona 2:** crear el proyecto en Weights and Biases (la herramienta de seguimiento de entrenamientos) e invitar al equipo. Proponer la métrica de selección del mejor modelo, por ejemplo puntaje F1 promedio en validación, con tamaño y velocidad como criterio de desempate.
- **Todas:** definir juntas el **contrato de entrada del modelo**: frecuencia de muestreo (por ejemplo 16 000 Hz), duración del clip (1 segundo), tipo de espectrograma, tamaño de ventana, número de bandas y forma del tensor. Queda escrito, porque la aplicación debe repetirlo igual. Cada una también instala Expo y corre la aplicación mínima en su propio celular o emulador.

---

# Fase 1: Datos y preprocesamiento
**Lidera: Persona 1**

**Resultado esperado:** dos conjuntos de datos listos (crudo y aumentado), con tensores de referencia para probar la aplicación.

- **Persona 1 (lidera):** filtrar el dataset a los 10 comandos. Dividir en entrenamiento, validación y prueba con los archivos oficiales `validation_list.txt` y `testing_list.txt` (así un mismo hablante no aparece en dos conjuntos). Revisar el balance de clases. Generar el dataset crudo (audio a espectrograma) con el contrato de entrada.
- **Persona 2:** implementar la aumentación de datos para audio, como SpecAugment (enmascarar franjas de tiempo y de frecuencia), y opcionalmente ruido de fondo o desplazamiento en el tiempo. Se aplica solo al conjunto de entrenamiento. Escribir la justificación con literatura (artículo de SpecAugment de Park y colaboradores, 2019, y otras fuentes). Generar el dataset aumentado.
- **Persona 3:** escribir la especificación del preprocesamiento paso a paso, para poder reproducirla en TypeScript. Guardar tensores de referencia: uno o más audios `.wav` con el espectrograma exacto que produce Python.
- **Todas:** cada una revisa el trabajo de otra (Persona 1 revisa la aumentación, Persona 2 revisa la especificación, Persona 3 revisa la división de datos) y guarda figuras de ejemplo para el informe. Sesión de explicación al final.

---

# Fase 2: Arquitecturas de los modelos
**Lidera: Persona 2**

**Resultado esperado:** Modelo A y Modelo B implementados con `torch.nn`, con diagramas, justificados y revisados.

- **Persona 2 (lidera):** diseñar el Modelo B, una arquitectura ligera pensada para celular (por ejemplo una versión reducida de MobileNet o ResNet). Justificar profundidad, anchos de bloque, tipo de bloques y activaciones, y cómo afectan la generalización y el costo de cálculo. Implementarlo con función de pérdida, optimizador y rutinas de entrenamiento, validación y prueba.
- **Persona 1:** implementar el Modelo A, una variante de LeNet-5 adaptada a audio. Justificar cada cambio respecto a la original.
- **Persona 3:** hacer los diagramas de ambos modelos, contar los parámetros y las operaciones de cálculo, y revisar que todas las capas se puedan exportar a ONNX Runtime Mobile.
- **Todas:** cada una explica en voz alta una capa de cada modelo a las otras. Cada persona revisa el código del modelo que no escribió. Sesión de explicación al final.

---

# Fase 3: Entrenamiento de los 12 modelos
**Lidera: Persona 3**

**Resultado esperado:** 12 entrenamientos registrados en Weights and Biases (4 combinaciones, 3 configuraciones cada una), con el mejor de cada combinación elegido usando solo validación.

Las 4 combinaciones son: Modelo A con datos crudos, Modelo A con datos aumentados, Modelo B con datos crudos y Modelo B con datos aumentados.

- **Persona 3 (lidera):** armar la tabla compartida de hiperparámetros (tasa de aprendizaje, tamaño de lote, dropout, optimizador, épocas) para que las tres configuraciones sean distintas. Entrenar las 4 combinaciones con la configuración 3. Hacer el análisis cruzado: qué efecto tuvo la aumentación y qué hiperparámetros marcaron más diferencia.
- **Persona 1:** entrenar las 4 combinaciones con la configuración 1. Analizar curvas de entrenamiento contra validación del Modelo A y explicar si hay sobreajuste (overfitting) o subajuste (underfitting).
- **Persona 2:** entrenar las 4 combinaciones con la configuración 2. Hacer el mismo análisis para el Modelo B.
- **Todas:** juntas eligen el mejor de cada combinación mirando Weights and Biases. Cada persona explica las curvas de un modelo que no entrenó. Sesión de explicación al final.

---

# Fase 4: Evaluación y selección del modelo final
**Lidera: Persona 1**

**Resultado esperado:** resultados en el conjunto de prueba, comparación completa y modelo final elegido con la métrica acordada.

- **Persona 1 (lidera):** hacer la comparación de los 12 modelos en Weights and Biases y armar la tabla de resultados. Elegir el modelo final aplicando la métrica acordada.
- **Persona 2:** evaluar en el conjunto de prueba los mejores Modelos A (una sola vez cada uno). Generar matrices de confusión y puntaje F1 por comando.
- **Persona 3:** evaluar en el conjunto de prueba los mejores Modelos B (una sola vez cada uno), con matrices de confusión y puntaje F1 por comando. Medir tamaño del archivo y tiempo de predicción de los candidatos.
- **Todas:** juntas interpretan los resultados: qué comandos se confunden más y por qué. Sesión de explicación al final.

---

# Fase 5: Exportación a ONNX
**Lidera: Persona 2**

**Resultado esperado:** archivo `.onnx` del modelo final, validado contra PyTorch, con el contrato de entrada documentado.

- **Persona 2 (lidera):** fijar los pesos entrenados, poner el modelo en modo de inferencia, exportar el archivo `.onnx` y anotar la versión de opset y la configuración usada.
- **Persona 1:** validar la exportación de forma independiente: cargar el `.onnx` en Python con ONNX Runtime y comparar sus predicciones contra las del modelo original con ejemplos de prueba. Deben coincidir numéricamente.
- **Persona 3:** documentar el contrato de entrada final (formato del audio, frecuencia de muestreo, preprocesamiento, forma exacta del tensor) y verificar que nombres, formas y tipos de entradas y salidas coinciden con el modelo original.
- **Todas:** cada una repite la validación en su propia computadora. Sesión de explicación al final.

---

# Fase 6: Aplicación móvil con Expo
**Lidera: Persona 3**

**Resultado esperado:** aplicación funcional con cuatro pantallas que se controlan por voz, con el modelo corriendo de forma local en el celular.

Las cuatro pantallas y sus comandos:
- Panel principal: up, down, left, right mueven el foco; yes abre la tarjeta.
- Dispositivo: on y off controlan el aparato; yes confirma; no regresa.
- Rutinas: up y down mueven; go inicia; stop detiene.
- Confirmación: yes ejecuta la acción; no la cancela.

- **Persona 3 (lidera):** capturar audio del micrófono en ventanas de 1 segundo, integrar ONNX Runtime Mobile (cargar el `.onnx` como recurso de la aplicación y leer la predicción) y configurar la versión de desarrollo compilada.
- **Persona 1:** portar el espectrograma a TypeScript (transformada de Fourier, filtros mel y normalización) y probarlo hasta que coincida con los tensores de referencia de la Fase 1. Construir las pantallas Panel principal y Dispositivo.
- **Persona 2:** construir las pantallas Rutinas y Confirmación, y la lógica que conecta cada predicción con una acción. Evitar disparos falsos: ignorar predicciones con poca confianza y no repetir el mismo comando dos veces seguidas por error.
- **Todas:** trabajan juntas en una sesión de integración (compartiendo pantalla) y rotan quién escribe el código. Cada persona corre la aplicación en su propio celular y prueba los cuatro flujos hablando. Sesión de explicación al final.

---

# Fase 7: Informe y entrega
**Lideran las tres, con escritura cruzada**

**Resultado esperado:** informe en LaTeX con la plantilla de IEEE, notebook ordenado, código de la aplicación y archivo comprimido final.

Cada persona escribe las secciones de fases que **no** lideró:
- **Persona 1 escribe:** exportación a ONNX (Fase 5), entrenamiento (Fase 3) y arquitectura del Modelo B (Fase 2).
- **Persona 2 escribe:** datos y preprocesamiento (Fase 1), evaluación y selección (Fase 4) y aplicación móvil (Fase 6).
- **Persona 3 escribe:** aumentación y su justificación con literatura (Fase 1), arquitectura del Modelo A (Fase 2) y la introducción y las conclusiones.

Luego:
- **Revisión cruzada:** quien lideró cada fase revisa que lo escrito sobre ella sea correcto.
- **Notebook:** cada persona deja limpia su parte, y las tres verifican que corre completo de principio a fin y que coincide con el informe.
- **Armado final:** una persona (la que se turnen) une todo en Overleaf; las otras dos revisan el PDF antes de entregar.
- **Archivo comprimido:** código fuente en LaTeX, PDF del informe, notebook y código de la aplicación.

---

# Cosas que tener en cuenta
- **El audio debe tratarse igual en entrenamiento y en la aplicación.** Si el espectrograma en el celular no coincide con el del entrenamiento, el modelo falla aunque tenga buena exactitud en el notebook.
- **El conjunto de prueba no se toca para ajustar nada.** Se usa una sola vez por modelo.
- **La aumentación solo va en entrenamiento**, nunca en validación ni en prueba.
- **Las capas se construyen con `torch.nn`.** No se usan modelos preentrenados ni definiciones de alto nivel.
- **Guardar los puntos de control del entrenamiento en Google Drive** para no perder corridas si se cae Google Colab.
- **Confirmar la fecha de entrega con el profesor** y planificar hacia atrás: al menos una semana para la aplicación y otra para el informe.

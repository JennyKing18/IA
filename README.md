# Proyecto I: Reconocimiento de comandos de voz

Curso de Inteligencia Artificial, Instituto Tecnológico de Costa Rica.

Reconoce 10 comandos de voz (yes, no, up, down, left, right, on, off, stop, go) entrenados con Speech Commands v0.02. Se compararon dos modelos de PyTorch (A: LeNet-5, B: MobileNetV2 reducido) en W&B; el ganador (B, F1 macro 0.974 en test) se exportó a ONNX y corre **dentro del celular** con ONNX Runtime Mobile, en la app *Smart Home Voice Navigator* (Expo / React Native): con la voz se navega entre pantallas y se encienden o apagan dispositivos simulados, sin internet.

| Carpeta | Contenido |
|---|---|
| `notebooks/proyecto.ipynb` | Todo el modelo: datos, aumentación, entrenamiento, evaluación y exportación a ONNX |
| `models/` | Modelo final (`modelo_final.onnx`), validación y referencias para la app |
| `docs/` | Plan, contrato de entrada (receta exacta del espectrograma) y análisis de resultados |
| `app/` | App móvil: pantallas, espectrograma en TypeScript (`src/audio/`) y el modelo en `assets/model/` |

## Requisitos (Windows)

- [Git](https://git-scm.com/) y [Node.js](https://nodejs.org/) 20 o superior
- [Android Studio](https://developer.android.com/studio) con un emulador (por ejemplo Pixel 7, API 35, x86_64)
- JDK 17 o 21 (por ejemplo [Temurin 21](https://adoptium.net/)). **No** el JDK 25 que trae Android Studio: falla al compilar.

## 1. Descargar el proyecto

```powershell
git clone https://github.com/JennyKing18/IA.git
cd IA\app
npm install
```

`npm install` también aplica `app/patches/` (dos arreglos a `onnxruntime-react-native` para Gradle 9 y Expo).

## 2. Configurar Android y Java (en cada terminal nueva de PowerShell)

```powershell
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-21.0.12.101-hotspot"   # ajustar a la carpeta de su JDK
```

## 3. Generar el APK

```powershell
npx expo prebuild --platform android
cd android
.\gradlew.bat assembleRelease
```

El APK queda en `app\android\app\build\outputs\apk\release\app-release.apk`. Funciona sin computadora ni internet: copiarlo al celular, abrirlo y permitir "instalar apps desconocidas". La primera compilación tarda 15–30 min (descarga el NDK y dependencias).

## Probar en el emulador (desarrollo)

Con el emulador abierto, desde `app\`:

```powershell
npx expo run:android
```

Instala la app y recarga al guardar cambios. Después, para volver a abrirla basta `npx expo start --dev-client` (tecla `a`). Para usar el micrófono: en el emulador, **⋯ → Microphone → "Virtual microphone uses host audio input"**; luego tocar el círculo verde y decir un comando en inglés.

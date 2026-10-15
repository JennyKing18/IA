// Espectrograma log-mel en TypeScript, idéntico al del notebook (sección 1.4).
// Receta completa en docs/contrato_entrada.md, sección "Receta exacta para la app".
// Entrada: audio mono a 16 kHz. Salida: Float32Array de 40 × 101 en orden [banda][tramo],
// lista para el tensor [1, 1, 40, 101] del modelo. La normalización (x − MEDIA) / DESV va
// dentro del .onnx, así que aquí NO se aplica.

export const SAMPLE_RATE = 16000;
export const NUM_SAMPLES = 16000; // 1 segundo
export const N_FFT = 480; // ventana de 30 ms
export const HOP_LENGTH = 160; // salto de 10 ms
export const N_MELS = 40;
export const N_FRAMES = 1 + NUM_SAMPLES / HOP_LENGTH; // 101
export const F_MIN = 20;
export const F_MAX = 8000;
export const LOG_EPS = 1e-6;

const N_FREQS = N_FFT / 2 + 1; // 241 coeficientes de la FFT real
const PAD = N_FFT / 2; // 240 muestras de relleno reflect a cada lado

// Ventana de Hann periódica (divide entre N_FFT, no entre N_FFT − 1), como torch.hann_window
const WINDOW = new Float64Array(N_FFT);
for (let n = 0; n < N_FFT; n++) WINDOW[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);

// Tabla de giros e^(−2πi·j/N_FFT) para la FFT
const COS = new Float64Array(N_FFT);
const SIN = new Float64Array(N_FFT);
for (let j = 0; j < N_FFT; j++) {
  COS[j] = Math.cos((2 * Math.PI * j) / N_FFT);
  SIN[j] = Math.sin((2 * Math.PI * j) / N_FFT);
}

// Banco de 40 filtros mel triangulares (escala HTK, sin normalizar), guardado como [banda][coeficiente]
const hzToMel = (f: number) => 2595 * Math.log10(1 + f / 700);
const melToHz = (m: number) => 700 * (10 ** (m / 2595) - 1);
const MEL_FB = new Float64Array(N_MELS * N_FREQS);
{
  const melMin = hzToMel(F_MIN);
  const melMax = hzToMel(F_MAX);
  const fPts = Array.from({ length: N_MELS + 2 }, (_, i) => melToHz(melMin + ((melMax - melMin) * i) / (N_MELS + 1)));
  for (let m = 0; m < N_MELS; m++) {
    for (let k = 0; k < N_FREQS; k++) {
      const fk = (k * (SAMPLE_RATE / 2)) / (N_FREQS - 1);
      const subida = (fk - fPts[m]) / (fPts[m + 1] - fPts[m]);
      const bajada = (fPts[m + 2] - fk) / (fPts[m + 2] - fPts[m + 1]);
      MEL_FB[m * N_FREQS + k] = Math.max(0, Math.min(subida, bajada));
    }
  }
}

// FFT compleja de base mixta (Cooley–Tukey recursiva). Sirve para cualquier largo que divida a N_FFT;
// 480 = 2⁵ · 3 · 5.
function fft(re: Float64Array, im: Float64Array): [Float64Array, Float64Array] {
  const n = re.length;
  if (n === 1) return [re, im];
  let p = 2;
  while (n % p !== 0) p++; // menor factor primo
  const m = n / p;
  const partes: [Float64Array, Float64Array][] = [];
  for (let r = 0; r < p; r++) {
    const sr = new Float64Array(m);
    const si = new Float64Array(m);
    for (let j = 0; j < m; j++) {
      sr[j] = re[j * p + r];
      si[j] = im[j * p + r];
    }
    partes.push(fft(sr, si));
  }
  const outR = new Float64Array(n);
  const outI = new Float64Array(n);
  const paso = N_FFT / n; // e^(−2πi·rk/n) = COS/SIN[(rk mod n) · N_FFT/n]
  for (let k = 0; k < n; k++) {
    const km = k % m;
    let sr = 0;
    let si = 0;
    for (let r = 0; r < p; r++) {
      const [yr, yi] = partes[r];
      const idx = ((r * k) % n) * paso;
      sr += yr[km] * COS[idx] + yi[km] * SIN[idx];
      si += yi[km] * COS[idx] - yr[km] * SIN[idx];
    }
    outR[k] = sr;
    outI[k] = si;
  }
  return [outR, outI];
}

/** Muestras int16 → float en [−1, 1] (int16 / 32768). */
export function pcm16ToFloat(pcm: Int16Array): Float32Array {
  const out = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) out[i] = pcm[i] / 32768;
  return out;
}

/** Primeras 16 000 muestras; si faltan, rellena con ceros al final. */
export function fixLength(wave: Float32Array): Float32Array {
  if (wave.length === NUM_SAMPLES) return wave;
  const out = new Float32Array(NUM_SAMPLES);
  out.set(wave.subarray(0, NUM_SAMPLES));
  return out;
}

/** Audio mono a 16 kHz → log-mel de 40 × 101 (Float32Array de 4 040 valores, índice banda · 101 + tramo). */
export function logMel(wave: Float32Array): Float32Array {
  const x = fixLength(wave);

  // Relleno reflect de 240 muestras a cada lado, sin repetir el borde: largo 16 480
  const padded = new Float64Array(NUM_SAMPLES + 2 * PAD);
  for (let i = 0; i < NUM_SAMPLES; i++) padded[PAD + i] = x[i];
  for (let i = 1; i <= PAD; i++) {
    padded[PAD - i] = x[i];
    padded[PAD + NUM_SAMPLES - 1 + i] = x[NUM_SAMPLES - 1 - i];
  }

  const out = new Float32Array(N_MELS * N_FRAMES);
  const potencia = new Float64Array(N_FREQS);
  const ceros = new Float64Array(N_FFT);
  for (let t = 0; t < N_FRAMES; t++) {
    // Tramo t ventaneado → espectro de potencia |X[k]|²
    const tramo = new Float64Array(N_FFT);
    for (let n = 0; n < N_FFT; n++) tramo[n] = padded[t * HOP_LENGTH + n] * WINDOW[n];
    const [re, im] = fft(tramo, ceros);
    for (let k = 0; k < N_FREQS; k++) potencia[k] = re[k] * re[k] + im[k] * im[k];

    // Filtros mel y logaritmo natural
    for (let m = 0; m < N_MELS; m++) {
      let suma = 0;
      const base = m * N_FREQS;
      for (let k = 0; k < N_FREQS; k++) suma += MEL_FB[base + k] * potencia[k];
      out[m * N_FRAMES + t] = Math.log(suma + LOG_EPS);
    }
  }
  return out;
}

/** Lee un .wav PCM de 16 bits, mono, 16 kHz y devuelve sus muestras en [−1, 1]. */
export function decodeWav16(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const texto = (o: number) => String.fromCharCode(bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]);
  if (texto(0) !== 'RIFF' || texto(8) !== 'WAVE') throw new Error('No es un archivo WAV');
  let o = 12;
  let formatoOk = false;
  while (o + 8 <= bytes.length) {
    const id = texto(o);
    const largo = view.getUint32(o + 4, true);
    if (id === 'fmt ') {
      const formato = view.getUint16(o + 8, true);
      const canales = view.getUint16(o + 10, true);
      const tasa = view.getUint32(o + 12, true);
      const bits = view.getUint16(o + 22, true);
      if (formato !== 1 || canales !== 1 || tasa !== SAMPLE_RATE || bits !== 16) {
        throw new Error(`WAV no soportado: formato ${formato}, ${canales} canales, ${tasa} Hz, ${bits} bits`);
      }
      formatoOk = true;
    } else if (id === 'data') {
      if (!formatoOk) throw new Error('WAV sin bloque fmt antes de data');
      const n = Math.floor(largo / 2);
      const pcm = new Int16Array(n);
      for (let i = 0; i < n; i++) pcm[i] = view.getInt16(o + 8 + 2 * i, true);
      return pcm16ToFloat(pcm);
    }
    o += 8 + largo + (largo % 2);
  }
  throw new Error('WAV sin bloque data');
}

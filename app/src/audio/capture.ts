// Preparación del audio capturado antes del log-mel: remuestreo a 16 kHz y elección del segundo con más voz.
import { NUM_SAMPLES, SAMPLE_RATE } from './logmel';

/** Remuestreo lineal a 16 kHz (solo si el micrófono no entregó 16 kHz). */
export function resample(x: Float32Array, fromRate: number): Float32Array {
  if (fromRate === SAMPLE_RATE) return x;
  const n = Math.floor((x.length * SAMPLE_RATE) / fromRate);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const pos = (i * fromRate) / SAMPLE_RATE;
    const j = Math.floor(pos);
    const frac = pos - j;
    out[i] = j + 1 < x.length ? x[j] * (1 - frac) + x[j + 1] * frac : x[x.length - 1];
  }
  return out;
}

/** El segundo (16 000 muestras) con más energía, para centrar la palabra como en el dataset. */
export function loudestSecond(x: Float32Array): { wave: Float32Array; rms: number } {
  if (x.length <= NUM_SAMPLES) {
    const wave = new Float32Array(NUM_SAMPLES);
    wave.set(x);
    let e = 0;
    for (let i = 0; i < x.length; i++) e += x[i] * x[i];
    return { wave, rms: Math.sqrt(e / NUM_SAMPLES) };
  }
  const cumsum = new Float64Array(x.length + 1);
  for (let i = 0; i < x.length; i++) cumsum[i + 1] = cumsum[i] + x[i] * x[i];
  let bestStart = 0;
  let bestEnergy = -1;
  for (let s = 0; s + NUM_SAMPLES <= x.length; s += 160) {
    const e = cumsum[s + NUM_SAMPLES] - cumsum[s];
    if (e > bestEnergy) {
      bestEnergy = e;
      bestStart = s;
    }
  }
  return { wave: x.slice(bestStart, bestStart + NUM_SAMPLES), rms: Math.sqrt(bestEnergy / NUM_SAMPLES) };
}

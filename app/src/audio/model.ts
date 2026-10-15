// Modelo final (B_aumentado_config1) con ONNX Runtime Mobile, ejecutado en el celular.
// Entrada: log-mel sin normalizar [1, 1, 40, 101]; salida: 10 logits en el orden de COMMANDS.
import { Asset } from 'expo-asset';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';

import { logMel, N_FRAMES, N_MELS } from './logmel';

export const COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'] as const;
export type Command = (typeof COMMANDS)[number];

export type Prediction = { command: Command; confidence: number; probabilities: Float32Array };

let sessionPromise: Promise<InferenceSession> | null = null;

async function loadSession(): Promise<InferenceSession> {
  // El .onnx viaja dentro de la app (assets/model) y se copia a un archivo local para abrirlo
  const asset = Asset.fromModule(require('../../assets/model/modelo_final.onnx'));
  await asset.downloadAsync();
  const uri = asset.localUri ?? asset.uri;
  try {
    return await InferenceSession.create(decodeURIComponent(uri.replace(/^file:\/\//, '')));
  } catch {
    // Alternativa: abrir el modelo desde sus bytes
    const bytes = new Uint8Array(await (await fetch(uri)).arrayBuffer());
    return InferenceSession.create(bytes);
  }
}

/** Carga el modelo una sola vez (las siguientes llamadas reutilizan la sesión). */
export function getSession(): Promise<InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = loadSession().catch((error) => {
      sessionPromise = null; // permite reintentar
      throw error;
    });
  }
  return sessionPromise;
}

/** Audio de 1 s a 16 kHz → comando más probable y su probabilidad (softmax). */
export async function predict(wave: Float32Array): Promise<Prediction> {
  const session = await getSession();
  const input = new Tensor('float32', logMel(wave), [1, 1, N_MELS, N_FRAMES]);
  const output = await session.run({ input });
  const logits = output.logits.data as Float32Array;

  const max = Math.max(...logits);
  const exp = logits.map((v) => Math.exp(v - max));
  const total = exp.reduce((a, b) => a + b, 0);
  const probabilities = exp.map((v) => v / total);
  let best = 0;
  for (let i = 1; i < probabilities.length; i++) if (probabilities[i] > probabilities[best]) best = i;
  return { command: COMMANDS[best], confidence: probabilities[best], probabilities };
}

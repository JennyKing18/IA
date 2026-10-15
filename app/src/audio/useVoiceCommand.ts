// "Presionar para hablar": graba un momento, elige el segundo con más voz, lo pasa por el modelo
// y, si la confianza alcanza, entrega el comando.
import { requestRecordingPermissionsAsync, setAudioModeAsync, useAudioStream, type AudioStreamBuffer } from 'expo-audio';
import { useCallback, useEffect, useRef, useState } from 'react';

import { loudestSecond, resample } from './capture';
import { SAMPLE_RATE } from './logmel';
import { getSession, predict, type Command, type Prediction } from './model';

const CAPTURE_MS = 1500; // margen para que la palabra quepa entera aunque se diga un poco tarde
const MIN_CONFIDENCE = 0.6; // por debajo, la predicción se ignora
const MIN_RMS = 0.002; // energía mínima del segundo elegido; por debajo se considera silencio

export type VoiceState = 'loading' | 'idle' | 'listening' | 'thinking' | 'error';

/** int16 (o float32) intercalado → float mono en [−1, 1] a su tasa original. */
function toMono(buffers: AudioStreamBuffer[]): Float32Array {
  const parts = buffers.map((b) => {
    const ints = new Int16Array(b.data);
    const out = new Float32Array(Math.floor(ints.length / b.channels));
    for (let i = 0; i < out.length; i++) out[i] = ints[i * b.channels] / 32768;
    return out;
  });
  const total = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const p of parts) {
    total.set(p, offset);
    offset += p.length;
  }
  return total;
}

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export function useVoiceCommand(onCommand: (command: Command) => void) {
  const buffers = useRef<AudioStreamBuffer[]>([]);
  const onBuffer = useCallback((buffer: AudioStreamBuffer) => { buffers.current.push(buffer); }, []);
  const { stream } = useAudioStream({ sampleRate: SAMPLE_RATE, channels: 1, encoding: 'int16', onBuffer });
  const [state, setState] = useState<VoiceState>('loading');
  const [message, setMessage] = useState('Loading voice model...');
  const [last, setLast] = useState<Prediction | null>(null);

  useEffect(() => {
    getSession()
      .then(() => { setState('idle'); setMessage('Tap the mic and say a command'); })
      .catch((error) => { setState('error'); setMessage(`Model error: ${String(error?.message ?? error)}`); });
  }, []);

  const listen = useCallback(async () => {
    if (state !== 'idle') return;
    const permission = await requestRecordingPermissionsAsync();
    if (!permission.granted) {
      setMessage('Microphone permission required');
      return;
    }
    try {
      await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
      buffers.current = [];
      setState('listening');
      setMessage('Listening...');
      await stream.start();
      await wait(CAPTURE_MS);
      stream.stop();

      setState('thinking');
      const captured = buffers.current;
      if (captured.length === 0) throw new Error('No audio captured');
      const rate = captured[0].sampleRate;
      const { wave, rms } = loudestSecond(resample(toMono(captured), rate));
      if (rms < MIN_RMS) {
        setMessage('No voice detected, try again');
        return;
      }
      const prediction = await predict(wave);
      setLast(prediction);
      const pct = Math.round(prediction.confidence * 100);
      if (prediction.confidence < MIN_CONFIDENCE) {
        setMessage(`Not sure (${prediction.command.toUpperCase()} ${pct}%), try again`);
        return;
      }
      setMessage(`Heard ${prediction.command.toUpperCase()} (${pct}%)`);
      onCommand(prediction.command);
    } catch (error: any) {
      setMessage(`Audio error: ${String(error?.message ?? error)}`);
    } finally {
      if (stream.isStreaming) stream.stop();
      setState((s) => (s === 'error' ? s : 'idle'));
    }
  }, [state, stream, onCommand]);

  return { state, message, last, listen };
}

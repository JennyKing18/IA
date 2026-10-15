// Prueba del espectrograma en TypeScript contra las referencias de Python (models/referencias_app.json).
// Uso (desde app/):  node scripts/test-logmel.mjs
// Los .wav se buscan en DATA_DIR (por defecto ~/Downloads/archive), igual que en el notebook.
import { readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { decodeWav16, logMel, N_FRAMES, N_MELS } from '../src/audio/logmel.ts';

const TOLERANCIA = 1e-3;
const refs = JSON.parse(readFileSync(new URL('../../models/referencias_app.json', import.meta.url), 'utf-8'));
const dataDir = process.env.DATA_DIR ?? join(homedir(), 'Downloads', 'archive');

let peor = 0;
for (const audio of refs.audios) {
  const wave = decodeWav16(readFileSync(join(dataDir, audio.archivo)));
  const inicio = performance.now();
  const lm = logMel(wave);
  const ms = performance.now() - inicio;
  let dif = 0;
  for (let m = 0; m < N_MELS; m++) {
    for (let t = 0; t < N_FRAMES; t++) dif = Math.max(dif, Math.abs(lm[m * N_FRAMES + t] - audio.logmel_40x101[m][t]));
  }
  peor = Math.max(peor, dif);
  console.log(`${audio.comando.padEnd(6)} ${audio.archivo.padEnd(28)} dif máx ${dif.toExponential(2)}  (${ms.toFixed(1)} ms)`);
}
console.log(peor < TOLERANCIA ? `OK: coincide con Python (peor diferencia ${peor.toExponential(2)} < ${TOLERANCIA})`
                              : `FALLA: diferencia ${peor.toExponential(2)} ≥ ${TOLERANCIA}`);
process.exit(peor < TOLERANCIA ? 0 : 1);

// Waveform decoding and drawing: the sample strip and the device's dot-matrix screen.

const peakCache = new Map();
let audioCtx = null;

export function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export async function loadPeaks(url, buckets = 800) {
  const key = `${url}#${buckets}`;
  if (peakCache.has(key)) return peakCache.get(key);
  const buf = await (await fetch(url)).arrayBuffer();
  audioCtx ??= new (window.AudioContext || window.webkitAudioContext)();
  const audio = await audioCtx.decodeAudioData(buf);
  const data = audio.getChannelData(0);
  const size = Math.max(1, Math.floor(data.length / buckets));
  const peaks = new Float32Array(buckets);
  let max = 1e-4;
  for (let b = 0; b < buckets; b++) {
    let sum = 0;
    const start = b * size;
    for (let i = start; i < start + size && i < data.length; i++) sum += data[i] * data[i];
    peaks[b] = Math.sqrt(sum / size);
    if (peaks[b] > max) max = peaks[b];
  }
  for (let b = 0; b < buckets; b++) peaks[b] = Math.min(1, peaks[b] / max);
  const out = { peaks, duration: audio.duration };
  peakCache.set(key, out);
  return out;
}

function fit(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: canvas.clientWidth, h: canvas.clientHeight };
}

function resample(peaks, n) {
  const out = new Float32Array(n);
  const step = peaks.length / n;
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (let j = Math.floor(i * step); j < Math.floor((i + 1) * step) && j < peaks.length; j++) m = Math.max(m, peaks[j]);
    out[i] = m;
  }
  return out;
}

// Thin bars for the voice sample strip.
export function drawStrip(canvas, peaks, played = 0) {
  const { ctx, w, h } = fit(canvas);
  ctx.clearRect(0, 0, w, h);
  if (!peaks) return;
  const bar = 2, gap = 2;
  const n = Math.floor(w / (bar + gap));
  const vals = resample(peaks, n);
  const on = token("--color-ink"), off = token("--color-field");
  for (let i = 0; i < n; i++) {
    const bh = Math.max(2, vals[i] * (h - 4));
    ctx.fillStyle = i / n < played ? on : off;
    ctx.fillRect(i * (bar + gap), (h - bh) / 2, bar, bh);
  }
}

// Live microphone level, newest value on the right.
export function drawLevels(canvas, levels) {
  const { ctx, w, h } = fit(canvas);
  ctx.clearRect(0, 0, w, h);
  const bar = 3, gap = 2;
  const n = Math.floor(w / (bar + gap));
  const start = Math.max(0, levels.length - n);
  ctx.fillStyle = token("--color-ink-2");
  for (let i = start; i < levels.length; i++) {
    const x = (i - start) * (bar + gap) + (n - (levels.length - start)) * (bar + gap);
    const bh = Math.max(2, Math.min(1, levels[i] * 3.2) * (h - 2));
    ctx.fillRect(x, (h - bh) / 2, bar, bh);
  }
}

// Deterministic stand-in for speech: words as bursts of syllables, quiet gaps between them, fading at both ends.
function speechShape(n) {
  const out = new Float32Array(n);
  const hash = (i) => { const x = Math.sin(i * 12.9898 + 4.1) * 43758.5453; return x - Math.floor(x); };
  for (let i = 0; i < n; i++) {
    const x = (i + 0.5) / n;
    const env = Math.pow(Math.sin(Math.PI * x), 0.45);
    const word = Math.abs(Math.sin(i * 0.105 + 0.6));
    const syll = Math.pow(Math.abs(Math.sin(i * 0.47 + 0.2)), 1.3);
    const gap = word < 0.16 ? 0.06 : 1;
    out[i] = env * gap * (0.12 + 0.88 * syll * (0.55 + 0.45 * hash(i)) * Math.pow(word, 0.35));
  }
  return out;
}

// The player screen: a dot-matrix display on deep navy. Empty, it shows a ghost of a voice; while audio is
// generated the same shape lights up left to right; afterwards it becomes the real waveform, played part in blue.
export class WaveLine {
  constructor(canvas) {
    this.canvas = canvas;
    this.mode = "idle";
    this.progress = 0;
    this.peaks = null;
    this.playhead = 0;
    this.padX = 0;
    this.ro = new ResizeObserver(() => this.render());
    this.ro.observe(canvas);
  }
  idle() { this.mode = "idle"; this.render(); }
  dark() { this.mode = "dark"; this.render(); }
  setProgress(p) { this.mode = "progress"; this.progress = Math.max(0, Math.min(1, p)); this.render(); }
  setWave(peaks) { this.mode = "wave"; this.peaks = peaks; this.playhead = 0; this.render(); }
  setPlayhead(f) { this.playhead = f; if (this.mode === "wave") this.render(); }
  columnAt(clientX) {
    const r = this.canvas.getBoundingClientRect();
    return Math.max(0, Math.min(1, (clientX - r.left - this.padX) / Math.max(1, r.width - this.padX * 2)));
  }

  render() {
    const { ctx, w, h } = fit(this.canvas);
    ctx.clearRect(0, 0, w, h);
    const pitch = w < 360 ? 8 : 9;
    const radius = pitch * 0.3;
    const cols = Math.max(1, Math.floor((w - pitch) / pitch) + 1);
    const rows = Math.max(1, Math.floor((h - pitch) / pitch) + 1);
    this.padX = (w - (cols - 1) * pitch) / 2;
    const padY = (h - (rows - 1) * pitch) / 2;
    const mid = (rows - 1) / 2;
    const half = mid;

    const off = token("--color-dot-off"), ghost = token("--color-dot-ghost");
    const dim = token("--color-accent-dim"), on = token("--color-accent"), lead = token("--color-focus");
    const dot = (c, r, color) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(this.padX + c * pitch, padY + r * pitch, radius, 0, Math.PI * 2);
      ctx.fill();
    };
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) dot(c, r, off);
    if (this.mode === "dark") return;

    const shape = this.mode === "wave" && this.peaks ? resample(this.peaks, cols) : speechShape(cols);
    const edge = this.progress * cols;
    for (let c = 0; c < cols; c++) {
      let color = ghost;
      if (this.mode === "progress") {
        if (c < edge) color = edge - c < 1.5 ? lead : on;
      } else if (this.mode === "wave") {
        color = c / cols < this.playhead ? on : dim;
      }
      const reach = Math.round(Math.max(0, Math.min(1, shape[c])) * half);
      for (let r = Math.round(mid - reach); r <= Math.round(mid + reach); r++) dot(c, r, color);
    }
  }
}

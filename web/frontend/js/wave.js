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

// The player: one thin blue line on deep navy. While audio is generated the line fills left to right;
// afterwards it becomes the waveform, played part in blue, the rest dimmed.
export class WaveLine {
  constructor(canvas) {
    this.canvas = canvas;
    this.mode = "idle";
    this.progress = 0;
    this.peaks = null;
    this.playhead = 0;
    this.pad = 2;
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
    return Math.max(0, Math.min(1, (clientX - r.left) / r.width));
  }

  render() {
    const { ctx, w, h } = fit(this.canvas);
    ctx.clearRect(0, 0, w, h);
    const mid = h / 2;
    const on = token("--color-accent"), dim = token("--color-accent-dim"), faint = token("--color-navy-2");
    ctx.lineCap = "round";
    const line = (x0, x1, color, width) => {
      ctx.strokeStyle = color; ctx.lineWidth = width;
      ctx.beginPath(); ctx.moveTo(x0, mid); ctx.lineTo(x1, mid); ctx.stroke();
    };
    if (this.mode === "dark") { line(this.pad, w - this.pad, faint, 1); return; }
    if (this.mode === "idle") { line(this.pad, w - this.pad, dim, 1); return; }
    if (this.mode === "progress") {
      const x = this.pad + (w - 2 * this.pad) * this.progress;
      line(this.pad, w - this.pad, faint, 2);
      if (this.progress > 0) line(this.pad, x, on, 2);
      ctx.fillStyle = on;
      ctx.beginPath(); ctx.arc(x, mid, 4, 0, Math.PI * 2); ctx.fill();
      return;
    }
    if (!this.peaks) return;
    const step = 4;
    const n = Math.floor((w - 2 * this.pad) / step);
    const vals = resample(this.peaks, n);
    ctx.lineWidth = 2;
    for (let i = 0; i < n; i++) {
      const x = this.pad + i * step + 1;
      const half = Math.max(1, vals[i] * (h / 2 - 3));
      ctx.strokeStyle = i / n < this.playhead ? on : dim;
      ctx.beginPath(); ctx.moveTo(x, mid - half); ctx.lineTo(x, mid + half); ctx.stroke();
    }
  }
}

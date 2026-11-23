// Microphone capture with MediaRecorder, a live level meter and a hard time limit.

import { drawLevels } from "./wave.js?v=4";

const TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];

export class Recorder {
  constructor({ canvas, maxSeconds = 30, onTick = () => {}, onStop = () => {} }) {
    this.canvas = canvas;
    this.maxSeconds = maxSeconds;
    this.onTick = onTick;
    this.onStop = onStop;
    this.recording = false;
  }

  static supported() {
    return !!(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
    const mimeType = TYPES.find((t) => MediaRecorder.isTypeSupported?.(t)) || "";
    this.mime = mimeType || "audio/webm";
    this.chunks = [];
    this.media = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
    this.media.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    this.media.onstop = () => this._finish();

    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    src.connect(this.analyser);
    this.levels = [];
    this.buf = new Float32Array(this.analyser.fftSize);

    this.media.start(250);
    this.recording = true;
    this.t0 = performance.now();
    this._loop();
  }

  _loop() {
    if (!this.recording) return;
    this.analyser.getFloatTimeDomainData(this.buf);
    let sum = 0;
    for (const v of this.buf) sum += v * v;
    this.levels.push(Math.sqrt(sum / this.buf.length));
    if (this.levels.length > 400) this.levels.shift();
    drawLevels(this.canvas, this.levels);
    const secs = (performance.now() - this.t0) / 1000;
    this.onTick(secs);
    if (secs >= this.maxSeconds) return this.stop();
    this.raf = requestAnimationFrame(() => this._loop());
  }

  stop() {
    if (!this.recording) return;
    this.recording = false;
    cancelAnimationFrame(this.raf);
    this.seconds = (performance.now() - this.t0) / 1000;
    this.media.state !== "inactive" && this.media.stop();
  }

  cancel() {
    this.onStop = () => {};
    this.stop();
  }

  _finish() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close();
    const ext = this.mime.includes("mp4") ? "m4a" : this.mime.includes("ogg") ? "ogg" : "webm";
    const blob = new Blob(this.chunks, { type: this.mime.split(";")[0] });
    const stamp = new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    const file = new File([blob], `Thu âm ${stamp}.${ext}`, { type: blob.type });
    this.onStop(file, this.seconds);
  }
}

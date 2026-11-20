// Thin wrappers around the backend API. Every call rejects with an Error whose message is human readable.

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(path, options);
  } catch {
    throw new Error("Không kết nối được máy chủ. Kiểm tra server còn chạy không.");
  }
  const type = res.headers.get("content-type") || "";
  const body = type.includes("application/json") ? await res.json().catch(() => ({})) : null;
  if (!res.ok) {
    const msg = body && (body.error || body.detail);
    throw new Error(typeof msg === "string" ? msg : `Máy chủ trả về lỗi ${res.status}.`);
  }
  return body;
}

const json = (method, data) => ({
  method,
  headers: { "content-type": "application/json" },
  body: JSON.stringify(data ?? {}),
});

export const api = {
  status: () => request("/api/status"),
  models: () => request("/api/models"),
  voices: () => request("/api/voices"),
  history: () => request("/api/history"),
  normalize: (text, normalize, split) => request("/api/normalize", json("POST", { text, normalize, split })),
  deleteVoice: (id) => request(`/api/voices/${id}`, { method: "DELETE" }),
  clearCache: () => request("/api/cache/clear", json("POST", { keep_voices: true })),
  createJob: (payload) => request("/api/jobs", json("POST", payload)),
  job: (id) => request(`/api/jobs/${id}`),
  cancelJob: (id) => request(`/api/jobs/${id}/cancel`, { method: "POST" }),
  uploadVoice(file, { name = "", source = "upload" } = {}) {
    const form = new FormData();
    form.append("file", file, file.name || "audio");
    form.append("name", name);
    form.append("source", source);
    return request("/api/voices", { method: "POST", body: form });
  },
  voiceAudioUrl: (id) => `/api/voices/${id}/audio`,

  // Server-sent events with a polling fallback (some tunnels buffer SSE).
  watchJob(id, onUpdate) {
    let closed = false;
    let es = null;
    let pollTimer = null;
    const terminal = (j) => ["done", "error", "cancelled"].includes(j.status);
    const finish = () => { closed = true; es?.close(); clearTimeout(pollTimer); };
    const poll = async () => {
      if (closed) return;
      try {
        const j = await api.job(id);
        onUpdate(j);
        if (terminal(j)) return finish();
      } catch (e) { /* transient; keep polling */ }
      pollTimer = setTimeout(poll, 400);
    };
    let gotEvent = false;
    try {
      es = new EventSource(`/api/jobs/${id}/events`);
      es.onmessage = (ev) => {
        gotEvent = true;
        const j = JSON.parse(ev.data);
        onUpdate(j);
        if (terminal(j)) finish();
      };
      es.onerror = () => { es.close(); if (!closed) poll(); };
      // if nothing arrives in 3 s the stream is probably buffered: switch to polling
      setTimeout(() => { if (!gotEvent && !closed) { es.close(); poll(); } }, 3000);
    } catch {
      poll();
    }
    return finish;
  },
};

import { api } from "./api.js?v=4";
import { WaveLine, drawStrip, loadPeaks } from "./wave.js?v=4";
import { Recorder } from "./recorder.js?v=4";
import { defaultsOf, renderParams, renderPresets, syncPresets } from "./params.js?v=4";

const $ = (id) => document.getElementById(id);
const nf = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 2 });
const nf1 = new Intl.NumberFormat("vi-VN", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

const SAMPLES = {
  numbers: "Nhiệt độ hôm nay là 30°C, độ ẩm 80%, giá vé 15k tại TP.HCM. Hạn nộp đồ án là 15/11/2026, mỗi nhóm 2-3 thành viên.",
  academic: "Sway Sampling là chiến lược lấy mẫu flow step mà F5-TTS (Chen et al., 2025) đề xuất cho giai đoạn inference. Trong quá trình huấn luyện, mô hình vẫn lấy flow step đều trên toàn bộ quỹ đạo từ noise đến dữ liệu. Khi suy luận, Sway Sampling dùng một hàm biến đổi đơn điệu để phân bố lại các bước, và một hệ số điều chỉnh quyết định các bước dồn về phía nào. Thực nghiệm cho thấy khi bật Sway Sampling, WER của F5-TTS trên LibriSpeech-PC giảm từ 2.84 xuống 2.41, còn SIM-o tăng từ 0.62 lên 0.66.",
  hello: "Xin chào thầy và các bạn. [nghỉ 1s] Đây là giọng nói được tạo bởi mô hình tổng hợp tiếng nói.",
};

// ---------- persistence (per-viewer convenience only) ----------
const saved = (() => {
  const KEY = "vong.v1";
  let data = {};
  try { data = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { data = {}; }
  return {
    get: (k, d) => (k in data ? data[k] : d),
    set(k, v) { data[k] = v; try { localStorage.setItem(KEY, JSON.stringify(data)); } catch { /* private mode */ } },
  };
})();

// ---------- state ----------
const state = {
  models: [],
  modelId: saved.get("model", "omnivoice"),
  params: saved.get("params", {}),
  voice: null,
  voices: [],
  status: null,
  job: null,
  stopWatch: null,
  result: null,
  history: [],
  refTexts: saved.get("refTexts", {}),
};

const model = () => state.models.find((m) => m.id === state.modelId) || state.models[0];
const engineState = () => state.status?.engines?.[state.modelId]?.state;
const values = () => ({ ...defaultsOf(model()), ...(state.params[state.modelId] || {}) });

const matrix = new WaveLine($("matrix"));
const sampleAudio = $("sampleAudio");
const resultAudio = $("resultAudio");

// ---------- helpers ----------
let toastTimer;
function toast(msg, tone = "error") {
  const t = $("toast");
  t.textContent = msg;
  t.dataset.tone = tone;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 6000);
}
function note(msg, tone = "") {
  const n = $("voiceNote");
  n.textContent = msg;
  n.dataset.tone = tone;
}
const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const secs = (s) => `${nf1.format(s)} s`;
const when = (t) => new Date(t * 1000).toLocaleString("vi-VN", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "2-digit" });
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------- server status ----------
async function refreshStatus() {
  try {
    state.status = await api.status();
  } catch {
    state.status = null;
  }
  renderStatus();
  updateGo();
  const loading = state.status && Object.values(state.status.engines).some((e) => e.state === "loading");
  clearTimeout(refreshStatus.t);
  refreshStatus.t = setTimeout(refreshStatus, loading ? 2500 : 15000);
}

function renderStatus() {
  const s = state.status;
  if (!s) return;
  $("charMax").textContent = s.limits.max_text;
  if ($("text").maxLength !== s.limits.max_text) { $("text").maxLength = s.limits.max_text; onTextInput(); }
  const c = s.cache;
  $("cVoices").textContent = c.voices;
  $("cPrompts").textContent = c.prompts;
  $("cSegments").textContent = c.segments;
  $("cResults").textContent = c.results;
  const hits = c.hits.prompt_memory + c.hits.prompt_disk + c.hits.segment + c.hits.result;
  $("cacheFoot").textContent = `${nf.format(c.bytes / 1048576)} MB trên đĩa. ${hits} lần dùng lại từ cache kể từ khi khởi động máy chủ.`;
}

// ---------- models ----------
function renderModelSwitch() {
  for (const m of state.models) {
    const label = document.querySelector(`label[for="m-${m.id}"]`);
    const input = $(`m-${m.id}`);
    if (!label || !input) continue;
    label.innerHTML = esc(m.name) + (m.placeholder ? ' <span class="tag">chưa gắn</span>' : "");
    input.checked = m.id === state.modelId;
  }
}

function renderModelCards() {
  $("modelCards").innerHTML = state.models.map((m) => `
    <article class="modelcard" data-placeholder="${m.placeholder}">
      <div><h3>${esc(m.name)}</h3><p class="modelcard__tag">${esc(m.tagline)}</p></div>
      <dl>${m.facts.map((f) => `<div><dt>${esc(f.label)}</dt><dd>${f.value.includes("/") && f.value.endsWith(".py") ? `<code>${esc(f.value)}</code>` : esc(f.value)}</dd></div>`).join("")}
      </dl>
    </article>`).join("");
}

function renderConfig() {
  const m = model();
  const v = values();
  const onChange = (key, val) => {
    state.params[state.modelId] = { ...(state.params[state.modelId] || {}), [key]: val };
    saved.set("params", state.params);
    syncPresets($("presets"), m, values());
    if (["normalize_vi", "split_sentences", "speed"].includes(key)) scheduleNormalize();
  };
  renderParams(m, v, { basic: $("paramsBasic"), advanced: $("paramsAdvanced"), pipeline: $("paramsPipeline") }, onChange);
  renderPresets($("presets"), m, v, (key, steps) => { onChange(key, steps); renderConfig(); });
}

function selectModel(id) {
  state.modelId = id;
  saved.set("model", id);
  renderModelSwitch();
  renderConfig();
  updateGo();
  const m = model();
  const noVoice = m && m.needs_voice === false;
  $("voiceReq").hidden = noVoice;
  if (noVoice) note(`${m.name} không dùng giọng mẫu.`);
  else if ($("voiceNote").textContent.includes("không dùng giọng mẫu")) note("");
  if (!state.result) idleScreen();
}

// ---------- device screen ----------
function screen({ pct = "", text = false, stage = "", detail = "", value = null, tone = "" }) {
  $("pct").textContent = pct;
  $("pct").dataset.text = String(text);
  $("stage").textContent = stage;
  $("detail").textContent = detail;
  $("device").dataset.state = tone;
  if (value === null) $("screen").removeAttribute("aria-valuenow");
  else $("screen").setAttribute("aria-valuenow", String(Math.round(value)));
}

function idleScreen() {
  const m = model();
  if (m?.placeholder) {
    matrix.dark();
    screen({ pct: "Chưa gắn", text: true, stage: `${m.name} chưa có trên máy chủ`, detail: "backend/engines/model2_engine.py" });
    return;
  }
  matrix.idle();
  const ready = state.voice || m?.needs_voice === false;
  screen({ pct: "Chưa có audio", text: true, stage: ready ? "" : "Cần giọng mẫu", value: 0 });
}

// ---------- go button ----------
function goBlocker() {
  const m = model();
  if (!state.status) return { hard: true, msg: "Máy chủ chưa phản hồi. Kiểm tra server rồi tải lại trang." };
  if (m?.placeholder) return { hard: true, msg: `${m.name} chưa được gắn. Thêm mô hình vào backend/engines/model2_engine.py.` };
  if (engineState() === "error") return { hard: true, msg: `Không nạp được ${m.name}: ${state.status.engines[m.id].error}` };
  return null;
}

function updateGo() {
  const block = goBlocker();
  const busy = !!state.job;
  const btn = $("generate");
  btn.disabled = busy || !!block?.hard;
  btn.setAttribute("aria-busy", String(busy));
  btn.textContent = busy ? "Đang đọc" : "Đọc bằng giọng này";
  $("cancel").hidden = !busy;
  let why = block?.msg || "";
  if (!why && engineState() === "loading") why = "Đang nạp mô hình";
  if (!why && state.status?.mock) why = "Chế độ thử, âm thanh giả";
  $("goWhy").textContent = busy ? "" : why;
}

// ---------- voices ----------
async function refreshVoices() {
  try {
    state.voices = (await api.voices()).voices;
  } catch (e) {
    state.voices = [];
  }
  renderSaved();
}

function renderSaved() {
  const own = state.voices.filter((v) => !v.preset);
  const presets = state.voices.filter((v) => v.preset).sort((a, b) => a.order - b.order);
  $("savedCount").textContent = own.length ? String(own.length) : "";
  $("savedEmpty").hidden = own.length > 0;
  $("savedList").innerHTML = own.map((v) => `
    <li>
      <button type="button" class="saved__pick" data-pick="${v.id}" aria-current="${state.voice?.id === v.id}">
        <span class="saved__name">${esc(v.name)}</span>
        <span class="saved__meta">${secs(v.duration)}</span>
      </button>
      <button type="button" class="linkbtn" data-del="${v.id}" aria-label="Xoá giọng ${esc(v.name)}">Xoá</button>
    </li>`).join("");
  $("presetList").innerHTML = presets.map((v) => `
    <li>
      <button type="button" class="iconbtn iconbtn--small" data-preview="${v.id}" aria-label="Nghe thử ${esc(v.name)}" data-playing="${previewing === v.id}">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path class="i-play" d="M8 5.5v13l10-6.5z" fill="currentColor"/><path class="i-pause" d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z" fill="currentColor"/></svg>
      </button>
      <button type="button" class="saved__pick" data-pick="${v.id}" aria-current="${state.voice?.id === v.id}">
        <span class="saved__name">${esc(v.name)}</span>
        <span class="saved__meta">${secs(v.duration)}</span>
      </button>
    </li>`).join("");
}

// one shared player for previewing presets
const previewAudio = new Audio();
let previewing = null;
previewAudio.addEventListener("ended", () => { previewing = null; renderSaved(); });
function togglePreview(id) {
  if (previewing === id) { previewAudio.pause(); previewing = null; }
  else { previewAudio.src = api.voiceAudioUrl(id); previewAudio.play().catch(() => {}); previewing = id; sampleAudio.pause(); }
  renderSaved();
}

async function setVoice(meta, { quiet = false } = {}) {
  state.voice = meta;
  saved.set("voiceId", meta?.id || null);
  sampleAudio.pause();
  $("samplePlay").dataset.playing = "false";
  if (!meta) { $("sample").hidden = true; renderSaved(); updateGo(); return; }
  $("sample").hidden = false;
  $("sampleName").textContent = meta.name;
  $("sampleDur").textContent = secs(meta.duration);
  $("refText").value = state.refTexts[meta.id] ?? (meta.preset ? meta.transcript : "");
  $("refText").placeholder = meta.transcript
    ? `Vọng đã chép: ${meta.transcript}`
    : "Để trống thì Vọng tự chép lời, kết quả được lưu lại cho lần sau";
  sampleAudio.src = api.voiceAudioUrl(meta.id);
  if (!quiet) {
    const d = meta.duration;
    if (meta.reused) note("Đã có trong cache.");
    else if (meta.truncated) note(`Đã cắt còn ${secs(d)} đầu tiên.`);
    else if (d < 3) note("Hơi ngắn, nên 3 đến 10 giây.");
    else if (d > 15) note("Hơi dài, nên 3 đến 10 giây.");
    else note("");
  }
  renderSaved();
  updateGo();
  if (!state.job && !state.result) idleScreen();
  try {
    const { peaks } = await loadPeaks(api.voiceAudioUrl(meta.id), 400);
    state.samplePeaks = peaks;
    drawStrip($("sampleWave"), peaks, 0);
  } catch { /* the strip is optional */ }
}

async function upload(file, source = "upload") {
  if (!file) return;
  const max = (state.status?.limits.max_upload_mb || 25) * 1048576;
  if (file.size > max) return note(`File lớn hơn ${state.status?.limits.max_upload_mb || 25} MB.`, "error");
  const drop = $("drop");
  drop.setAttribute("aria-busy", "true");
  note(source === "record" ? "Đang lưu bản thu" : `Đang tải lên và giải mã ${file.name}`);
  try {
    const meta = await api.uploadVoice(file, { source, name: file.name.replace(/\.[^.]+$/, "") });
    await refreshVoices();
    await setVoice(meta);
    refreshStatus();
  } catch (e) {
    note(e.message, "error");
  } finally {
    drop.removeAttribute("aria-busy");
    $("fileInput").value = "";
  }
}

function showSource(src) {
  document.querySelectorAll(".source").forEach((el) => { el.hidden = el.dataset.source !== src; });
}

// ---------- recorder ----------
let recorder = null;
async function toggleRecord() {
  const btn = $("recBtn");
  if (recorder?.recording) { recorder.stop(); return; }
  if (!Recorder.supported()) return note("Trình duyệt không hỗ trợ thu âm.", "error");
  recorder = new Recorder({
    canvas: $("recCanvas"),
    maxSeconds: Math.min(30, state.status?.limits.max_ref_seconds || 30),
    onTick: (s) => {
      $("recTime").textContent = nf1.format(s);
    },
    onStop: (file, seconds) => {
      btn.setAttribute("aria-pressed", "false");
      $("recLabel").textContent = "Thu lại";
      if (seconds < 1) return note("Bản thu quá ngắn.", "error");
      upload(file, "record");
    },
  });
  try {
    await recorder.start();
    btn.setAttribute("aria-pressed", "true");
    $("recLabel").textContent = "Dừng thu";
    note("");
  } catch (e) {
    note(e.name === "NotAllowedError" ? "Trình duyệt chưa được cấp quyền micro. Cho phép micro rồi thử lại." : "Không mở được micro.", "error");
  }
}

// ---------- text ----------
let normTimer;
function scheduleNormalize() {
  clearTimeout(normTimer);
  normTimer = setTimeout(runNormalize, 350);
}
async function runNormalize() {
  const text = $("text").value;
  const v = values();
  if (!text.trim()) { $("segments").replaceChildren(); return; }
  try {
    const res = await api.normalize(text, v.normalize_vi ?? true, v.split_sentences ?? true);
    const pause = (ms) => `<span class="pausemark">nghỉ ${nf.format(ms / 1000)} s</span>`;
    $("segments").innerHTML = res.segments.map((s, i) =>
      `<li>${res.pauses_before[i] ? pause(res.pauses_before[i]) : ""}${esc(s)}${i === res.segments.length - 1 && res.tail_ms ? pause(res.tail_ms) : ""}</li>`).join("");
  } catch { /* keep the last preview */ }
}
function onTextInput() {
  const t = $("text");
  $("charCount").textContent = String(t.value.length);
  $("charCount").parentElement.dataset.over = String(t.maxLength > 0 && t.value.length >= t.maxLength);
  t.removeAttribute("aria-invalid");
  saved.set("text", t.value);
  scheduleNormalize();
}

// ---------- generate ----------
async function generate() {
  if (state.job) return;
  const block = goBlocker();
  if (block) return toast(block.msg);
  if (!state.voice && model().needs_voice !== false) {
    note("Cần giọng mẫu.", "error");
    document.querySelector('label[for="src-upload"]').scrollIntoView({ block: "center" });
    $("src-upload").focus();
    return;
  }
  const text = $("text").value.trim();
  if (!text) {
    $("text").setAttribute("aria-invalid", "true");
    $("text").focus();
    $("goWhy").textContent = "Nhập văn bản cần đọc.";
    return;
  }
  resultAudio.pause();
  state.job = { pending: true };
  updateGo();
  matrix.setProgress(0);
  screen({ pct: "0%", stage: "Đang gửi yêu cầu", value: 0 });
  try {
    const job = await api.createJob({
      engine: state.modelId, voice_id: state.voice?.id || "", text,
      ref_text: $("refText").value.trim(), params: values(),
    });
    state.job = job;
    onJob(job);
    state.stopWatch = api.watchJob(job.id, onJob);
  } catch (e) {
    state.job = null;
    updateGo();
    matrix.idle();
    screen({ pct: "Lỗi", text: true, stage: e.message, tone: "error" });
    toast(e.message);
  }
}

function onJob(j) {
  if (!state.job || (state.job.id && state.job.id !== j.id)) return;
  state.job = j;
  const pct = Math.round(j.progress * 100);
  if (j.status === "queued") {
    matrix.setProgress(0);
    screen({ pct: "0%", stage: j.queue_position > 0 ? `Đang xếp hàng, còn ${j.queue_position} job phía trước` : "Đang chờ tới lượt", value: 0 });
  } else if (j.status === "running") {
    matrix.setProgress(j.progress);
    screen({ pct: `${pct}%`, stage: j.stage, detail: j.detail, value: pct });
  } else if (j.status === "done") {
    finishJob();
    showResult(j.result, { autoplay: true });
    refreshHistory();
    refreshStatus();
    refreshVoices();
  } else if (j.status === "cancelled") {
    finishJob();
    matrix.idle();
    screen({ pct: "Đã huỷ", text: true, stage: "Các câu đã sinh xong vẫn được giữ trong cache." });
  } else if (j.status === "error") {
    finishJob();
    matrix.dark();
    screen({ pct: "Lỗi", text: true, stage: j.error || "Không sinh được audio.", tone: "error" });
    toast(j.error || "Không sinh được audio.");
  }
}

function finishJob() {
  state.stopWatch?.();
  state.stopWatch = null;
  state.job = null;
  updateGo();
}

async function cancel() {
  const id = state.job?.id;
  if (!id) return;
  $("cancel").disabled = true;
  try { await api.cancelJob(id); } catch (e) { toast(e.message); }
  $("cancel").disabled = false;
}

// ---------- result + playback ----------
async function showResult(r, { autoplay = false, label = "" } = {}) {
  state.result = r;
  ["sElapsed", "sDur", "sRtf", "sCache"].forEach((id) => { $(id).dataset.empty = "false"; });
  $("stats").hidden = false;
  $("sElapsed").textContent = secs(r.elapsed);
  $("sDur").textContent = secs(r.duration);
  $("sRtf").textContent = r.rtf == null ? "chưa đo" : nf.format(r.rtf);
  $("sCache").textContent = r.result_cached ? "toàn bộ" : `${r.segments_cached}/${r.segments} câu`;
  const dl = $("download");
  dl.href = r.url;
  dl.removeAttribute("aria-disabled");
  dl.removeAttribute("tabindex");
  $("play").disabled = false;
  $("tTotal").textContent = mmss(r.duration);
  $("tNow").textContent = "0:00";
  resultAudio.src = r.url;
  const how = r.result_cached ? "Lấy nguyên từ cache, không chạy GPU" :
    r.segments_cached ? `${r.segments - r.segments_cached} câu mới sinh, ${r.segments_cached} câu lấy từ cache` : `${r.segments} câu, sinh theo lô`;
  screen({ pct: mmss(r.duration), text: false, stage: label || how, detail: `Sinh trong ${secs(r.elapsed)}`, value: 100 });
  if (r.ref_text && !$("refText").value.trim() && state.voice) {
    $("refText").placeholder = `Vọng đã chép: ${r.ref_text}`;
  }
  try {
    const { peaks } = await loadPeaks(r.url, 800);
    matrix.setWave(peaks);
  } catch { matrix.idle(); }
  if (autoplay) resultAudio.play().catch(() => {});
  highlightHistory();
}

function tick() {
  if (!resultAudio.duration) return;
  const f = resultAudio.currentTime / resultAudio.duration;
  matrix.setPlayhead(f);
  $("tNow").textContent = mmss(resultAudio.currentTime);
  if (!resultAudio.paused) requestAnimationFrame(tick);
}

// ---------- history ----------
async function refreshHistory() {
  try {
    state.history = (await api.history()).items;
    $("historyError").hidden = true;
  } catch {
    $("historyBody").replaceChildren();
    $("historyError").hidden = false;
    return;
  }
  renderHistory();
}

function renderHistory() {
  const body = $("historyBody");
  $("historyEmpty").hidden = state.history.length > 0;
  $("historyTable").hidden = state.history.length === 0;
  const LIMIT = 6;
  const more = $("historyMore");
  more.hidden = state.history.length <= LIMIT;
  more.textContent = state.historyAll ? "Thu gọn" : `Xem tất cả ${state.history.length} lần`;
  more.setAttribute("aria-expanded", String(!!state.historyAll));
  body.innerHTML = state.history.map((h, i) => {
    const steps = h.params.num_step ?? h.params.nfe_step ?? "";
    const cache = h.result_cached ? "toàn bộ" : `${h.segments_cached}/${h.segments}`;
    return `<tr data-url="${h.url}"${!state.historyAll && i >= LIMIT ? " hidden" : ""}>
      <td class="when">${when(h.created)}</td>
      <td class="text" title="${esc(h.text)}"><span>${esc(h.text.length > 90 ? h.text.slice(0, 88).trimEnd() + "…" : h.text)}</span></td>
      <td>${esc(h.engine_name)}</td>
      <td class="num" data-label="Bước">${steps}</td>
      <td class="num" data-label="Thời gian sinh">${h.result_cached ? "từ cache" : secs(h.elapsed)}</td>
      <td class="num" data-label="Độ dài">${secs(h.duration)}</td>
      <td class="num" data-label="RTF">${h.rtf == null ? "chưa đo" : nf.format(h.rtf)}</td>
      <td class="num" data-label="Cache">${cache}</td>
      <td class="acts">
        <button type="button" class="linkbtn" data-listen="${h.id}">Nghe</button>
        <button type="button" class="linkbtn" data-reuse="${h.id}">Dùng lại</button>
        <a class="linkbtn" href="${h.url}" download>Tải</a>
      </td></tr>`;
  }).join("");
  highlightHistory();
}

function highlightHistory() {
  document.querySelectorAll("#historyBody tr").forEach((tr) => {
    tr.setAttribute("aria-current", String(tr.dataset.url === state.result?.url));
  });
}

function reuse(h) {
  if (state.models.some((m) => m.id === h.engine)) selectModel(h.engine);
  state.params[h.engine] = { ...h.params };
  saved.set("params", state.params);
  renderConfig();
  $("text").value = h.text;
  onTextInput();
  const v = state.voices.find((x) => x.id === h.voice_id);
  if (v) setVoice(v, { quiet: true });
  else if (h.voice_id) note("Giọng mẫu này đã bị xoá.", "error");
  $("studio").scrollIntoView({ block: "start" });
}

// ---------- wiring ----------
function wire() {
  $("modelSwitch").addEventListener("change", (e) => selectModel(e.target.value));
  $("sourceSwitch").addEventListener("change", (e) => showSource(e.target.value));

  $("fileInput").addEventListener("change", (e) => upload(e.target.files[0]));
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.dataset.over = "true"; }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => { drop.dataset.over = "false"; }));
  drop.addEventListener("drop", (e) => { e.preventDefault(); upload(e.dataTransfer.files[0]); });

  $("recBtn").addEventListener("click", toggleRecord);

  $("presetList").addEventListener("click", (e) => {
    const prev = e.target.closest("[data-preview]");
    const pick = e.target.closest("[data-pick]");
    if (prev) togglePreview(prev.dataset.preview);
    if (pick) setVoice(state.voices.find((v) => v.id === pick.dataset.pick));
  });
  $("savedList").addEventListener("click", async (e) => {
    const pick = e.target.closest("[data-pick]");
    const del = e.target.closest("[data-del]");
    if (pick) setVoice(state.voices.find((v) => v.id === pick.dataset.pick));
    if (del) {
      const v = state.voices.find((x) => x.id === del.dataset.del);
      if (!confirm(`Xoá giọng “${v.name}” và voice prompt đã mã hoá của nó?`)) return;
      try {
        await api.deleteVoice(v.id);
        if (state.voice?.id === v.id) setVoice(null);
        await refreshVoices();
        refreshStatus();
      } catch (err) { toast(err.message); }
    }
  });

  $("samplePlay").addEventListener("click", () => {
    if (sampleAudio.paused) sampleAudio.play().catch(() => {}); else sampleAudio.pause();
  });
  sampleAudio.addEventListener("play", () => { $("samplePlay").dataset.playing = "true"; requestAnimationFrame(sampleTick); });
  sampleAudio.addEventListener("pause", () => { $("samplePlay").dataset.playing = "false"; });
  sampleAudio.addEventListener("ended", () => { drawStrip($("sampleWave"), state.samplePeaks, 0); });
  function sampleTick() {
    if (sampleAudio.duration) drawStrip($("sampleWave"), state.samplePeaks, sampleAudio.currentTime / sampleAudio.duration);
    if (!sampleAudio.paused) requestAnimationFrame(sampleTick);
  }

  $("refText").addEventListener("input", (e) => {
    if (!state.voice) return;
    state.refTexts[state.voice.id] = e.target.value;
    saved.set("refTexts", state.refTexts);
  });

  $("text").addEventListener("input", onTextInput);
  $("insertPause").addEventListener("click", () => {
    const t = $("text");
    const start = t.selectionStart ?? t.value.length, end = t.selectionEnd ?? start;
    const before = t.value.slice(0, start), after = t.value.slice(end);
    const mark = `${before && !/\s$/.test(before) ? " " : ""}[nghỉ 1s]${after && !/^\s/.test(after) ? " " : ""}`;
    t.value = before + mark + after;
    t.focus();
    const caret = before.length + mark.indexOf("1s");
    t.setSelectionRange(caret, caret + 2);  // select "1s" so the duration can be typed over
    onTextInput();
  });
  document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => {
    $("text").value = SAMPLES[b.dataset.sample];
    onTextInput();
    $("text").focus();
  }));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); generate(); }
  });

  $("generate").addEventListener("click", generate);
  $("cancel").addEventListener("click", cancel);
  $("resetParams").addEventListener("click", () => {
    delete state.params[state.modelId];
    saved.set("params", state.params);
    renderConfig();
    scheduleNormalize();
  });

  $("play").addEventListener("click", () => {
    if (resultAudio.paused) resultAudio.play().catch(() => {}); else resultAudio.pause();
  });
  resultAudio.addEventListener("play", () => { $("device").dataset.playing = "true"; requestAnimationFrame(tick); });
  resultAudio.addEventListener("pause", () => { $("device").dataset.playing = "false"; tick(); });
  resultAudio.addEventListener("ended", () => { matrix.setPlayhead(0); $("tNow").textContent = "0:00"; });
  $("matrix").addEventListener("click", (e) => {
    if (!state.result || !resultAudio.duration || state.job) return;
    resultAudio.currentTime = matrix.columnAt(e.clientX) * resultAudio.duration;
    tick();
  });
  $("screen").addEventListener("click", (e) => {
    if (e.target === $("matrix")) return;
    $("matrix").dispatchEvent(new MouseEvent("click", { clientX: e.clientX }));
  });

  $("historyBody").addEventListener("click", (e) => {
    const listen = e.target.closest("[data-listen]");
    const re = e.target.closest("[data-reuse]");
    const h = state.history.find((x) => x.id === (listen || re)?.dataset.listen || x.id === (listen || re)?.dataset.reuse);
    if (!h) return;
    if (listen) { showResult(h, { autoplay: true, label: "Phát lại từ lịch sử" }); $("device").scrollIntoView({ block: "center" }); }
    if (re) reuse(h);
  });

  $("historyMore").addEventListener("click", () => { state.historyAll = !state.historyAll; renderHistory(); });
  $("clearCache").addEventListener("click", async () => {
    if (!confirm("Xoá voice prompt, các câu đã sinh, kết quả và lịch sử? Giọng mẫu vẫn được giữ.")) return;
    try {
      await api.clearCache();
      state.result = null;
      await Promise.all([refreshStatus(), refreshHistory()]);
      idleScreen();
    } catch (e) { toast(e.message); }
  });

  // hero actions reuse the studio's own controls
  $("heroRecord").addEventListener("click", () => {
    $("src-record").checked = true;
    showSource("record");
    $("studio").scrollIntoView({ block: "start" });
    $("recBtn").focus({ preventScroll: true });
  });
  $("heroUpload").addEventListener("click", () => {
    $("src-upload").checked = true;
    showSource("upload");
    $("fileInput").click();
  });


  window.addEventListener("resize", () => state.samplePeaks && drawStrip($("sampleWave"), state.samplePeaks, 0));
}

async function init() {
  wire();
  $("text").value = saved.get("text", SAMPLES.hello);
  onTextInput();
  try {
    state.models = (await api.models()).models;
  } catch (e) {
    screen({ pct: "Mất kết nối", text: true, stage: e.message, tone: "error" });
    toast(e.message);
    return;
  }
  if (!state.models.some((m) => m.id === state.modelId)) state.modelId = state.models[0].id;
  renderModelSwitch();
  renderModelCards();
  renderConfig();
  await Promise.all([refreshStatus(), refreshVoices(), refreshHistory()]);
  const vid = saved.get("voiceId", null);
  const v = state.voices.find((x) => x.id === vid);
  if (v) await setVoice(v, { quiet: true });
  else showSource(document.querySelector('input[name="source"]:checked').value);
  selectModel(state.modelId);
}

init();

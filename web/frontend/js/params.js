// Builds the settings panel from the model's parameter schema (served by /api/models).

const fmt = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 2 });

export function formatValue(p, v) {
  if (p.type === "bool") return v ? "bật" : "tắt";
  if (p.type === "select") return (p.options.find((o) => o.value === v) || {}).label || v;
  if (p.key.endsWith("_ms")) return `${fmt.format(v)} ms`;
  if (p.key === "speed") return `${fmt.format(v)}×`;
  return fmt.format(v);
}

function setFill(input) {
  const min = Number(input.min), max = Number(input.max);
  input.style.setProperty("--fill", `${((Number(input.value) - min) / (max - min)) * 100}%`);
}

function control(p, value, onChange, disabled) {
  const id = `p-${p.key}`;
  const wrap = document.createElement("div");
  wrap.className = `param param--${p.type}`;
  const help = "";
  const tip = p.help ? ` title="${p.help.replace(/"/g, "&quot;")}"` : "";
  const described = "";

  if (p.type === "bool") {
    wrap.innerHTML = `
      <label class="param__label" for="${id}"${tip}>${p.label}</label>
      <span class="switch"><input type="checkbox" role="switch" id="${id}" ${described} ${value ? "checked" : ""} ${disabled ? "disabled" : ""}><span></span></span>
      ${help}`;
    wrap.querySelector("input").addEventListener("change", (e) => onChange(p.key, e.target.checked));
    return wrap;
  }

  if (p.type === "select") {
    const opts = p.options.map((o) => `<option value="${o.value}" ${o.value === value ? "selected" : ""}>${o.label}</option>`).join("");
    wrap.innerHTML = `
      <div class="param__top"><label class="param__label" for="${id}"${tip}>${p.label}</label>
      <select id="${id}" ${described} ${disabled ? "disabled" : ""}>${opts}</select></div>
      ${help}`;
    wrap.querySelector("select").addEventListener("change", (e) => onChange(p.key, e.target.value));
    return wrap;
  }

  wrap.innerHTML = `
    <div class="param__top"><label class="param__label" for="${id}"${tip}>${p.label}</label>
    <output class="param__value" for="${id}" id="${id}-out">${formatValue(p, value)}</output></div>
    <input type="range" id="${id}" min="${p.min}" max="${p.max}" step="${p.step}" value="${value}" ${described} ${disabled ? "disabled" : ""}>
    ${help}`;
  const input = wrap.querySelector("input");
  const out = wrap.querySelector("output");
  setFill(input);
  input.addEventListener("input", () => {
    const v = p.type === "int" ? parseInt(input.value, 10) : parseFloat(input.value);
    out.textContent = formatValue(p, v);
    setFill(input);
    onChange(p.key, v);
  });
  return wrap;
}

export function renderParams(model, values, targets, onChange, disabled = false) {
  for (const el of Object.values(targets)) el.replaceChildren();
  for (const p of model.params) {
    const target = targets[p.group] || targets.basic;
    target.append(control(p, values[p.key] ?? p.default, onChange, disabled));
  }
}

export function defaultsOf(model) {
  return Object.fromEntries(model.params.map((p) => [p.key, p.default]));
}

// Presets move only the decoding-steps knob, which is the speed/quality trade-off the report studies.
export function stepsParam(model) {
  return model.params.find((p) => p.group === "basic" && /step/.test(p.key));
}

export const PRESETS = [
  { id: "fast", label: "Nhanh", steps: 8 },
  { id: "balanced", label: "Cân bằng", steps: 16 },
  { id: "quality", label: "Chất lượng", steps: 32 },
];

export function renderPresets(fieldset, model, values, onPick, disabled = false) {
  const sp = stepsParam(model);
  fieldset.querySelectorAll("input, label").forEach((n) => n.remove());
  fieldset.hidden = !sp;
  if (!sp) return;
  for (const pr of PRESETS) {
    const id = `preset-${pr.id}`;
    const input = document.createElement("input");
    Object.assign(input, { type: "radio", name: "preset", id, value: pr.id, disabled });
    input.checked = values[sp.key] === pr.steps;
    input.addEventListener("change", () => onPick(sp.key, pr.steps));
    const label = document.createElement("label");
    label.htmlFor = id;
    label.innerHTML = `${pr.label} <span class="count num">${pr.steps}</span>`;
    fieldset.append(input, label);
  }
}

export function syncPresets(fieldset, model, values) {
  const sp = stepsParam(model);
  if (!sp) return;
  fieldset.querySelectorAll("input").forEach((i) => {
    const pr = PRESETS.find((x) => x.id === i.value);
    i.checked = pr && values[sp.key] === pr.steps;
  });
}

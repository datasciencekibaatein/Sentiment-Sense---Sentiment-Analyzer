// SentimentSense - front-end logic
// Talks to the Flask server: POST /predict and POST /predict_batch

const $ = (id) => document.getElementById(id);

const MODEL_NAMES = { lstm: "LSTM", rnn: "Simple RNN" };

const EXAMPLES = {
  positive: "This movie was absolutely wonderful. The acting was superb, the story kept me hooked, and I left the cinema smiling.",
  negative: "Terrible plot, wooden acting and a painfully slow second half. A complete waste of time and money.",
  tricky: "I expected to hate it, but it slowly won me over. Not perfect, though the ending was not bad at all."
};

// ---------- Helpers ----------
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  let data = {};
  try { data = await res.json(); } catch (e) { /* ignore */ }
  if (!res.ok) throw new Error(data.error || `Server error (${res.status})`);
  return data;
}

function setMessage(el, text, isError = false) {
  el.textContent = text;
  el.classList.toggle("is-error", isError);
}

const pct = (x) => `${(x * 100).toFixed(1)}%`;

// ---------- Tabs ----------
function showTab(name) {
  for (const t of ["single", "batch"]) {
    const active = t === name;
    $(`tab-${t}`).classList.toggle("is-active", active);
    $(`tab-${t}`).setAttribute("aria-selected", String(active));
    $(`panel-${t}`).hidden = !active;
  }
}
$("tab-single").addEventListener("click", () => showTab("single"));
$("tab-batch").addEventListener("click", () => showTab("batch"));

// ---------- One review ----------
document.querySelectorAll("[data-example]").forEach((btn) => {
  btn.addEventListener("click", () => {
    $("reviewInput").value = EXAMPLES[btn.dataset.example];
    $("reviewInput").focus();
  });
});

function buildResultCard(modelKey, r) {
  const card = document.createElement("article");
  card.className = "result-card";

  const top = document.createElement("div");
  top.className = "result-top";
  const name = document.createElement("span");
  name.className = "model-name";
  name.textContent = MODEL_NAMES[modelKey] || modelKey;
  const verdict = document.createElement("span");
  verdict.className = `verdict ${r.label}`;
  verdict.textContent = r.label;
  top.append(name, verdict);

  const meter = document.createElement("div");
  meter.className = "meter";
  const track = document.createElement("div");
  track.className = "meter-track";
  const marker = document.createElement("div");
  marker.className = "meter-marker";
  track.appendChild(marker);
  const scale = document.createElement("div");
  scale.className = "meter-scale";
  scale.innerHTML = "<span>Negative</span><span>Positive</span>";
  meter.append(track, scale);

  const detail = document.createElement("p");
  detail.className = "detail";
  detail.textContent = `Positive score ${pct(r.score)}. Confidence ${pct(r.confidence)}.`;

  card.append(top, meter, detail);

  // move the marker after the card is on the page, so it slides into place
  requestAnimationFrame(() => requestAnimationFrame(() => {
    marker.style.left = `${r.score * 100}%`;
  }));
  return card;
}

async function analyzeSingle() {
  const text = $("reviewInput").value.trim();
  const model = $("modelSelect").value;
  const msg = $("singleMessage");
  const box = $("singleResults");

  if (!text) {
    setMessage(msg, "Enter a review to analyze.", true);
    box.innerHTML = "";
    return;
  }

  setMessage(msg, "Analyzing...");
  $("analyzeBtn").disabled = true;
  try {
    const data = await postJSON("/predict", { text, model });
    box.innerHTML = "";
    for (const [key, r] of Object.entries(data.results)) {
      box.appendChild(buildResultCard(key, r));
    }
    setMessage(msg, "");
  } catch (err) {
    box.innerHTML = "";
    setMessage(msg, `${err.message} Check that the server is running.`, true);
  } finally {
    $("analyzeBtn").disabled = false;
  }
}

$("analyzeBtn").addEventListener("click", analyzeSingle);
$("reviewInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyzeSingle();
});

// ---------- Many reviews ----------
let lastBatch = null;   // { texts, results } kept for the CSV download

function splitCSVLine(line) {
  const out = [];
  let cur = "", inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (ch === '"') inQuotes = false;
      else cur += ch;
    } else if (ch === '"') inQuotes = true;
    else if (ch === ",") { out.push(cur); cur = ""; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}

function parseReviews(content, filename) {
  const lines = content.split(/\r?\n/).filter((l) => l.trim() !== "");
  if (!filename.toLowerCase().endsWith(".csv")) return lines.map((l) => l.trim());

  const header = splitCSVLine(lines[0]).map((h) => h.trim().toLowerCase());
  let col = header.findIndex((h) => h === "review" || h === "text");
  let start = 1;
  if (col === -1) {          // no header found: use the first column of every line
    col = 0;
    start = 0;
  }
  return lines.slice(start).map((l) => (splitCSVLine(l)[col] || "").trim()).filter(Boolean);
}

function renderBatch(texts, data) {
  const { total, positive, negative, uncertain } = data.summary;
  const share = (n) => (total ? (n / total) * 100 : 0);

  const bar = $("stackBar");
  bar.innerHTML = "";
  [["seg-pos", positive], ["seg-mid", uncertain], ["seg-neg", negative]].forEach(([cls, n]) => {
    const s = document.createElement("span");
    s.className = cls;
    s.style.width = `${share(n)}%`;
    bar.appendChild(s);
  });

  const counts = $("counts");
  counts.innerHTML = "";
  [["Reviews", total], ["Positive", positive], ["Uncertain", uncertain], ["Negative", negative]].forEach(([label, n]) => {
    const wrap = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = label === "Reviews" ? String(n) : `${n} (${share(n).toFixed(0)}%)`;
    wrap.append(dt, dd);
    counts.appendChild(wrap);
  });

  const body = $("rowsBody");
  body.innerHTML = "";
  texts.slice(0, 25).forEach((t, i) => {
    const r = data.results[i];
    const tr = document.createElement("tr");
    const cells = [String(i + 1), t.length > 140 ? t.slice(0, 140) + "..." : t, r.label, pct(r.confidence)];
    cells.forEach((value, c) => {
      const td = document.createElement("td");
      td.textContent = value;           // textContent keeps uploaded text safe
      if (c === 1) td.className = "review";
      if (c === 2) td.className = `tag-${r.label}`;
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });

  $("batchResults").hidden = false;
}

async function analyzeFile() {
  const file = $("fileInput").files[0];
  const msg = $("batchMessage");
  if (!file) {
    setMessage(msg, "Choose a .txt or .csv file first.", true);
    return;
  }

  setMessage(msg, "Reading file...");
  $("batchBtn").disabled = true;
  try {
    const content = await file.text();
    const texts = parseReviews(content, file.name);
    if (texts.length === 0) throw new Error("No reviews found in the file.");

    setMessage(msg, `Analyzing ${texts.length} reviews...`);
    const data = await postJSON("/predict_batch", { texts, model: $("batchModel").value });
    lastBatch = { texts, results: data.results };
    renderBatch(texts, data);
    setMessage(msg, "");
  } catch (err) {
    $("batchResults").hidden = true;
    setMessage(msg, err.message, true);
  } finally {
    $("batchBtn").disabled = false;
  }
}
$("batchBtn").addEventListener("click", analyzeFile);

function downloadCSV() {
  if (!lastBatch) return;
  const esc = (s) => `"${String(s).replace(/"/g, '""')}"`;
  const rows = ["review,label,positive_score,confidence"];
  lastBatch.texts.forEach((t, i) => {
    const r = lastBatch.results[i];
    rows.push([esc(t), r.label, r.score, r.confidence].join(","));
  });
  const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "sentimentsense_results.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}
$("downloadBtn").addEventListener("click", downloadCSV);
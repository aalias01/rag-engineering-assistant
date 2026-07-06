const API_BASE = (window.RAG_CONFIG && window.RAG_CONFIG.API_BASE) || "http://localhost:8000";
const REFUSAL_PHRASE = "I don't have information on that in the provided documents.";
const COLD_START_LINE = "This runs on a free tier that sleeps between visitors. First start takes 30 to 60 seconds; runs after that are quick.";

const DOCS = {
  "doe_hdbk_1012_v1_thermodynamics.pdf": "DOE 1012-1",
  "doe_hdbk_1012_v2_heat_transfer.pdf": "DOE 1012-2",
  "doe_hdbk_1018_v2_mechanical_science.pdf": "DOE 1018-2",
  "nasa_systems_engineering_handbook.pdf": "NASA SP-2016",
  "osha_3132_process_safety_management.pdf": "OSHA 3132",
  "doe_final_rule_2017_cac_hp_efficiency.pdf": "82 FR 1786",
};

const els = {};
const state = {
  queries: [],
  remaining: [],
  gradedRuns: 0,
  theme: "day",
};

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  setInitialTheme();
  bindEvents();
  clearRunSurface();
  loadGradedQueries();
  checkHealth();
});

function cacheElements() {
  els.statusLine = document.getElementById("status-line");
  els.modeToggle = document.getElementById("mode-toggle");
  els.questionText = document.getElementById("question-text");
  els.answerText = document.getElementById("answer-text");
  els.gradedLine = document.getElementById("graded-line");
  els.error = document.getElementById("reading-error");
  els.runWarm = document.getElementById("run-warm-meter");
  els.gradedButton = document.getElementById("graded-button");
  els.queryButton = document.getElementById("query-button");
  els.queryInput = document.getElementById("query-input");
  els.stream = document.getElementById("stream-toggle");
  els.hybrid = document.getElementById("hybrid-toggle");
  els.reranker = document.getElementById("reranker-toggle");
  els.topK = document.getElementById("top-k");
  els.runLog = document.getElementById("run-log");
  els.citationList = document.getElementById("citation-list");
  els.chunksList = document.getElementById("chunks-list");
  els.stats = {
    latency: document.getElementById("stat-latency"),
    chunks: document.getElementById("stat-chunks"),
    prompt: document.getElementById("stat-prompt"),
    completion: document.getElementById("stat-completion"),
    model: document.getElementById("stat-model"),
  };
}

function bindEvents() {
  els.modeToggle.addEventListener("click", () => {
    applyTheme(state.theme === "day" ? "night" : "day", true);
  });

  els.gradedButton.addEventListener("click", runGradedQuestion);
  els.queryButton.addEventListener("click", runOwnQuestion);

  els.queryInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      runOwnQuestion();
    }
  });

  els.topK.addEventListener("change", () => {
    const value = Math.max(1, Math.min(10, Number(els.topK.value || 4)));
    els.topK.value = String(value);
  });
}

function setInitialTheme() {
  const initial = document.documentElement.getAttribute("data-theme") || "day";
  applyTheme(initial, false);
}

function applyTheme(theme, persist) {
  state.theme = theme;
  const mode = theme === "night" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", theme);
  els.modeToggle.setAttribute("aria-pressed", theme === "night" ? "true" : "false");
  els.modeToggle.setAttribute("aria-label", theme === "night" ? "Switch to light mode" : "Switch to dark mode");

  if (persist) {
    localStorage.setItem("mode", mode);
    document.cookie = `mode=${mode}; Max-Age=31536000; Path=/; Domain=.alvinalias.com; SameSite=Lax`;
  }
}

async function loadGradedQueries() {
  try {
    const response = await fetch("AA-03_graded_queries.json", { cache: "no-store" });
    const data = await response.json();
    state.queries = data.queries || [];
    resetRotation();
  } catch {
    state.queries = [];
    state.remaining = [];
  }
}

function resetRotation() {
  state.remaining = shuffle([...state.queries]);
}

function shuffle(items) {
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

async function checkHealth() {
  els.statusLine.textContent = "checking server";

  try {
    const response = await fetch(`${API_BASE}/health`, { signal: timeoutSignal(70000) });
    const data = await response.json();

    if (response.ok && data.chroma_loaded) {
      els.statusLine.textContent = `${data.collection_size} chunks loaded`;
    } else {
      els.statusLine.textContent = "no vector store loaded";
    }
  } catch {
    els.statusLine.textContent = "server unreachable right now";
  }
}

function timeoutSignal(ms) {
  const controller = new AbortController();
  window.setTimeout(() => controller.abort(), ms);
  return controller.signal;
}

async function runGradedQuestion() {
  if (!state.queries.length) {
    await loadGradedQueries();
  }
  if (!state.queries.length) {
    showError("Query failed. [NEW STRING] graded questions did not load");
    return;
  }

  if (!state.remaining.length) {
    resetRotation();
  }

  let sample;
  if (state.gradedRuns === 0) {
    const firstPool = state.remaining.filter((query) => query.query_type !== "out_of_corpus");
    sample = firstPool[Math.floor(Math.random() * firstPool.length)];
    state.remaining = state.remaining.filter((query) => query !== sample);
  } else {
    sample = state.remaining.shift();
  }

  state.gradedRuns += 1;
  els.queryInput.value = sample.query;
  await runQuery(sample.query, { sample, runIndex: state.gradedRuns });
}

async function runOwnQuestion() {
  const query = els.queryInput.value.trim();
  if (!query) return;
  await runQuery(query, { sample: null, runIndex: null });
}

async function runQuery(query, context) {
  const controls = readControls();
  clearRunSurface();
  setBusy(true);
  els.questionText.textContent = query;

  if (context.sample) {
    if (context.sample.query_type === "out_of_corpus") {
      appendLog("> trap query · the correct answer is a refusal");
    } else {
      appendLog(`> query ${context.runIndex} of ${state.queries.length} · labeled ${docLabel(context.sample.expected_source_doc)} p.${formatPages(context.sample.expected_source_pages)}`);
    }
  }

  const warm = startWarmMeter(els.runWarm, { compact: false, onLog: appendLog });
  const startedAt = performance.now();
  let result = null;

  try {
    if (controls.useStream) {
      result = await runStreaming(query, controls, warm, startedAt);
    } else {
      result = await runBlocking(query, controls, warm);
    }
  } finally {
    setBusy(false);
  }

  if (!result) {
    return;
  }

  if (context.sample) {
    renderGrading(context.sample, result);
  } else {
    setGradedLine("your question · ungraded · check the citations against the shelf", false);
  }
}

function readControls() {
  return {
    useStream: els.stream.checked,
    useHybrid: els.hybrid.checked,
    useReranker: els.reranker.checked,
    topK: Math.max(1, Math.min(10, Number(els.topK.value || 4))),
  };
}

async function runBlocking(query, controls, warm) {
  try {
    const response = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: controls.topK,
        use_hybrid: controls.useHybrid,
        use_reranker: controls.useReranker,
      }),
    });

    const data = await response.json().catch(() => ({}));
    warm.ready();

    if (!response.ok) {
      showError(`Query failed. ${data.detail || response.statusText}`);
      return null;
    }

    renderAnswer(data.answer || "");
    renderEvidence(data.sources || [], data.chunks || []);
    renderChunks(data.chunks || []);
    renderStats(data);
    appendLog(`> answered in ${seconds(data.latency_ms)} s`);
    return data;
  } catch (error) {
    warm.stop();
    showError(`${COLD_START_LINE} Try again in a moment.`, error.message);
    return null;
  }
}

function runStreaming(query, controls, warm, startedAt) {
  const params = new URLSearchParams({
    q: query,
    top_k: String(controls.topK),
    use_hybrid: String(controls.useHybrid),
    use_reranker: String(controls.useReranker),
  });
  const url = `${API_BASE}/query/stream?${params.toString()}`;

  return new Promise((resolve) => {
    const source = new EventSource(url);
    let answer = "";
    let firstToken = false;

    source.onmessage = (event) => {
      const token = event.data;

      if (!firstToken) {
        firstToken = true;
        warm.ready();
      }

      if (token.startsWith("__METADATA__:")) {
        source.close();
        const latencyMs = Math.round(performance.now() - startedAt);
        try {
          const meta = JSON.parse(token.replace("__METADATA__:", ""));
          const result = {
            ...meta,
            answer,
            latency_ms: latencyMs,
          };
          renderEvidence(meta.sources || [], meta.chunks || []);
          renderChunks(meta.chunks || []);
          renderStats(result);
          appendLog(`> streamed in ${seconds(latencyMs)} s`);
          resolve(result);
        } catch (error) {
          showError(`Query failed. ${error.message}`);
          resolve(null);
        }
        return;
      }

      if (token.startsWith("ERROR:")) {
        source.close();
        warm.stop();
        showError(`Query failed. ${token.replace("ERROR:", "").trim()}`);
        resolve(null);
        return;
      }

      answer += token;
      renderAnswer(answer);
    };

    source.onerror = () => {
      source.close();
      warm.stop();
      const partial = answer ? "The partial answer above is incomplete. " : "";
      showError(`The stream dropped. ${partial}Run it again.`);
      resolve(null);
    };
  });
}

function renderGrading(sample, data) {
  if (sample.query_type === "out_of_corpus") {
    if (wasRefused(data)) {
      setGradedLine("graded · outside the shelf · declined · correct", false);
    } else {
      setGradedLine("graded · outside the shelf · it answered anyway · miss", true);
    }
    return;
  }

  const sources = data.sources || [];
  const expectedPages = sample.expected_source_pages || [];
  const match = sources.find((source) => {
    return sameDoc(source.source, sample.expected_source_doc) && expectedPages.includes(Number(source.page));
  });

  if (match) {
    setGradedLine(`graded · labeled source ${docLabel(sample.expected_source_doc)} pages ${formatPages(expectedPages)} · cited page ${match.page} · hit`, false);
  } else {
    setGradedLine(`graded · labeled source ${docLabel(sample.expected_source_doc)} pages ${formatPages(expectedPages)} · cited ${formatCited(sources)} · miss`, true);
  }
}

function wasRefused(data) {
  if (Object.prototype.hasOwnProperty.call(data, "refused")) {
    return data.refused === true;
  }
  return (data.answer || "").includes(REFUSAL_PHRASE);
}

function sameDoc(source, expected) {
  return fileName(source) === fileName(expected);
}

function fileName(value) {
  return String(value || "").split("/").pop();
}

function docLabel(source) {
  return DOCS[fileName(source)] || fileName(source) || "N/A";
}

function formatPages(pages) {
  if (!pages || pages.length === 0) return "none";
  if (pages.length > 1 && pages.every((page, index) => index === 0 || page === pages[index - 1] + 1)) {
    return `${pages[0]} to ${pages[pages.length - 1]}`;
  }
  return pages.join(", ");
}

function formatCited(sources) {
  if (!sources || sources.length === 0) return "none";
  return sources
    .slice(0, 3)
    .map((source) => `${docLabel(source.source)} p.${source.page}`)
    .join(", ");
}

function renderEvidence(sources, chunks) {
  const unique = [];
  const seen = new Set();

  sources.forEach((source) => {
    const key = `${fileName(source.source)}:${source.page}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(source);
    }
  });

  els.citationList.innerHTML = "";
  unique.forEach((source, index) => {
    const card = document.createElement("article");
    card.className = "citation-card";
    const chunk = chunks.find((item) => sameDoc(item.source, source.source) && Number(item.page) === Number(source.page));
    const method = citationMethod(chunk);
    card.innerHTML = `
      <h3>[${index + 1}] ${escapeHtml(docLabel(source.source))} · page ${escapeHtml(source.page)}</h3>
      <p>${escapeHtml(method)}</p>
    `;
    els.citationList.appendChild(card);
  });
}

function citationMethod(chunk) {
  if (!chunk) return "retrieved";
  const pieces = [chunk.retrieval_method || "retrieved"];
  if (chunk.rerank_score != null) pieces.push("reranked");
  return pieces.join(", ");
}

function renderChunks(chunks) {
  els.chunksList.innerHTML = "";
  chunks.forEach((chunk) => {
    const detail = document.createElement("details");
    detail.className = "chunk-row";
    const method = chunk.retrieval_method || "retrieved";
    const score = formatScore(chunk);
    detail.innerHTML = `
      <summary>${escapeHtml(docLabel(chunk.source))} p.${escapeHtml(chunk.page)} · ${escapeHtml(method)}${score}</summary>
      <p>${escapeHtml(chunk.text || "")}</p>
    `;
    els.chunksList.appendChild(detail);
  });
}

function formatScore(chunk) {
  if (chunk.rerank_score != null) return ` · rerank ${Number(chunk.rerank_score).toFixed(3)}`;
  if (chunk.rrf_score != null) return ` · rrf ${Number(chunk.rrf_score).toFixed(4)}`;
  return "";
}

function renderStats(data) {
  els.stats.latency.textContent = data.latency_ms != null ? `${seconds(data.latency_ms)} s` : "";
  els.stats.chunks.textContent = data.chunks_used ?? "";
  els.stats.prompt.textContent = data.prompt_tokens ?? "";
  els.stats.completion.textContent = data.completion_tokens ?? "";
  els.stats.model.textContent = data.model || "";
}

function renderAnswer(answer) {
  els.answerText.innerHTML = formatAnswer(answer);
}

function formatAnswer(answer) {
  const text = normalizeAnswer(answer);
  if (!text) return "";

  const blocks = text.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean);
  let html = "";
  let inOrderedList = false;

  blocks.forEach((block) => {
    const numbered = block.match(/^(\d+)\.\s+([\s\S]*)$/);
    if (numbered) {
      if (!inOrderedList) {
        html += "<ol>";
        inOrderedList = true;
      }
      html += `<li>${formatBlockContent(numbered[2])}</li>`;
      return;
    }

    if (inOrderedList) {
      html += "</ol>";
      inOrderedList = false;
    }

    html += `<p>${formatBlockContent(block)}</p>`;
  });

  if (inOrderedList) {
    html += "</ol>";
  }

  return html;
}

function normalizeAnswer(answer) {
  return String(answer || "")
    .trim()
    .replace(/\r\n/g, "\n")
    .replace(/\]\s*(?=\d+\.\s+\*\*)/g, "]\n\n")
    .replace(/(\*\*[^*\n][^*]*\*\*)\s*(?=\d+\.\s+\*\*)/g, "$1\n\n")
    .replace(/([.!?])\s+(?=\d+\.\s+\*\*)/g, "$1\n\n")
    .replace(/\s+\*\s+(?=[A-Z][A-Za-z /&-]+:)/g, "\n- ")
    .replace(/\s+\[Source:/g, "\n[Source:");
}

function formatBlockContent(block) {
  const lines = block.split(/\n+/).map((line) => line.trim()).filter(Boolean);
  const parts = [];
  let bullets = [];

  function flushBullets() {
    if (!bullets.length) return;
    parts.push(`<ul>${bullets.map((item) => `<li>${formatInline(item)}</li>`).join("")}</ul>`);
    bullets = [];
  }

  lines.forEach((line) => {
    const bullet = line.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      bullets.push(bullet[1]);
      return;
    }

    flushBullets();
    const formatted = formatInline(line);
    if (line.startsWith("[Source:")) {
      parts.push(`<span class="answer-source">${formatted}</span>`);
    } else {
      parts.push(formatted);
    }
  });

  flushBullets();
  return parts.join("<br>");
}

function formatInline(value) {
  return escapeHtml(value)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function setGradedLine(text, isMiss) {
  els.gradedLine.textContent = text;
  els.gradedLine.classList.toggle("miss", isMiss);
}

function appendLog(line) {
  const row = document.createElement("p");
  row.textContent = line;
  els.runLog.appendChild(row);
}

function showError(message, detail = "") {
  els.error.hidden = false;
  els.error.innerHTML = `<p>${escapeHtml(message)}</p>${detail ? `<pre>${escapeHtml(detail)}</pre>` : ""}`;
}

function clearRunSurface() {
  els.answerText.innerHTML = "";
  els.gradedLine.textContent = "";
  els.gradedLine.classList.remove("miss");
  els.error.hidden = true;
  els.error.innerHTML = "";
  els.runWarm.hidden = true;
  els.runWarm.innerHTML = "";
  els.runLog.innerHTML = "";
  els.citationList.innerHTML = "";
  els.chunksList.innerHTML = "";
  Object.values(els.stats).forEach((node) => {
    node.textContent = "";
  });
}

function setBusy(isBusy) {
  els.gradedButton.disabled = isBusy;
  els.queryButton.disabled = isBusy;
  els.queryInput.disabled = isBusy;
  els.stream.disabled = isBusy;
  els.hybrid.disabled = isBusy;
  els.reranker.disabled = isBusy;
  els.topK.disabled = isBusy;
}

function seconds(ms) {
  return (Number(ms || 0) / 1000).toFixed(1);
}

function startWarmMeter(host, options) {
  const startedAt = performance.now();
  const compact = options.compact;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let shown = false;
  let intervalId = null;
  const delayId = window.setTimeout(show, 2500);

  function show() {
    shown = true;
    host.hidden = false;
    host.innerHTML = warmMarkup(compact);
    update(60, "estimated seconds to warm", false);
    options.onShow?.();
    options.onLog?.("> server was asleep · sent the wake call");
    options.onLog?.("> warm-up estimate counting · this is an estimate, not progress");
    intervalId = window.setInterval(tick, 1000);
  }

  function tick() {
    const elapsed = Math.floor((performance.now() - startedAt) / 1000);
    const remaining = 60 - elapsed;
    if (remaining >= 0) {
      update(remaining, "estimated seconds to warm", false);
    } else {
      update(Math.abs(remaining), "seconds elapsed · still starting", true);
      if (!host.dataset.overrun) {
        host.dataset.overrun = "true";
        options.onLog?.("> past the usual window · still waiting, counting up honestly");
      }
    }
  }

  function update(value, label, overrun) {
    const number = host.querySelector("[data-warm-number]");
    const labelNode = host.querySelector("[data-warm-label]");
    const marker = host.querySelector("[data-warm-marker]");
    if (!number || !labelNode || !marker) return;

    const markerSeconds = overrun ? 0 : value;
    const x = 20 + ((60 - markerSeconds) / 60) * 280;
    number.textContent = String(value);
    labelNode.textContent = label;
    marker.setAttribute("points", `${x},7 ${x - 6},19 ${x + 6},19`);
    marker.classList.toggle("overrun", overrun);
  }

  return {
    ready() {
      window.clearTimeout(delayId);
      if (intervalId) window.clearInterval(intervalId);
      if (!shown) return;

      const measured = ((performance.now() - startedAt) / 1000).toFixed(1);
      update(0, "ready", false);
      options.onLog?.(`> awake · measured wake time ${measured} s`);

      const remove = () => {
        host.hidden = true;
        host.innerHTML = "";
        delete host.dataset.overrun;
      };

      if (reduceMotion) {
        remove();
      } else {
        window.setTimeout(remove, 4000);
      }
    },
    stop() {
      window.clearTimeout(delayId);
      if (intervalId) window.clearInterval(intervalId);
      host.hidden = true;
      host.innerHTML = "";
      delete host.dataset.overrun;
    },
  };
}

function warmMarkup(compact) {
  const ticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    .map((tick) => {
      const x = 20 + (tick / 60) * 280;
      const major = tick % 15 === 0;
      return `
        <line x1="${x}" y1="24" x2="${x}" y2="${major ? 10 : 16}"></line>
        ${major ? `<text x="${x}" y="39">${tick}</text>` : ""}
      `;
    })
    .join("");

  return `
    <div class="warm-meter${compact ? " compact" : ""}">
      <svg viewBox="0 0 320 44" aria-hidden="true">
        <line x1="20" y1="24" x2="300" y2="24" class="warm-baseline"></line>
        ${ticks}
        <polygon data-warm-marker points="300,7 294,19 306,19"></polygon>
      </svg>
      <div class="warm-readout">
        <strong data-warm-number>60</strong>
        <span data-warm-label>estimated seconds to warm</span>
      </div>
      <p>> warm-up estimate counting · this is an estimate, not progress</p>
      <p>${COLD_START_LINE}</p>
    </div>
  `;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

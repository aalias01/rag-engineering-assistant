// Frontend runtime for the Engineering Evidence Workbench.

const API_BASE = (window.RAG_CONFIG && window.RAG_CONFIG.API_BASE) || "http://localhost:8000";

const els = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindEvents();
  updateCharCount();
  checkHealth();
});

function cacheElements() {
  els.badge = document.getElementById("status-badge");
  els.conversation = document.getElementById("conversation");
  els.input = document.getElementById("query-input");
  els.send = document.getElementById("send-btn");
  els.charCount = document.getElementById("char-count");
  els.stream = document.getElementById("stream-toggle");
  els.hybrid = document.getElementById("hybrid-toggle");
  els.reranker = document.getElementById("reranker-toggle");
  els.topK = document.getElementById("top-k-select");
}

function bindEvents() {
  els.input.addEventListener("input", updateCharCount);
  els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendQuery();
    }
  });
  els.send.addEventListener("click", sendQuery);

  document.querySelectorAll(".example-btn").forEach((button) => {
    button.addEventListener("click", () => {
      els.input.value = button.dataset.query || button.textContent.trim();
      updateCharCount();
      els.input.focus();
    });
  });

  document.querySelectorAll(".doc-row").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".doc-row").forEach((row) => row.classList.remove("active"));
      button.classList.add("active");
    });
  });
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();

    if (data.chroma_loaded) {
      els.badge.textContent = `Ready / ${data.collection_size} chunks`;
      els.badge.className = "badge badge-ready";
    } else {
      els.badge.textContent = "Degraded / no vector store";
      els.badge.className = "badge badge-degraded";
    }
  } catch {
    els.badge.textContent = "API offline";
    els.badge.className = "badge badge-degraded";
  }
}

function updateCharCount() {
  els.charCount.textContent = `${els.input.value.length} / 1000`;
}

function readControls() {
  return {
    useStream: els.stream.checked,
    useHybrid: els.hybrid.checked,
    useReranker: els.reranker.checked,
    topK: Number(els.topK.value || 4),
  };
}

async function sendQuery() {
  const query = els.input.value.trim();
  if (!query) return;

  const controls = readControls();
  appendUserTurn(query);
  els.input.value = "";
  updateCharCount();
  clearSidePanel();
  setBusy(true);

  if (controls.useStream) {
    await sendStreaming(query, controls);
  } else {
    await sendBlocking(query, controls);
  }

  setBusy(false);
  els.input.focus();
}

function setBusy(isBusy) {
  els.send.disabled = isBusy;
  els.input.disabled = isBusy;
  els.topK.disabled = isBusy;
  els.stream.disabled = isBusy;
  els.hybrid.disabled = isBusy;
  els.reranker.disabled = isBusy;
  els.send.textContent = isBusy ? "Running" : "Run Query";
}

async function sendBlocking(query, controls) {
  const bubble = appendAssistantTurn("Retrieving evidence...", true);

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: controls.topK,
        use_hybrid: controls.useHybrid,
        use_reranker: controls.useReranker,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      bubble.textContent = `Error ${res.status}: ${err.detail || "Query failed."}`;
      bubble.classList.remove("streaming");
      return;
    }

    const data = await res.json();
    bubble.textContent = data.answer;
    bubble.classList.remove("streaming");

    renderSources(data.sources);
    renderChunks(data.chunks);
    renderStats({
      latency_ms: data.latency_ms,
      chunks_used: data.chunks_used,
      prompt_tokens: data.prompt_tokens,
      completion_tokens: data.completion_tokens,
      cost_usd: data.cost_usd,
      model: data.model,
    });
  } catch (err) {
    bubble.textContent = `Network error: ${err.message}`;
    bubble.classList.remove("streaming");
  }
}

async function sendStreaming(query, controls) {
  const bubble = appendAssistantTurn("", true);
  const cursor = document.createElement("span");
  cursor.className = "cursor-blink";
  bubble.appendChild(cursor);

  const t0 = performance.now();
  const url = `${API_BASE}/query/stream?q=${encodeURIComponent(query)}&top_k=${controls.topK}`;

  return new Promise((resolve) => {
    const es = new EventSource(url);
    let fullText = "";

    es.onmessage = (event) => {
      const token = event.data;

      if (token.startsWith("__METADATA__:")) {
        es.close();
        cursor.remove();
        bubble.classList.remove("streaming");

        try {
          const meta = JSON.parse(token.replace("__METADATA__:", ""));
          renderSources(meta.sources || []);
          renderChunks([]);
          renderStats({
            latency_ms: Math.round(performance.now() - t0),
            chunks_used: meta.chunks_used,
            prompt_tokens: "-",
            completion_tokens: "-",
            cost_usd: meta.cost_usd,
            model: meta.model,
          });
        } catch {
          renderChunks([]);
        }

        resolve();
        return;
      }

      fullText += token;
      bubble.textContent = fullText;
      bubble.appendChild(cursor);
      scrollConversationToBottom();
    };

    es.onerror = () => {
      es.close();
      cursor.remove();
      if (!fullText) bubble.textContent = "Stream error. Check the API connection.";
      bubble.classList.remove("streaming");
      resolve();
    };
  });
}

function appendUserTurn(text) {
  removeWelcome();

  const turn = document.createElement("article");
  turn.className = "turn user-turn";

  const label = document.createElement("div");
  label.className = "turn-meta";
  label.textContent = "Query";

  const bubble = document.createElement("div");
  bubble.className = "user-bubble";
  bubble.textContent = text;

  turn.append(label, bubble);
  els.conversation.appendChild(turn);
  scrollConversationToBottom();
}

function appendAssistantTurn(text, streaming = false) {
  const turn = document.createElement("article");
  turn.className = "turn assistant-turn";

  const label = document.createElement("div");
  label.className = "assistant-label";
  label.textContent = "Grounded answer";

  const bubble = document.createElement("div");
  bubble.className = `assistant-bubble${streaming ? " streaming" : ""}`;
  bubble.textContent = text;

  turn.append(label, bubble);
  els.conversation.appendChild(turn);
  scrollConversationToBottom();
  return bubble;
}

function removeWelcome() {
  const welcome = els.conversation.querySelector(".welcome-message");
  if (welcome) welcome.remove();
}

function scrollConversationToBottom() {
  els.conversation.scrollTop = els.conversation.scrollHeight;
}

function clearSidePanel() {
  document.getElementById("sources-list").innerHTML = '<p class="placeholder-text">Waiting for citations.</p>';
  document.getElementById("chunks-list").innerHTML = '<p class="placeholder-text">Waiting for retrieved chunks.</p>';
  renderStats({
    latency_ms: "-",
    chunks_used: "-",
    prompt_tokens: "-",
    completion_tokens: "-",
    cost_usd: "-",
    model: "-",
  });
}

function renderSources(sources) {
  const list = document.getElementById("sources-list");
  if (!sources || sources.length === 0) {
    list.innerHTML = '<p class="placeholder-text">No citations returned.</p>';
    return;
  }

  const seen = new Set();
  const unique = sources.filter((source) => {
    const key = `${source.source}::${source.page}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  list.innerHTML = unique
    .map((source) => `
      <div class="citation-card">
        <div class="citation-doc">${escapeHtml(source.source)}</div>
        <div class="citation-page">Page ${escapeHtml(source.page)}</div>
      </div>
    `)
    .join("");
}

function renderChunks(chunks) {
  const list = document.getElementById("chunks-list");
  if (!chunks || chunks.length === 0) {
    list.innerHTML = '<p class="placeholder-text">Chunk previews are returned by the full JSON endpoint.</p>';
    return;
  }

  list.innerHTML = chunks
    .map((chunk, index) => {
      const method = chunk.retrieval_method || (chunk.rerank_score != null ? "reranked" : "retrieved");
      const score = formatScore(chunk);
      return `
        <details class="chunk-item">
          <summary>
            <span class="chunk-rank">${index + 1}</span>
            <span class="chunk-title">${escapeHtml(chunk.source)} p.${escapeHtml(chunk.page)}</span>
            <span class="chunk-meta">${escapeHtml(method)}${score ? ` / ${score}` : ""}</span>
          </summary>
          <pre class="chunk-body">${escapeHtml(chunk.text)}</pre>
        </details>
      `;
    })
    .join("");
}

function formatScore(chunk) {
  if (chunk.rerank_score != null) return `rerank ${Number(chunk.rerank_score).toFixed(3)}`;
  if (chunk.rrf_score != null) return `rrf ${Number(chunk.rrf_score).toFixed(4)}`;
  return "";
}

function renderStats({ latency_ms, chunks_used, prompt_tokens, completion_tokens, cost_usd, model }) {
  document.getElementById("stat-latency").textContent =
    latency_ms !== "-" ? `${latency_ms} ms` : "-";
  document.getElementById("stat-chunks").textContent = chunks_used ?? "-";
  document.getElementById("stat-prompt-tokens").textContent = prompt_tokens ?? "-";
  document.getElementById("stat-completion-tokens").textContent = completion_tokens ?? "-";
  document.getElementById("stat-cost").textContent =
    cost_usd != null && cost_usd !== "-" ? `$${Number(cost_usd).toFixed(5)}` : "-";
  document.getElementById("stat-model").textContent = model ?? "-";
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

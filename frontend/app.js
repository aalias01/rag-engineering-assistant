// RAG Engineering Assistant — frontend logic
// Communicates with FastAPI backend via REST (POST /query) or SSE (GET /query/stream)

// API_BASE comes from window.RAG_CONFIG (see config.js). This avoids per-environment
// code edits — point at the deployed Render URL by changing config.js (or injecting
// a window.RAG_CONFIG override via Vercel project settings) rather than this file.
const API_BASE = (window.RAG_CONFIG && window.RAG_CONFIG.API_BASE) || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Health check on load
// ---------------------------------------------------------------------------

async function checkHealth() {
  const badge = document.getElementById("status-badge");
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error();
    const data = await res.json();
    if (data.chroma_loaded) {
      badge.textContent = `Ready · ${data.collection_size} chunks`;
      badge.className = "badge badge-ready";
    } else {
      badge.textContent = "Degraded — no documents ingested";
      badge.className = "badge badge-degraded";
    }
  } catch {
    badge.textContent = "API offline";
    badge.className = "badge badge-degraded";
  }
}

checkHealth();

// ---------------------------------------------------------------------------
// Example query buttons
// ---------------------------------------------------------------------------

function useExample(btn) {
  document.getElementById("query-input").value = btn.textContent.trim();
  updateCharCount();
}

// ---------------------------------------------------------------------------
// Character count
// ---------------------------------------------------------------------------

document.getElementById("query-input").addEventListener("input", updateCharCount);
document.getElementById("query-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
});

function updateCharCount() {
  const val = document.getElementById("query-input").value.length;
  document.getElementById("char-count").textContent = `${val} / 1000`;
}

// ---------------------------------------------------------------------------
// Send query
// ---------------------------------------------------------------------------

async function sendQuery() {
  const input = document.getElementById("query-input");
  const query = input.value.trim();
  if (!query) return;

  const useStream = document.getElementById("stream-toggle").checked;

  appendUserBubble(query);
  input.value = "";
  updateCharCount();
  clearSidePanel();
  document.getElementById("send-btn").disabled = true;

  if (useStream) {
    await sendStreaming(query);
  } else {
    await sendBlocking(query);
  }

  document.getElementById("send-btn").disabled = false;
}

// ---------------------------------------------------------------------------
// Blocking (POST /query)
// ---------------------------------------------------------------------------

async function sendBlocking(query) {
  const bubble = appendAssistantBubble("Thinking…", true);
  const t0 = performance.now();

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 4, use_hybrid: true, use_reranker: true }),
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

// ---------------------------------------------------------------------------
// Streaming (GET /query/stream — SSE)
// ---------------------------------------------------------------------------

async function sendStreaming(query) {
  const bubble = appendAssistantBubble("", true);
  const cursor = document.createElement("span");
  cursor.className = "cursor-blink";
  bubble.appendChild(cursor);

  const t0 = performance.now();
  const url = `${API_BASE}/query/stream?q=${encodeURIComponent(query)}&top_k=4`;

  return new Promise((resolve) => {
    const es = new EventSource(url);
    let fullText = "";

    es.onmessage = (e) => {
      const token = e.data;

      // Final metadata token
      if (token.startsWith("__METADATA__:")) {
        es.close();
        cursor.remove();
        bubble.classList.remove("streaming");
        try {
          const meta = JSON.parse(token.replace("__METADATA__:", ""));
          renderSources(meta.sources || []);
          renderStats({
            latency_ms: Math.round(performance.now() - t0),
            chunks_used: meta.chunks_used,
            prompt_tokens: "—",
            completion_tokens: "—",
            cost_usd: meta.cost_usd,
            model: meta.model,
          });
        } catch {}
        resolve();
        return;
      }

      fullText += token;
      // Update bubble text, keeping cursor at end
      bubble.textContent = fullText;
      bubble.appendChild(cursor);
      scrollConversationToBottom();
    };

    es.onerror = () => {
      es.close();
      cursor.remove();
      if (!fullText) bubble.textContent = "Stream error — check if API is running.";
      bubble.classList.remove("streaming");
      resolve();
    };
  });
}

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

function appendUserBubble(text) {
  const conv = document.getElementById("conversation");
  // Remove welcome message on first query
  const welcome = conv.querySelector(".welcome-message");
  if (welcome) welcome.remove();

  const turn = document.createElement("div");
  turn.className = "turn";

  const bubble = document.createElement("div");
  bubble.className = "user-bubble";
  bubble.textContent = text;
  turn.appendChild(bubble);
  conv.appendChild(turn);
  scrollConversationToBottom();
}

function appendAssistantBubble(text, streaming = false) {
  const conv = document.getElementById("conversation");

  const turn = document.createElement("div");
  turn.className = "turn";

  const label = document.createElement("div");
  label.className = "assistant-label";
  label.textContent = "Assistant";
  turn.appendChild(label);

  const bubble = document.createElement("div");
  bubble.className = "assistant-bubble" + (streaming ? " streaming" : "");
  bubble.textContent = text;
  turn.appendChild(bubble);
  conv.appendChild(turn);
  scrollConversationToBottom();
  return bubble;
}

function scrollConversationToBottom() {
  const conv = document.getElementById("conversation");
  conv.scrollTop = conv.scrollHeight;
}

function clearSidePanel() {
  document.getElementById("sources-list").innerHTML = '<p class="placeholder-text">Loading…</p>';
  document.getElementById("chunks-list").innerHTML = '<p class="placeholder-text">Loading…</p>';
  document.getElementById("stat-latency").textContent = "—";
  document.getElementById("stat-chunks").textContent = "—";
  document.getElementById("stat-prompt-tokens").textContent = "—";
  document.getElementById("stat-completion-tokens").textContent = "—";
  document.getElementById("stat-cost").textContent = "—";
  document.getElementById("stat-model").textContent = "—";
}

// ---------------------------------------------------------------------------
// Source citations
// ---------------------------------------------------------------------------

function renderSources(sources) {
  const list = document.getElementById("sources-list");
  if (!sources || sources.length === 0) {
    list.innerHTML = '<p class="placeholder-text">No sources cited.</p>';
    return;
  }

  // Deduplicate by source+page
  const seen = new Set();
  const unique = sources.filter((s) => {
    const key = `${s.source}::${s.page}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  list.innerHTML = unique
    .map(
      (s) => `
    <div class="source-pill">
      <div class="doc-name">${escapeHtml(s.source)}</div>
      <div class="page-num">Page ${s.page}</div>
    </div>`
    )
    .join("");
}

// ---------------------------------------------------------------------------
// Retrieved chunks (expandable accordion)
// ---------------------------------------------------------------------------

function renderChunks(chunks) {
  const list = document.getElementById("chunks-list");
  if (!chunks || chunks.length === 0) {
    list.innerHTML = '<p class="placeholder-text">No chunks available.</p>';
    return;
  }

  list.innerHTML = chunks
    .map(
      (c, i) => `
    <div class="chunk-item">
      <div class="chunk-header" onclick="toggleChunk(${i})">
        <span><span class="chunk-source">${escapeHtml(c.source)}</span> <span class="chunk-page">p.${c.page}</span></span>
        <span class="chunk-toggle" id="toggle-${i}">▼</span>
      </div>
      <div class="chunk-body" id="chunk-body-${i}">
${escapeHtml(c.text)}
      </div>
      ${c.rerank_score != null ? `<div class="chunk-score">Rerank score: ${c.rerank_score.toFixed(3)}</div>` : ""}
    </div>`
    )
    .join("");
}

function toggleChunk(i) {
  const body = document.getElementById(`chunk-body-${i}`);
  const toggle = document.getElementById(`toggle-${i}`);
  body.classList.toggle("open");
  toggle.classList.toggle("open");
}

// ---------------------------------------------------------------------------
// Stats panel
// ---------------------------------------------------------------------------

function renderStats({ latency_ms, chunks_used, prompt_tokens, completion_tokens, cost_usd, model }) {
  document.getElementById("stat-latency").textContent =
    latency_ms !== "—" ? `${latency_ms} ms` : "—";
  document.getElementById("stat-chunks").textContent = chunks_used ?? "—";
  document.getElementById("stat-prompt-tokens").textContent = prompt_tokens ?? "—";
  document.getElementById("stat-completion-tokens").textContent = completion_tokens ?? "—";
  document.getElementById("stat-cost").textContent =
    cost_usd != null && cost_usd !== "—" ? `$${cost_usd.toFixed(5)}` : "—";
  document.getElementById("stat-model").textContent = model ?? "—";
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

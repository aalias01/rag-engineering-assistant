// Frontend runtime config.
//
// Exposes window.RAG_CONFIG so app.js can read API_BASE without code changes
// per environment.
//
// Local dev: keep the default below.
// Production: edit API_BASE to the deployed Render URL before deploying to
// Vercel, OR override window.RAG_CONFIG in an inline <script> tag injected
// via Vercel project settings.
window.RAG_CONFIG = window.RAG_CONFIG || {
  // Local dev (served from localhost) talks to a local API; anything else
  // (the deployed Vercel site) talks to the deployed Render API.
  API_BASE:
    location.hostname === "localhost" || location.hostname === "127.0.0.1"
      ? "http://localhost:8000"
      : "https://rag-engineering-assistant-api.onrender.com",
};

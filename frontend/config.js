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
  API_BASE: "http://localhost:8000",
};

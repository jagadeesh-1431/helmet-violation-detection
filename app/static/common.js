// Helpers shared across all pages.
window.HV = (() => {
  "use strict";
  const $ = (id) => document.getElementById(id);

  function make(tag, className, text) {
    const n = document.createElement(tag);
    if (className) n.className = className;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function fmtBytes(n) {
    if (n === null || n === undefined || isNaN(n)) return "0 KB";
    return n >= 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB";
  }

  function fmtTime(s) {
    if (s === null || s === undefined || isNaN(s)) return "0:00";
    const m = Math.floor(s / 60);
    return m + ":" + String(Math.floor(s % 60)).padStart(2, "0");
  }

  function fmtNumber(n) {
    if (n === null || n === undefined || isNaN(n)) return "0";
    return Number(n).toLocaleString();
  }

  async function errorFrom(res) {
    try {
      const body = await res.json();
      if (typeof body.detail === "string") return body.detail;
      if (Array.isArray(body.detail)) return "Invalid request: " + body.detail.map((d) => d.msg).join("; ");
    } catch (_) { /* not JSON */ }
    return "Request failed (" + res.status + ").";
  }

  // Fills the model badge in the header and sidebar. Resolves with the /api/health payload (or null).
  async function loadHealth() {
    const badge = $("model-status");
    const sbFile = $("sb-model-file");
    const sbDev = $("sb-device");
    const heroStatus = $("hero-engine-status");
    try {
      const res = await fetch("/api/health");
      const h = await res.json();
      if (h.model_loaded) {
        if (badge) {
          badge.className = "model-status ok";
          badge.textContent = "Model ready: " + h.model_file + " on " + h.device.toUpperCase();
        }
        if (sbFile) sbFile.textContent = h.model_file;
        if (sbDev) sbDev.textContent = h.device.toUpperCase();
        if (heroStatus) heroStatus.textContent = h.model_file + " (" + h.device.toUpperCase() + ")";
      } else {
        if (badge) {
          badge.className = "model-status bad";
          badge.textContent = "Model not loaded. " + (h.model_error || "");
          badge.title = "Looking for: " + (h.model_path || h.model_file) + (h.model_path_exists ? " (file found)" : " (file NOT found)")
            + ". Click to check again.";
        }
        if (sbFile) sbFile.textContent = "Not Loaded";
        if (sbDev) sbDev.textContent = "Unavailable";
        if (heroStatus) heroStatus.textContent = "Offline / Unloaded";
      }
      return h;
    } catch (_) {
      if (badge) {
        badge.className = "model-status bad";
        badge.textContent = "Cannot reach the server.";
      }
      return null;
    }
  }

  return { $, make, fmtBytes, fmtTime, fmtNumber, errorFrom, loadHealth };
})();

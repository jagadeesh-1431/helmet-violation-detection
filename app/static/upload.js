// Traffic Intelligence System — Overview & Video Analysis Modules
(() => {
  "use strict";
  const { $, fmtBytes, fmtNumber, errorFrom, loadHealth, make } = window.HV;

  const el = {
    dropzone: $("dropzone"), fileInput: $("file-input"), uploadMsg: $("upload-msg"),
    uploadProgress: $("upload-progress"), uploadBar: $("upload-bar"),
    inputWrap: $("input-wrap"), inputVideo: $("input-video"), inputMeta: $("input-meta"), inputNote: $("input-note"),
    conf: $("conf"), confOut: $("conf-out"), imgsz: $("imgsz"), stride: $("stride"),
    sanity: $("sanity"), track: $("track"), diag: $("diag"), outwidth: $("outwidth"),
    start: $("start"), startHint: $("start-hint"), startMsg: $("start-msg"),
    overviewMod: $("overview-module"), analyzeMod: $("analyze-module"),
    navOverview: $("nav-mod-overview"), navAnalyze: $("nav-mod-analyze"),
  };

  const state = {
    videoId: null, filename: null, maxMb: 500, extensions: [],
    modelReady: false, starting: false, currentTab: "overview",
  };

  const show = (n, on = true) => { if (n) n.hidden = !on; };
  const text = (n, t) => { if (n) n.textContent = t; };

  // ------------------------------------------------------------------ Tab Switching
  function switchModule(targetTab) {
    state.currentTab = targetTab;
    const isAnalyze = targetTab === "analyze";

    if (el.overviewMod) el.overviewMod.classList.toggle("active", !isAnalyze);
    if (el.analyzeMod) el.analyzeMod.classList.toggle("active", isAnalyze);

    if (el.navOverview) el.navOverview.classList.toggle("active", !isAnalyze);
    if (el.navAnalyze) el.navAnalyze.classList.toggle("active", isAnalyze);

    if (isAnalyze && window.location.hash !== "#analyze") {
      history.replaceState(null, "", "#analyze");
    } else if (!isAnalyze && window.location.hash === "#analyze") {
      history.replaceState(null, "", "#overview");
    }
  }

  // ------------------------------------------------------------------ Start Button State
  function updateStart() {
    const ready = !!state.videoId && state.modelReady && !state.starting;
    if (el.start) el.start.disabled = !ready;
    if (!el.startHint) return;

    if (state.starting) text(el.startHint, "Starting detection…");
    else if (!state.videoId) text(el.startHint, "Upload a video to continue.");
    else if (!state.modelReady) text(el.startHint, "The model is not loaded. Check model status in sidebar.");
    else text(el.startHint, "Detection runs on the server. You will move to the results page.");
  }

  function uploadError(msg) {
    if (!el.uploadMsg) return;
    text(el.uploadMsg, msg);
    show(el.uploadMsg, !!msg);
  }

  function showVideo(videoId, filename, info, previewUrl, sizeBytes) {
    state.videoId = videoId;
    state.filename = filename;
    const v = info || {};
    const metaStr = (filename ? filename + ": " : "")
      + (v.width && v.height ? v.width + "×" + v.height : "")
      + (v.fps ? " at " + Math.round(v.fps) + " FPS" : "")
      + (v.duration_s ? ", " + v.duration_s + " s" : "")
      + (sizeBytes ? ", " + fmtBytes(sizeBytes) : "");

    text(el.inputMeta, metaStr);
    show(el.inputNote, false);
    if (el.inputVideo) el.inputVideo.src = previewUrl;
    show(el.inputWrap, true);
    updateStart();

    // Switch to analyze workspace so the user sees the preview immediately
    switchModule("analyze");
  }

  // ------------------------------------------------------------------ File Upload Handlers
  function handleFile(file) {
    if (!file) return;
    uploadError("");
    const dot = file.name.lastIndexOf(".");
    const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
    if (state.extensions.length && !state.extensions.includes(ext)) {
      return uploadError("Unsupported file type '" + (ext || "none") + "'. Allowed: " + state.extensions.join(", ") + ".");
    }
    if (file.size === 0) return uploadError("That file is empty.");
    if (file.size > state.maxMb * 1048576) {
      return uploadError("That file is " + fmtBytes(file.size) + ". The limit is " + state.maxMb + " MB.");
    }

    state.videoId = null;
    show(el.inputWrap, false);
    updateStart();

    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    show(el.uploadProgress, true);
    if (el.uploadBar) el.uploadBar.style.width = "0%";

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && el.uploadBar) {
        const pct = Math.round((e.loaded / e.total) * 100);
        el.uploadBar.style.width = pct + "%";
      }
    };

    xhr.onload = () => {
      show(el.uploadProgress, false);
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) { /* ignore */ }
      if (xhr.status !== 200) {
        return uploadError(typeof body.detail === "string" ? body.detail : "Upload failed (" + xhr.status + ").");
      }
      showVideo(body.video_id, body.filename, body.video, body.preview_url, body.size_bytes);
    };

    xhr.onerror = () => {
      show(el.uploadProgress, false);
      uploadError("The upload could not be completed. Check that the server is running.");
    };

    xhr.send(form);
  }

  // Returning from the results page with ?video=<id>: reuse previously uploaded video
  async function restoreFromQuery() {
    const id = new URLSearchParams(window.location.search).get("video");
    if (!id || !/^[0-9a-f]{32}$/.test(id)) return;
    try {
      const res = await fetch("/api/uploads/" + id);
      if (!res.ok) throw new Error(await errorFrom(res));
      const info = await res.json();
      showVideo(info.video_id, null, info.video, info.preview_url, info.size_bytes);
      switchModule("analyze");
    } catch (_) {
      uploadError("The earlier video is no longer on the server. Please upload it again.");
    }
  }

  // ------------------------------------------------------------------ Start Detection
  async function startDetection() {
    if (!state.videoId) return;
    state.starting = true;
    show(el.startMsg, false);
    updateStart();

    const payload = {
      video_id: state.videoId,
      filename: state.filename,
      confidence: el.conf ? Number(el.conf.value) : 0.25,
      imgsz: el.imgsz ? Number(el.imgsz.value) : 640,
      frame_stride: el.stride ? Number(el.stride.value) : 1,
      track: el.track ? el.track.checked : false,
      sanity_filter: el.sanity ? el.sanity.checked : true,
      diagnostics: el.diag ? el.diag.checked : false,
      output_width: el.outwidth ? Number(el.outwidth.value) : 1280,
    };

    try {
      const res = await fetch("/api/jobs", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await errorFrom(res));
      const job = await res.json();
      window.location.assign("/results/" + job.job_id);
    } catch (err) {
      state.starting = false;
      text(el.startMsg, err.message);
      show(el.startMsg, true);
      updateStart();
    }
  }

  // ------------------------------------------------------------------ System Stats
  async function loadSystemStats() {
    try {
      const res = await fetch("/api/system/stats");
      if (!res.ok) return;
      const data = await res.json();

      text($("dash-val-jobs"), fmtNumber(data.completed_jobs));
      text($("dash-val-frames"), fmtNumber(data.total_frames_processed));

      const totalDets = data.total_detections !== undefined ? data.total_detections :
        Object.values(data.detection_totals || {}).reduce((a, b) => a + b, 0);
      text($("dash-val-detections"), fmtNumber(totalDets));

      if (data.latest_duration_s !== null && data.latest_duration_s !== undefined) {
        text($("dash-val-duration"), data.latest_duration_s + " s");
      } else {
        text($("dash-val-duration"), "None yet");
      }
    } catch (_) {
      // Backend stats fallback
    }
  }

  // ------------------------------------------------------------------ Recent Analyses
  async function loadHistory() {
    const tbody = $("history-table-body");
    if (!tbody) return;

    try {
      const res = await fetch("/api/history");
      if (!res.ok) throw new Error("Failed to load history");
      const data = await res.json();
      const jobs = data.jobs || [];

      tbody.replaceChildren();

      if (!jobs.length) {
        const tr = make("tr");
        const td = make("td", "text-center py-4 text-muted", "No completed analyses found yet. Upload and analyze a video to view records here.");
        td.colSpan = 6;
        tr.append(td);
        tbody.append(tr);
        return;
      }

      jobs.forEach((j) => {
        const tr = make("tr");

        // Filename
        const tdFile = make("td", "font-bold", j.filename || "Traffic Video");

        // Frames
        const tdFrames = make("td", "font-mono", fmtNumber(j.frames_read));

        // Duration
        const tdElapsed = make("td", "", j.elapsed_s ? j.elapsed_s + " s" : "n/a");

        // Total detections
        const totalD = j.total_detections !== undefined ? j.total_detections :
          Object.values(j.counts || {}).reduce((a, b) => a + b, 0);
        const tdDets = make("td", "font-mono font-bold text-blue", fmtNumber(totalD));

        // Date/Time
        const tdDate = make("td", "text-muted", j.date_formatted || "Recent");

        // Action
        const tdAction = make("td");
        const aView = make("a", "btn btn-sm btn-secondary", "View Results");
        aView.href = j.url;
        tdAction.append(aView);

        tr.append(tdFile, tdFrames, tdElapsed, tdDets, tdDate, tdAction);
        tbody.append(tr);
      });
    } catch (_) {
      tbody.replaceChildren();
      const tr = make("tr");
      const td = make("td", "text-center py-4 text-muted", "Could not load history from server.");
      td.colSpan = 6;
      tr.append(td);
      tbody.append(tr);
    }
  }

  // ------------------------------------------------------------------ Event Listeners
  if (el.inputVideo) el.inputVideo.addEventListener("error", () => show(el.inputNote, true));

  if (el.dropzone) {
    el.dropzone.addEventListener("click", () => el.fileInput.click());
    el.dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.fileInput.click(); }
    });
    ["dragenter", "dragover"].forEach((t) => el.dropzone.addEventListener(t, (e) => {
      e.preventDefault(); el.dropzone.classList.add("drag");
    }));
    ["dragleave", "drop"].forEach((t) => el.dropzone.addEventListener(t, (e) => {
      e.preventDefault(); el.dropzone.classList.remove("drag");
    }));
    el.dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
  }

  if (el.fileInput) {
    el.fileInput.addEventListener("change", () => {
      handleFile(el.fileInput.files[0]);
      el.fileInput.value = "";
    });
  }

  if (el.conf) {
    el.conf.addEventListener("input", () => text(el.confOut, Number(el.conf.value).toFixed(2)));
  }

  if (el.start) el.start.addEventListener("click", startDetection);

  // Tab & Module Navigation Events
  const heroBtnStart = $("hero-btn-start-analysis");
  if (heroBtnStart) {
    heroBtnStart.addEventListener("click", () => {
      switchModule("analyze");
      if (el.analyzeMod) el.analyzeMod.scrollIntoView({ behavior: "smooth" });
    });
  }

  const btnBackOverview = $("btn-back-to-overview");
  if (btnBackOverview) {
    btnBackOverview.addEventListener("click", () => {
      switchModule("overview");
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  if (el.navOverview) el.navOverview.addEventListener("click", (e) => { e.preventDefault(); switchModule("overview"); });
  if (el.navAnalyze) el.navAnalyze.addEventListener("click", (e) => { e.preventDefault(); switchModule("analyze"); });

  // Refresh History
  const btnRefreshHist = $("btn-refresh-history");
  if (btnRefreshHist) btnRefreshHist.addEventListener("click", () => loadHistory());

  // Model health checks
  async function recheckModel() {
    const h = await loadHealth();
    if (h) { state.modelReady = h.model_loaded; updateStart(); }
  }

  const modelBadge = $("model-status");
  if (modelBadge) modelBadge.addEventListener("click", () => { if (!state.modelReady) recheckModel(); });
  window.addEventListener("focus", () => { if (!state.modelReady) recheckModel(); });

  // Handle URL hash changes
  window.addEventListener("hashchange", () => {
    if (window.location.hash === "#analyze") switchModule("analyze");
    else if (window.location.hash === "#overview" || !window.location.hash) switchModule("overview");
  });

  // Initialization
  (async () => {
    const pageApp = $("main-page-app");
    const defaultTab = pageApp ? pageApp.dataset.defaultTab : "overview";

    if (window.location.hash === "#analyze" || defaultTab === "analyze") {
      switchModule("analyze");
    } else {
      switchModule("overview");
    }

    const h = await loadHealth();
    if (h) {
      state.maxMb = h.max_upload_mb;
      state.extensions = h.allowed_extensions;
      state.modelReady = h.model_loaded;
    }
    updateStart();
    loadSystemStats();
    loadHistory();
    restoreFromQuery();
  })();
})();

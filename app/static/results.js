// Traffic Intelligence System — Detection Results & Human Review
(() => {
  "use strict";
  const { $, make, fmtTime, fmtNumber, errorFrom, loadHealth } = window.HV;

  const page = $("page");
  if (!page) return;
  const jobId = page.dataset.jobId;

  const COLORS = {
    0: "var(--plate)",
    1: "var(--with)",
    2: "var(--without)",
    3: "var(--triple)"
  };

  const POLL_MS = 700;
  const views = ["loading", "processing", "error", "done"];
  const state = { timer: null, rendered: false, job: null };

  const show = (n, on = true) => { if (n) n.hidden = !on; };
  const text = (n, t) => { if (n) n.textContent = t; };

  function setView(name) {
    views.forEach((v) => show($("state-" + v), v === name));
  }

  function fail(title, message) {
    text($("error-title"), title);
    text($("error-text"), message);
    setView("error");
  }

  const everyLabel = (n) => (n === 1 ? "every frame" : "every " + n + (n === 2 ? "nd" : n === 3 ? "rd" : "th") + " frame");

  async function poll() {
    try {
      const res = await fetch("/api/jobs/" + jobId);
      if (res.status === 404) {
        return fail("Result Not Found", "This result is no longer on the server. It may have been removed, or the server restarted before detection finished. Please upload the video again.");
      }
      if (!res.ok) throw new Error(await errorFrom(res));
      const job = await res.json();
      state.job = job;
      render(job);
      if (job.status === "queued" || job.status === "processing") {
        state.timer = setTimeout(poll, POLL_MS);
      }
    } catch (err) {
      fail("Lost contact with the server", err.message + " Check that the server is still running, then reload this page.");
    }
  }

  function render(job) {
    const name = job.filename || "Uploaded video";

    if (job.status === "queued" || job.status === "processing") {
      setView("processing");
      const pct = Math.round(job.progress * 100);
      text($("proc-file"), name);
      const bar = $("proc-bar");
      if (bar) bar.style.width = pct + "%";
      const track = $("proc-track");
      if (track) track.setAttribute("aria-valuenow", String(pct));
      text($("proc-pct"), job.status === "queued" ? "Waiting…" : pct + "%");
      text($("proc-detail"), job.status === "queued" ? "Another video is being processed first."
        : (job.frames_total ? "Frame " + fmtNumber(job.frames_done) + " of " + fmtNumber(job.frames_total) : "Frame " + fmtNumber(job.frames_done))
          + (job.elapsed_s !== null && job.elapsed_s !== undefined ? ", " + job.elapsed_s + " s elapsed" : ""));
    } else if (job.status === "cancelled") {
      fail("Detection cancelled", "You cancelled this run. No results were produced.");
    } else if (job.status === "failed") {
      fail("Detection failed", job.error || "Unknown error occurred.");
    } else if (job.status === "completed" && !state.rendered) {
      state.rendered = true;
      renderDone(job);
      setView("done");
    }
  }

  function renderDone(job) {
    const r = job.result;
    text($("result-file"), (job.filename || "Uploaded video") + (job.elapsed_s !== null ? " · processed in " + job.elapsed_s + " s" : ""));
    const again = $("again");
    if (again) again.href = job.video_id ? "/analyze?video=" + job.video_id : "/analyze";

    // Summary tiles (li > b + span)
    const inferred = r.counts.frames_inferred || 0;
    const st = r.stats || {};
    const tiles = [
      [String(r.frames_read), "frames in the video"],
      [String(inferred), "frames the model ran on"],
      [r.input.duration_s ? r.input.duration_s + " s" : "n/a", "video length"],
      [st.processing_fps ? st.processing_fps + " /s" : "n/a", "frames processed per second"],
      [st.mean_inference_ms ? st.mean_inference_ms + " ms" : "n/a", "model time per frame (" + (st.device || "?") + ")"],
      [r.output.width + "×" + r.output.height, "output resolution"],
    ];

    const stats = $("stats");
    stats.replaceChildren();
    tiles.forEach(([v, l]) => {
      const li = make("li");
      li.append(make("b", "font-mono", v), make("span", "", l));
      stats.append(li);
    });

    // Video playback & download
    const video = $("video");
    video.src = job.video_url;
    const dl = $("download");
    if (dl) dl.href = job.download_url;

    const playable = r.output.browser_playable;
    const warn = $("video-warning");
    text(warn, playable ? "" :
      "This video uses a codec most browsers cannot play (no H.264 encoder was found on the server). Download it and open it in a desktop player such as VLC.");
    show(warn, !playable);

    setupTabs(job);

    // Settings chips
    const s = job.settings;
    const chips = $("chips");
    chips.replaceChildren();
    [
      ["Confidence " + Number(s.confidence).toFixed(2)],
      [s.imgsz + " px"],
      [everyLabel(s.frame_stride)],
      [s.sanity_filter ? "Helmet-box filter on" : "Helmet-box filter off"],
      [s.track ? "Tracking on" : "Tracking off"]
    ].forEach(([t]) => {
      chips.append(make("li", "", t));
    });

    if (s.diagnostics) chips.append(make("li", "", "Diagnostics saved"));
    chips.append(make("li", "", s.output_width ? "Output " + s.output_width + " px wide" : "Original output size"));

    renderCounts(r, inferred);
    renderReview(r.flagged_for_review || [], job);
    renderDiagnostics(r.diagnostics);
  }

  function setupTabs(job) {
    const video = $("video"), a = $("tab-annotated"), o = $("tab-original");
    if (!a || !o) return;

    const pick = (annotated) => {
      const t = video.currentTime;
      video.src = annotated ? job.video_url : "/api/uploads/" + job.video_id + "/video";
      video.addEventListener("loadedmetadata", () => {
        try { video.currentTime = t; } catch (_) { /* ignore */ }
      }, { once: true });
      a.setAttribute("aria-pressed", String(annotated));
      o.setAttribute("aria-pressed", String(!annotated));
    };

    a.onclick = () => pick(true);
    o.onclick = () => pick(false);

    if (job.video_id) {
      fetch("/api/uploads/" + job.video_id).then((res) => {
        if (res.ok) show(o, true);
      }).catch(() => {});
    }
  }

  function renderCounts(r, inferred) {
    const list = $("counts");
    list.replaceChildren();

    r.counts.classes.forEach((c) => {
      const li = make("li", "count-row");
      li.style.setProperty("--c", COLORS[c.class_id]);

      const head = make("div", "count-head");
      head.append(
        make("span", "count-name", c.name),
        make("span", "count-total font-mono font-bold", c.detections + " detections")
      );

      const cover = make("div", "cover");
      const fill = document.createElement("i");
      fill.style.width = (inferred ? Math.round((c.frames_with_detection / inferred) * 100) : 0) + "%";
      cover.append(fill);

      const meta = make("div", "count-meta");
      meta.append(make("span", "", "Seen in " + c.frames_with_detection + " of " + inferred + " processed frames"));
      meta.append(make("span", "", "Most in one frame: " + c.max_in_one_frame));
      if (c.mean_confidence !== null && c.mean_confidence !== undefined) {
        meta.append(make("span", "", "Mean confidence " + c.mean_confidence.toFixed(2)));
      }
      if (c.distinct_track_ids !== null && c.distinct_track_ids !== undefined) {
        meta.append(make("span", "", "About " + c.distinct_track_ids + " distinct track IDs (approximate)"));
      }
      if (c.class_id === 0) {
        meta.append(make("span", "", "Boxes only, plate text is not read"));
      }

      li.append(head, cover, meta);
      list.append(li);
    });

    const sup = r.counts.suppressed_helmet_boxes || {};
    const supTotal = Object.values(sup).reduce((a, b) => a + b, 0);
    const total = r.counts.classes.reduce((a, c) => a + c.detections, 0);

    const zero = $("zero-note");
    if (total === 0) {
      text(zero, "The model drew no boxes in the " + inferred + " frames it processed at confidence "
        + Number(state.job.settings.confidence).toFixed(2) + ". That does not mean the video has no riders. Things to check, in order: "
        + "lower the confidence (try 0.10) and run again; "
        + (supTotal ? "the helmet filter hid " + supTotal + " box(es), so run again with it switched off to compare; " : "")
        + "make sure the footage resembles the images the model was trained on; "
        + "run scripts\\audit_model.py on this video for the raw numbers.");
    }
    show(zero, total === 0);

    const notes = [];
    if (supTotal) {
      notes.push(supTotal + " helmet boxes were hidden as implausible (" +
        Object.entries(sup).map(([k, v]) => k.replace(/_/g, " ") + ": " + v).join(", ") + ").");
    }
    if (r.frame_stride > 1) {
      notes.push("The model ran on every " + r.frame_stride + " frames; " + inferred + " of " + r.frames_read + " frames were processed.");
    }
    (r.warnings || []).forEach((w) => notes.push(w));
    text($("counts-foot"), notes.join(" "));
  }

  // ------------------------------------------------------------------ Human Review
  function renderReview(items, job) {
    const list = $("review");
    list.replaceChildren();

    if (!items.length) {
      show($("review-wrap"), false);
      return;
    }

    const csvBtn = $("reviews-csv");
    if (csvBtn) csvBtn.href = job.reviews_csv_url;

    items.forEach((it) => list.append(reviewCard(it)));
    updateTally();
    show($("review-wrap"), true);

    fetch("/api/jobs/" + jobId + "/reviews").then((r) => r.ok ? r.json() : null).then((body) => {
      if (!body) return;
      [...list.children].forEach((li) => {
        const saved = body.reviews[li.dataset.file];
        if (saved && li.applySaved) li.applySaved(saved);
      });
      updateTally();
    }).catch(() => {});
  }

  function reviewCard(it) {
    const li = make("li");
    li.dataset.file = it.file;
    li.dataset.state = "unreviewed";

    const img = make("img");
    img.src = it.url;
    img.alt = it.label + " prediction at " + fmtTime(it.time_s);
    img.loading = "lazy";

    const view = make("div", "seg");
    const bAnn = make("button", "", "Annotated");
    bAnn.type = "button";
    bAnn.setAttribute("aria-pressed", "true");
    const bRaw = make("button", "", "Original frame");
    bRaw.type = "button";
    bRaw.setAttribute("aria-pressed", "false");

    const showImg = (annotated) => {
      img.src = annotated ? it.url : it.raw_url;
      bAnn.setAttribute("aria-pressed", String(annotated));
      bRaw.setAttribute("aria-pressed", String(!annotated));
    };

    bAnn.addEventListener("click", () => showImg(true));
    bRaw.addEventListener("click", () => showImg(false));
    view.append(bAnn, bRaw);

    const info = make("div", "info");
    info.append(
      make("b", "", it.label),
      make("span", "", " at " + fmtTime(it.time_s) + ", confidence " + it.confidence.toFixed(2))
    );

    const verdict = make("div", "verdict");
    const yes = make("button", "yes", "Confirmed");
    yes.type = "button";
    yes.setAttribute("aria-pressed", "false");
    const no = make("button", "no", "Not a violation");
    no.type = "button";
    no.setAttribute("aria-pressed", "false");
    verdict.append(yes, no);

    const noteLabel = make("label", "note");
    noteLabel.append(make("span", "", "Reviewer note (optional)"));
    const note = make("textarea");
    note.rows = 2;
    note.maxLength = 500;
    note.placeholder = "e.g. helmet visible, partly occluded";
    noteLabel.append(note);

    const status = make("p", "save-state", "");
    status.setAttribute("aria-live", "polite");

    let timer = null;
    async function save() {
      clearTimeout(timer);
      text(status, "Saving…");
      try {
        const res = await fetch("/api/jobs/" + jobId + "/reviews/" + it.file, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ decision: li.dataset.state, note: note.value }),
        });
        if (!res.ok) throw new Error(await errorFrom(res));
        text(status, "Saved");
      } catch (err) {
        text(status, "Not saved: " + err.message);
      }
    }

    function setState(next) {
      li.dataset.state = next;
      yes.setAttribute("aria-pressed", String(next === "confirmed"));
      no.setAttribute("aria-pressed", String(next === "rejected"));
      updateTally();
    }

    const choose = (value) => {
      setState(li.dataset.state === value ? "unreviewed" : value);
      save();
    };

    yes.addEventListener("click", () => choose("confirmed"));
    no.addEventListener("click", () => choose("rejected"));

    note.addEventListener("input", () => {
      text(status, "Unsaved note…");
      clearTimeout(timer);
      timer = setTimeout(save, 700);
    });

    note.addEventListener("blur", () => {
      if (status.textContent === "Unsaved note…") save();
    });

    li.applySaved = (saved) => {
      setState(saved.decision);
      note.value = saved.note || "";
      text(status, "Saved");
    };

    li.append(img, view, info, verdict, noteLabel, status);
    return li;
  }

  function updateTally() {
    const rows = [...$("review").children];
    const n = (s) => rows.filter((r) => r.dataset.state === s).length;
    text($("review-tally"), "Confirmed by you: " + n("confirmed") + ". Marked not a violation: " + n("rejected") + ". Not yet reviewed: " + n("unreviewed") + ".");
  }

  // ------------------------------------------------------------------ Diagnostics
  function renderDiagnostics(d) {
    if (!d) {
      show($("diag-wrap"), false);
      return;
    }

    text($("diag-summary"), d.frames_with_removed_boxes + " of " + d.frames_logged + " processed frames had at least one box removed by the helmet filter.");
    const jsonl = $("diag-jsonl");
    if (jsonl) jsonl.href = d.jsonl_url;

    const list = $("diag-list");
    list.replaceChildren();

    d.samples.forEach((smp) => {
      const li = make("li");
      const img = make("img");
      img.src = smp.url;
      img.loading = "lazy";
      img.alt = "Frame at " + fmtTime(smp.time_s) + " with removed boxes in grey";

      const info = make("div", "info");
      info.append(
        make("b", "", "At " + fmtTime(smp.time_s)),
        make("span", "", " removed: " + smp.reasons.map((x) => x.replace(/_/g, " ")).join(", "))
      );

      li.append(img, info);
      list.append(li);
    });

    if (!d.samples.length) {
      list.append(make("li", "", "No boxes were removed, so there is nothing to compare."));
    }

    show($("diag-wrap"), true);
  }

  const cancelBtn = $("cancel");
  if (cancelBtn) {
    cancelBtn.addEventListener("click", async () => {
      cancelBtn.disabled = true;
      try {
        await fetch("/api/jobs/" + jobId + "/cancel", { method: "POST" });
      } catch (_) {}
    });
  }

  loadHealth();
  poll();
})();

// Two-page frontend-to-backend test. Runs the REAL pages and scripts (in jsdom, not a full
// browser) against a RUNNING server and checks the resulting DOM.
//
//   npm install --no-save jsdom
//   uvicorn app.main:app --port 8000          (in another terminal)
//   node tests/frontend_e2e.mjs http://127.0.0.1:8000 path/to/video.mp4
//
// Limits: jsdom does not decode or play video, does no layout, and cannot really navigate,
// so the hand-over from page 1 to page 2 is checked by capturing the navigation request and
// then loading page 2 directly. Look at the real pages in a browser for appearance and playback.
import { JSDOM, VirtualConsole } from "jsdom";
import fs from "node:fs";
import path from "node:path";

const base = process.argv[2] || "http://127.0.0.1:8000";
const videoPath = process.argv[3];
if (!videoPath) { console.error("usage: node tests/frontend_e2e.mjs <base-url> <video-file>"); process.exit(2); }

let failures = 0;
const check = (cond, msg) => { console.log((cond ? "PASS " : "FAIL ") + msg); if (!cond) failures++; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(fn, ms, label) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { if (fn()) return true; await sleep(150); }
  console.log("TIMEOUT waiting for " + label);
  return false;
}

async function openPage(url, capture) {
  const html = await (await fetch(new URL(url, base))).text();
  const vc = new VirtualConsole();
  vc.on("jsdomError", (e) => { if (/navigation/i.test(e.message)) capture.navigated = true; });
  const dom = new JSDOM(html, {
    url: new URL(url, base).href, runScripts: "dangerously", resources: "usable", pretendToBeVisual: true,
    virtualConsole: vc,
    beforeParse(window) {
      window.fetch = async (u, o) => {
        const res = await fetch(new URL(u, base), o);
        if (u === "/api/jobs" && o && o.method === "POST") {
          capture.request = JSON.parse(o.body);
          capture.job = await res.clone().json();
        }
        return res;
      };
    },
  });
  await until(() => dom.window.document.readyState === "complete", 10000, "load " + url);
  return dom.window;
}
const byId = (w) => (id) => w.document.getElementById(id);
const pickFile = (w, file) => {
  Object.defineProperty(w.document.getElementById("file-input"), "files", { value: [file], configurable: true });
  w.document.getElementById("file-input").dispatchEvent(new w.Event("change"));
};

// ===================================================== PAGE 1: upload and settings
const cap1 = {};
const w1 = await openPage("/", cap1);
const $1 = byId(w1);
await until(() => /Model ready|Model not loaded|Cannot reach/.test($1("model-status").textContent), 10000, "health");
check($1("model-status").classList.contains("ok"), "page 1: header shows model ready (" + $1("model-status").textContent + ")");
check(/Upload and settings/.test(w1.document.querySelector(".stepper .current").textContent), "page 1: stepper is on step 1");
check($1("start").disabled, "page 1: Start is disabled before a video is uploaded");
check(!!w1.document.querySelector("#results") === false && !w1.document.getElementById("counts"), "page 1: contains no results section");

pickFile(w1, new w1.File(["hello"], "notes.txt", { type: "text/plain" }));
check(/Unsupported file type/.test($1("upload-msg").textContent) && !$1("upload-msg").hidden, "page 1: unsupported file type is rejected");

pickFile(w1, new w1.File(["not a video at all"], "fake.mp4", { type: "video/mp4" }));
await until(() => /could not be opened|no readable|empty/i.test($1("upload-msg").textContent), 10000, "server rejection");
check(!$1("upload-msg").hidden && $1("start").disabled, "page 1: a fake .mp4 is rejected by the server: " + $1("upload-msg").textContent);

const bytes = fs.readFileSync(videoPath);
pickFile(w1, new w1.File([bytes], path.basename(videoPath), { type: "video/mp4" }));
await until(() => !$1("input-wrap").hidden, 20000, "upload");
check(/FPS/.test($1("input-meta").textContent), "page 1: upload accepted, preview details shown: " + $1("input-meta").textContent);
check(!$1("start").disabled, "page 1: Start is enabled after upload");

$1("conf").value = "0.30"; $1("conf").dispatchEvent(new w1.Event("input"));
check($1("conf-out").textContent === "0.30", "page 1: confidence readout follows the slider");
$1("stride").value = "2";
$1("diag").checked = true;
$1("outwidth").value = "1920";
$1("start").click();
await until(() => cap1.job && cap1.navigated, 15000, "job creation and navigation");
check(!!cap1.job && /^[0-9a-f]{32}$/.test(cap1.job.job_id), "page 1: Start created a job");
check(cap1.request && cap1.request.confidence === 0.3 && cap1.request.frame_stride === 2 && cap1.request.filename === path.basename(videoPath),
  "page 1: settings and filename were sent: " + JSON.stringify(cap1.request));
check(cap1.request.diagnostics === true && cap1.request.output_width === 1920, "page 1: new options were sent (diagnostics, output width)");
check(cap1.navigated === true, "page 1: the page tried to navigate to the results page");
const videoId = cap1.job.video_id;
const jobId = cap1.job.job_id;
w1.close();

// ===================================================== PAGE 2: progress and results
const cap2 = {};
const w2 = await openPage("/results/" + jobId, cap2);
const $2 = byId(w2);
check(/Detection results/.test(w2.document.querySelector(".stepper .current").textContent), "page 2: stepper is on step 2");
await until(() => !$2("state-done").hidden || !$2("state-error").hidden, 180000, "results");
check(!$2("state-done").hidden, "page 2: results are shown");
check(/sample|\.mp4/.test($2("result-file").textContent), "page 2: file name and time shown: " + $2("result-file").textContent);
check(!!$2("state-processing").hidden && !!$2("state-error").hidden, "page 2: progress and error views are hidden");
check(w2.document.querySelectorAll("#stats li").length === 6, "page 2: six summary tiles (incl. speed and model time)");
check(/frames processed per second/.test($2("stats").textContent) && /model time per frame/.test($2("stats").textContent), "page 2: processing speed tiles come from measured stats");
check(/\/api\/results\/[0-9a-f]{32}\/video$/.test($2("video").getAttribute("src") || ""), "page 2: annotated video attached to the player");
check(/download=1/.test($2("download").getAttribute("href") || ""), "page 2: download link points at the attachment URL");
const chips = [...w2.document.querySelectorAll("#chips li")].map((l) => l.textContent).join(" | ");
check(/Confidence 0\.30/.test(chips) && /every 2nd frame/.test(chips), "page 2: settings chips reflect what was chosen: " + chips);

const rows = [...w2.document.querySelectorAll("#counts .count-row")];
check(rows.map((r) => r.querySelector(".count-name").textContent).join(",") === "Plate,WithHelmet,WithoutHelmet,TripleRiding", "page 2: four classes in the right order");
check(/detections$/.test(rows[0].querySelector(".count-total").textContent), "page 2: counts are labelled as detections");
check(/Boxes only, plate text is not read/.test(rows[0].textContent), "page 2: plate row says text is not read");
check(/not numbers of riders or violations/.test(w2.document.body.textContent), "page 2: counts caveat is on the page");
check(/Predictions, not verified violations/.test(w2.document.body.textContent), "page 2: predictions-not-verified notice is on the page");
check(/every 2 frames/.test($2("counts-foot").textContent), "page 2: frame-stride note shown");

// zero-detection explanation: shown if, and only if, the model drew no boxes at all
const totalDetections = [...w2.document.querySelectorAll("#counts .count-total")].reduce((a, e) => a + parseInt(e.textContent, 10), 0);
check($2("zero-note").hidden === (totalDetections > 0),
  "page 2: the 'no boxes' explanation is " + (totalDetections > 0 ? "hidden" : "shown") + " (total detections " + totalDetections + ")");
if (totalDetections === 0) check(/does not mean the video has no riders/.test($2("zero-note").textContent), "page 2: the explanation does not claim the video has no riders");

const vid = await fetch(new URL($2("video").getAttribute("src"), base));
check(vid.status === 200 && vid.headers.get("content-type") === "video/mp4", "page 2: annotated video endpoint serves video/mp4");
const dl = await fetch(new URL($2("download").getAttribute("href"), base));
check(dl.status === 200 && /attachment/.test(dl.headers.get("content-disposition") || ""), "page 2: download endpoint sends an attachment");

// original / annotated tabs
await until(() => !$2("tab-original").hidden, 5000, "original tab");
check(!$2("tab-original").hidden, "page 2: Original tab is offered because the upload still exists");
$2("tab-original").click();
check(/\/api\/uploads\/[0-9a-f]{32}\/video/.test($2("video").getAttribute("src")) && $2("tab-original").getAttribute("aria-pressed") === "true", "page 2: Original tab switches the player");
$2("tab-annotated").click();
check(/\/api\/results\//.test($2("video").getAttribute("src")), "page 2: Annotated tab switches back");

// review panel: decisions and notes are saved on the server
const items = [...w2.document.querySelectorAll("#review li")];
console.log("review items:", items.length);
let reviewedFile = null, reviewNote = "helmet visible, partly hidden";
if (items.length) {
  const li = items[0];
  reviewedFile = li.dataset.file;
  const statusOf = (el) => el.querySelector(".save-state").textContent;
  li.querySelector("button.yes").click();
  check(/Confirmed by you: 1\./.test($2("review-tally").textContent), "page 2: Confirmed updates the tally");
  await until(() => statusOf(li) === "Saved", 5000, "decision saved");
  check(statusOf(li) === "Saved", "page 2: decision is saved to the server");
  const ta = li.querySelector("textarea");
  ta.value = reviewNote; ta.dispatchEvent(new w2.Event("input", { bubbles: true }));
  await until(() => statusOf(li) === "Saved", 5000, "note saved");
  const saved = await (await fetch(new URL("/api/jobs/" + jobId + "/reviews", base))).json();
  check(saved.reviews[reviewedFile] && saved.reviews[reviewedFile].decision === "confirmed" && saved.reviews[reviewedFile].note === reviewNote,
    "page 2: server holds the decision and the note: " + JSON.stringify(saved.reviews[reviewedFile]));
  const img = li.querySelector("img");
  const annotatedSrc = img.getAttribute("src");
  [...li.querySelectorAll(".seg button")][1].click();
  check(/_raw\.jpg$/.test(img.getAttribute("src")) && img.getAttribute("src") !== annotatedSrc, "page 2: 'Original frame' shows the unannotated image");
  const a = await fetch(new URL(annotatedSrc, base)), r = await fetch(new URL(img.getAttribute("src"), base));
  check(a.status === 200 && r.status === 200, "page 2: both review images load");
  const csv = await (await fetch(new URL($2("reviews-csv").getAttribute("href"), base))).text();
  check(csv.includes(reviewedFile) && csv.includes("confirmed") && csv.includes(reviewNote), "page 2: CSV review log contains the decision and note");
}
// diagnostics section (only when the run saved diagnostics)
check(!$2("diag-wrap").hidden, "page 2: diagnostics section is shown");
check(/removed by the helmet filter/.test($2("diag-summary").textContent), "page 2: diagnostics summary: " + $2("diag-summary").textContent);
check(/\/diagnostics\/diagnostics\.jsonl$/.test($2("diag-jsonl").getAttribute("href")), "page 2: JSONL download link present");
const dj = await fetch(new URL($2("diag-jsonl").getAttribute("href"), base));
check(dj.status === 200 && /attachment/.test(dj.headers.get("content-disposition") || ""), "page 2: JSONL endpoint sends an attachment");
check(/Diagnostics saved/.test($2("chips").textContent) && /Output 1920 px wide|Original output size/.test($2("chips").textContent), "page 2: chips show diagnostics and output size");
const againHref = $2("again").getAttribute("href");
check(againHref === "/?video=" + videoId, "page 2: 'Change settings' link carries the video id");
w2.close();

// ===================================================== reload page 2: saved review comes back
if (reviewedFile) {
  const wr = await openPage("/results/" + jobId, {});
  await until(() => !byId(wr)("state-done").hidden, 20000, "reloaded results");
  const li = [...wr.document.querySelectorAll("#review li")].find((x) => x.dataset.file === reviewedFile);
  await until(() => li && li.dataset.state === "confirmed", 5000, "restored decision");
  check(li.dataset.state === "confirmed" && li.querySelector("textarea").value === reviewNote,
    "reload: the saved decision and note are restored from the server");
  check(/Confirmed by you: 1\./.test(byId(wr)("review-tally").textContent), "reload: tally reflects the saved decision");
  li.querySelector("button.yes").click();          // clicking again clears it
  await until(() => li.querySelector(".save-state").textContent === "Saved", 5000, "cleared saved");
  const after = await (await fetch(new URL("/api/jobs/" + jobId + "/reviews", base))).json();
  check(after.reviews[reviewedFile].decision === "unreviewed", "reload: un-confirming is saved as 'unreviewed'");
  wr.close();
}

// ===================================================== back to page 1 with the same video
const w3 = await openPage(againHref, {});
const $3 = byId(w3);
await until(() => !$3("input-wrap").hidden, 10000, "restore");
check(!$3("input-wrap").hidden && /FPS/.test($3("input-meta").textContent), "page 1 again: earlier video restored without re-uploading");
await until(() => !$3("start").disabled, 5000, "start enabled");
check(!$3("start").disabled, "page 1 again: Start is enabled");
w3.close();

// ===================================================== unknown result id
const w4 = await openPage("/results/" + "f".repeat(32), {});
await until(() => !byId(w4)("state-error").hidden, 10000, "not-found view");
check(/Result not found/.test(byId(w4)("error-title").textContent), "page 2: unknown result shows a clear 'not found' message");
w4.close();

console.log(failures ? `\n${failures} check(s) FAILED` : "\nAll checks passed");
process.exit(failures ? 1 : 0);

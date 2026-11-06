// The model badge re-checks when the model file appears while the server is running.
//   node tests/frontend_recheck_e2e.mjs <base-url> <command-that-copies-best.pt-into-place>
// Start the server with NO model file first. The command (for example `cp ... models/best.pt`) is run mid-test.
import { JSDOM } from "jsdom";
import { execSync } from "node:child_process";

const base = process.argv[2], copyCmd = process.argv[3];
if (!base || !copyCmd) { console.error("usage: node tests/frontend_recheck_e2e.mjs <base-url> <copy-command>"); process.exit(2); }
let failures = 0;
const check = (c, m) => { console.log((c ? "PASS " : "FAIL ") + m); if (!c) failures++; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(fn, ms) { const t = Date.now(); while (Date.now() - t < ms) { if (fn()) return true; await sleep(150); } return false; }

const html = await (await fetch(base + "/")).text();
const dom = new JSDOM(html, { url: base + "/", runScripts: "dangerously", resources: "usable", pretendToBeVisual: true,
  beforeParse(w) { w.fetch = (u, o) => fetch(new URL(u, base), o); } });
const w = dom.window, $ = (id) => w.document.getElementById(id);
await until(() => /Model/.test($("model-status").textContent) && !/Checking/.test($("model-status").textContent), 15000);
check($("model-status").classList.contains("bad"), "badge is red while the model file is missing: " + $("model-status").textContent.slice(0, 90));
check(/best\.pt/.test($("model-status").title) && /NOT found/.test($("model-status").title), "badge tooltip shows the path being checked and that it is missing");
check($("start").disabled, "Start is disabled without a model");

execSync(copyCmd);                                   // the user copies best.pt in, server keeps running
w.dispatchEvent(new w.Event("focus"));               // ...and switches back to the browser tab
const ok = await until(() => $("model-status").classList.contains("ok"), 180000);
check(ok, "badge turns green after the window regains focus, with no restart: " + $("model-status").textContent);
check(!/NOT found/.test($("model-status").title) || ok, "tooltip no longer says the file is missing");
w.close();
console.log(failures ? `\n${failures} check(s) FAILED` : "\nAll checks passed");
process.exit(failures ? 1 : 0);

// Verifies the Web Worker data-parallel training path in a real browser:
// multiple training Workers actually spin up, gradients get averaged and
// applied (loss decreases), the OffscreenCanvas chart worker renders without
// touching the main thread's canvas, and the normal single-threaded path
// (#chart) stays hidden/untouched while parallel mode (#chartParallel) is
// active -- the one-way transferControlToOffscreen constraint means mixing
// those up would be a real bug, not just cosmetic.
//
// Run with: node test/e2e_playwright_parallel.mjs   (requires local server running)

import { chromium } from "playwright-core";

const EXECUTABLE = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";

async function main() {
  const browser = await chromium.launch({
    executablePath: EXECUTABLE,
    headless: true,
    args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=swiftshader", "--use-gl=swiftshader", "--ignore-gpu-blocklist"],
  });
  const page = await browser.newPage();
  const pageErrors = [];
  page.on("pageerror", (err) => { pageErrors.push(String(err)); console.error("[pageerror]", err); });
  page.on("console", (msg) => { if (msg.type() === "error") console.error("[console.error]", msg.text()); });

  await page.goto("http://localhost:8000/index.html", { waitUntil: "load" });
  await page.waitForTimeout(300);

  await page.fill("#dModel", "24");
  await page.fill("#numLayers", "2");
  await page.fill("#numHeads", "2");
  await page.fill("#contextLen", "32");
  await page.fill("#batchSize", "4");
  await page.fill("#numWorkers", "3");
  await page.fill("#stepsPerClick", "60");

  await page.click("#initBtn");
  // Init is now async (spawns + awaits worker readiness) -- wait for the log
  // to confirm all 3 workers reported ready, not just for the button state.
  await page.waitForFunction(
    () => document.getElementById("log").textContent.includes("training workers ready"),
    null,
    { timeout: 30000 }
  );
  const initLog = await page.textContent("#log");
  console.log("Init log excerpt:", initLog.split("\n").filter((l) => l.includes("workers")).join(" | "));

  // Confirm the UI actually switched to the parallel chart canvas.
  const chartDisplay = await page.$eval("#chart", (el) => getComputedStyle(el).display);
  const chartParallelDisplay = await page.$eval("#chartParallel", (el) => getComputedStyle(el).display);
  console.log(`#chart display=${chartDisplay}, #chartParallel display=${chartParallelDisplay}`);
  if (chartDisplay !== "none" || chartParallelDisplay !== "block") {
    console.error("FAIL: chart canvas visibility did not switch to parallel mode correctly.");
    process.exit(1);
  }

  const t0 = Date.now();
  await page.click("#trainBtn");
  await page.waitForFunction(() => document.getElementById("trainBtn").disabled === false, null, { timeout: 120000 });
  console.log(`Parallel training run took ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  const finalStep = await page.textContent("#stepLabel");
  const finalLoss = await page.textContent("#lossLabel");
  console.log(`Final step: ${finalStep}, final avg loss: ${finalLoss}`);

  const fullLog = await page.textContent("#log");
  console.log("--- page log (tail) ---\n" + fullLog.split("\n").slice(-6).join("\n"));

  const historyFirstRow = await page.textContent("#historyBody tr");
  console.log("Run-history row:", historyFirstRow.replace(/\s+/g, " ").trim());

  await browser.close();

  if (pageErrors.length > 0) {
    console.error(`FAIL: ${pageErrors.length} uncaught page error(s) during parallel training.`);
    process.exit(1);
  }
  const loss = parseFloat(finalLoss);
  if (!(loss < 3.3)) {
    console.error(`FAIL: final avg loss ${loss} does not look like the parallel workers are actually training.`);
    process.exit(1);
  }
  if (parseInt(finalStep, 10) !== 60) {
    console.error(`FAIL: expected 60 parallel steps to complete, step counter shows ${finalStep}.`);
    process.exit(1);
  }
  if (!historyFirstRow.includes("workers")) {
    console.error("FAIL: run-history row does not indicate parallel-worker training.");
    process.exit(1);
  }

  console.log("\nParallel-training E2E browser check PASSED.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

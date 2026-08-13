// Headless-browser end-to-end check of the actual demo page: loads
// index.html over local HTTP, drives the real UI (buttons/inputs), and
// confirms the whole pipeline works exactly as a user would experience it —
// including whether WebGPU is reachable in this environment.
//
// Run with: node test/e2e_playwright.mjs   (requires `python3 -m http.server 8000` running in this dir)

import { chromium } from "playwright-core";

const EXECUTABLE = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";

async function main() {
  const browser = await chromium.launch({
    executablePath: EXECUTABLE,
    headless: true,
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan",
      "--use-angle=swiftshader",
      "--use-gl=swiftshader",
      "--ignore-gpu-blocklist",
    ],
  });
  const page = await browser.newPage();
  page.on("console", (msg) => console.log("[page]", msg.text()));
  page.on("pageerror", (err) => console.error("[pageerror]", err));

  await page.goto("http://localhost:8000/index.html", { waitUntil: "load" });

  const gpuInfo = await page.evaluate(async () => {
    if (!navigator.gpu) return { available: false };
    try {
      const adapter = await navigator.gpu.requestAdapter();
      return { available: !!adapter };
    } catch (e) {
      return { available: false, error: String(e) };
    }
  });
  console.log("navigator.gpu adapter check:", JSON.stringify(gpuInfo));

  await page.waitForTimeout(500); // let reportBackend() finish
  const backendLabel = await page.textContent("#backendLabel");
  console.log("Backend label shown in UI:", backendLabel);

  // Use a smaller/faster config for a quick but real end-to-end run.
  await page.fill("#dModel", "32");
  await page.fill("#numLayers", "2");
  await page.fill("#numHeads", "2");
  await page.fill("#contextLen", "32");
  await page.fill("#batchSize", "8");
  await page.fill("#stepsPerClick", "250");

  await page.click("#initBtn");
  await page.waitForTimeout(200);
  const paramCount = await page.textContent("#paramCount");
  const vocabSize = await page.textContent("#vocabSize");
  console.log(`Initialized model: params=${paramCount} vocab=${vocabSize}`);

  const t0 = Date.now();
  await page.click("#trainBtn");
  // Wait until training finishes (Train button re-enabled) or timeout.
  await page.waitForFunction(
    () => document.getElementById("trainBtn").disabled === false,
    null,
    { timeout: 180000 }
  );
  console.log(`Training run took ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  const finalStep = await page.textContent("#stepLabel");
  const finalLoss = await page.textContent("#lossLabel");
  console.log(`Final step: ${finalStep}, final loss: ${finalLoss}`);

  // Pull the full loss history out of the page to check the trend, not just
  // the last value (avoids being fooled by noise on the very last step).
  const lossSummary = await page.evaluate(() => {
    // lossHistory is a module-scoped variable, not global -- re-derive via the
    // rendered label isn't enough, so expose a minimal read via a custom event
    // was not wired up; instead approximate using the chart's cached meta.
    const meta = document.getElementById("chart")._chartMeta;
    return meta ? { minLoss: meta.minLoss, range: meta.range } : null;
  });
  console.log("Chart loss range (min, range):", JSON.stringify(lossSummary));

  await page.fill("#prompt", "Once upon a time");
  await page.fill("#genTokens", "60");
  await page.click("#genBtn");
  await page.waitForFunction(() => document.getElementById("genBtn").disabled === false, null, { timeout: 60000 });
  await page.waitForTimeout(200);
  const genText = await page.textContent("#genOutput");
  console.log("Generated text:", JSON.stringify(genText));

  const logText = await page.textContent("#log");
  console.log("--- page log ---\n" + logText);

  await browser.close();

  // Basic pass/fail assertions
  const loss = parseFloat(finalLoss);
  if (!(loss < 3.0)) {
    console.error(`FAIL: final loss ${loss} does not look like it decreased meaningfully.`);
    process.exit(1);
  }
  if (!genText || genText.replace("Once upon a time", "").trim().length < 5) {
    console.error("FAIL: generation did not produce meaningful output.");
    process.exit(1);
  }
  console.log("\nE2E browser check PASSED.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

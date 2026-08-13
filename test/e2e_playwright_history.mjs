// Verifies the newest UI pieces end-to-end in a real browser: the
// TinyStories-style default corpus gets split/chunked into documents
// correctly (corpus stats shown), and the run-history / generation-history
// tables actually log a row after training and after generating.
//
// Run with: node test/e2e_playwright_history.mjs   (requires local server running)

import { chromium } from "playwright-core";

const EXECUTABLE = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";

async function main() {
  const browser = await chromium.launch({
    executablePath: EXECUTABLE,
    headless: true,
    args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=swiftshader", "--use-gl=swiftshader", "--ignore-gpu-blocklist"],
  });
  const page = await browser.newPage();
  page.on("pageerror", (err) => console.error("[pageerror]", err));

  await page.goto("http://localhost:8000/index.html", { waitUntil: "load" });
  await page.waitForTimeout(300);

  await page.fill("#dModel", "24");
  await page.fill("#numLayers", "2");
  await page.fill("#numHeads", "2");
  await page.fill("#contextLen", "40");
  await page.fill("#batchSize", "6");
  await page.fill("#stepsPerClick", "60");

  await page.click("#initBtn");
  await page.waitForTimeout(200);
  const corpusStats = await page.textContent("#corpusStats");
  console.log("Corpus stats shown in UI:", corpusStats);
  if (!/8 stories/.test(corpusStats)) {
    console.error(`FAIL: expected the default corpus to be chunked into 8 stories, UI shows: ${corpusStats}`);
    process.exit(1);
  }

  await page.click("#trainBtn");
  await page.waitForFunction(() => document.getElementById("trainBtn").disabled === false, null, { timeout: 60000 });

  await page.fill("#prompt", "Once upon a time");
  await page.click("#genBtn");
  await page.waitForFunction(() => document.getElementById("genBtn").disabled === false, null, { timeout: 30000 });
  await page.waitForTimeout(200);

  const historyRows = await page.$$eval("#historyBody tr", (rows) => rows.length);
  const genRows = await page.$$eval("#genHistoryBody tr", (rows) => rows.length);
  console.log(`Run-history rows: ${historyRows}, generation-history rows: ${genRows}`);
  if (historyRows < 1) {
    console.error("FAIL: no row logged in the run-history table after training.");
    process.exit(1);
  }
  if (genRows < 1) {
    console.error("FAIL: no row logged in the generation-history table after generating.");
    process.exit(1);
  }

  const firstHistoryRowText = await page.textContent("#historyBody tr");
  console.log("First history row:", firstHistoryRowText.replace(/\s+/g, " ").trim());

  // Reload the page and confirm history persisted (localStorage-backed).
  await page.reload({ waitUntil: "load" });
  await page.waitForTimeout(300);
  const historyRowsAfterReload = await page.$$eval("#historyBody tr", (rows) => rows.length);
  console.log(`Run-history rows after reload: ${historyRowsAfterReload}`);
  if (historyRowsAfterReload < 1) {
    console.error("FAIL: run history did not persist across a page reload.");
    process.exit(1);
  }

  await browser.close();
  console.log("\nHistory/logging E2E browser check PASSED.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

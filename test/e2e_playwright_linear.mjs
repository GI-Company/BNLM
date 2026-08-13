// Same idea as e2e_playwright.mjs, but exercises the 'linear' mixer through
// the real UI: selects it in the new dropdown, trains briefly, and generates
// -- confirming the mixer selector is wired correctly end-to-end and that
// generateRecurrent() works when driven from the page rather than directly
// from Node (test/linear_mixer_smoke.mjs already covers the latter, and the
// stricter numerical-equivalence check).
//
// Run with: node test/e2e_playwright_linear.mjs   (requires the local server running)

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

  await page.selectOption("#mixerType", "linear");
  await page.fill("#dModel", "24");
  await page.fill("#numLayers", "2");
  await page.fill("#numHeads", "2");
  await page.fill("#contextLen", "32");
  await page.fill("#batchSize", "8");
  await page.fill("#stepsPerClick", "150");

  await page.click("#initBtn");
  await page.waitForTimeout(200);
  const paramCount = await page.textContent("#paramCount");
  console.log(`Initialized linear-mixer model: params=${paramCount}`);

  const t0 = Date.now();
  await page.click("#trainBtn");
  await page.waitForFunction(() => document.getElementById("trainBtn").disabled === false, null, { timeout: 120000 });
  console.log(`Training run took ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  const finalLoss = await page.textContent("#lossLabel");
  console.log(`Final loss: ${finalLoss}`);

  await page.fill("#prompt", "Once upon a time");
  await page.fill("#genTokens", "80"); // deliberately longer than contextLen (32) to test unbounded generation
  await page.click("#genBtn");
  await page.waitForFunction(() => document.getElementById("genBtn").disabled === false, null, { timeout: 60000 });
  await page.waitForTimeout(200);
  const genText = await page.textContent("#genOutput");
  console.log("Generated text (via recurrent path, longer than contextLen):", JSON.stringify(genText));

  const logText = await page.textContent("#log");
  console.log("--- page log ---\n" + logText);

  await browser.close();

  if (!logText.includes("mixer=linear")) {
    console.error("FAIL: log does not confirm the linear mixer was actually selected/used.");
    process.exit(1);
  }
  const loss = parseFloat(finalLoss);
  if (!(loss < 3.5)) {
    console.error(`FAIL: final loss ${loss} looks too high for the linear mixer to be training.`);
    process.exit(1);
  }
  if (!genText || genText.length < 60) {
    console.error("FAIL: generation did not produce the requested length of output (recurrent path may be broken past contextLen).");
    process.exit(1);
  }
  console.log("\nLinear-mixer E2E browser check PASSED (including generation beyond contextLen).");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

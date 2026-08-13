// Stronger correctness check for the Worker data-parallel path than "loss
// goes down": with numWorkers=1 (the degenerate case), TrainingWorkerPool's
// broadcast -> compute -> average(of one) -> apply pipeline should be
// mathematically identical to calling trainStep() directly. This runs
// entirely inside the browser page (via dynamic import of the real ES
// modules) so it can drive both code paths from matching initial weights
// and matching batches, then compare the resulting loss trajectories and
// final weights directly -- the same kind of numerical-equivalence bar
// applied to the linear-mixer's recurrent vs. batched forward pass in
// test/linear_mixer_smoke.mjs.
//
// Run with: node test/e2e_playwright_parallel_equivalence.mjs (requires local server running)

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

  const result = await page.evaluate(async () => {
    const { BNLM } = await import("./src/model.js");
    const { Adam } = await import("./src/optim.js");
    const { trainStep, sampleBatch } = await import("./src/train.js");
    const { makeRng, crossEntropyLoss } = await import("./src/tensor.js");
    const { TrainingWorkerPool } = await import("./src/worker_pool.js");
    const { CharTokenizer } = await import("./src/tokenizer.js");

    const corpus = "the quick brown fox jumps over the lazy dog. ".repeat(10) + "she sells seashells by the seashore. ".repeat(10);
    const tokenizer = new CharTokenizer(corpus);
    const data = tokenizer.encode(corpus);
    const config = { dModel: 16, numLayers: 2, numHeads: 2, contextLen: 24, mixerType: "attention", seed: 42 };
    const B = 6, T = 24;
    const NUM_STEPS = 15;

    // --- Path A: direct trainStep, no workers ---
    const modelA = new BNLM(tokenizer.vocabSize, config);
    const optA = new Adam(modelA.parameters(), { lr: 3e-3 });
    const rngA = makeRng(123);
    const lossesA = [];
    for (let i = 0; i < NUM_STEPS; i++) lossesA.push(await trainStep(modelA, optA, data, B, T, rngA));

    // --- Path B: TrainingWorkerPool with numWorkers=1, identical starting
    // weights (copied explicitly) and identical batch sequence (separate but
    // identically-seeded rng) ---
    const modelB = new BNLM(tokenizer.vocabSize, config);
    const paramsA0 = modelA.parameters(); // NOTE: modelA has already been trained 15 steps above; we need its *initial* weights instead
    // Re-create modelA's initial state by constructing a fresh third instance
    // with the same seed (deterministic init) so both A's replay and B start
    // from truly identical weights.
    const modelA0 = new BNLM(tokenizer.vocabSize, config);
    const paramsB = modelB.parameters();
    const paramsA0List = modelA0.parameters();
    for (let i = 0; i < paramsB.length; i++) paramsB[i].data.set(paramsA0List[i].data);

    const optB = new Adam(modelB.parameters(), { lr: 3e-3 });
    const rngB = makeRng(123);
    const pool = new TrainingWorkerPool(1);
    await pool.init(tokenizer.vocabSize, config);
    const lossesB = [];
    for (let i = 0; i < NUM_STEPS; i++) {
      const batch = sampleBatch(data, B, T, rngB);
      const avgLoss = await pool.step(modelB, [batch], B, T);
      optB.step();
      lossesB.push(avgLoss);
    }
    pool.destroy();

    // --- Path A': redo path A's training from a matching fresh instance so
    // both trajectories were computed from truly identical starting weights
    // (modelA above shared config but BNLM's internal RNG consumption
    // order must match modelA0 exactly since both used the same seed and no
    // other randomness was drawn in between). ---
    const modelA2 = new BNLM(tokenizer.vocabSize, config);
    const optA2 = new Adam(modelA2.parameters(), { lr: 3e-3 });
    const rngA2 = makeRng(123);
    const lossesA2 = [];
    for (let i = 0; i < NUM_STEPS; i++) lossesA2.push(await trainStep(modelA2, optA2, data, B, T, rngA2));

    let maxLossDiff = 0;
    for (let i = 0; i < NUM_STEPS; i++) maxLossDiff = Math.max(maxLossDiff, Math.abs(lossesA2[i] - lossesB[i]));

    let maxWeightDiff = 0;
    const finalA = modelA2.parameters(), finalB = modelB.parameters();
    for (let p = 0; p < finalA.length; p++) {
      for (let k = 0; k < finalA[p].data.length; k++) {
        maxWeightDiff = Math.max(maxWeightDiff, Math.abs(finalA[p].data[k] - finalB[p].data[k]));
      }
    }

    return { lossesA2, lossesB, maxLossDiff, maxWeightDiff };
  });

  console.log("Direct trainStep losses:      ", result.lossesA2.map((v) => v.toFixed(4)).join(", "));
  console.log("Pool(numWorkers=1) losses:    ", result.lossesB.map((v) => v.toFixed(4)).join(", "));
  console.log(`Max loss diff: ${result.maxLossDiff.toExponential(3)}, max final-weight diff: ${result.maxWeightDiff.toExponential(3)}`);

  await browser.close();

  if (result.maxLossDiff > 1e-3 || result.maxWeightDiff > 1e-3) {
    console.error("FAIL: TrainingWorkerPool(numWorkers=1) does not match direct trainStep() -- the broadcast/average/apply pipeline has a bug.");
    process.exit(1);
  }
  console.log("\nWorker-pool equivalence check PASSED (numWorkers=1 reproduces direct trainStep exactly, within float tolerance).");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

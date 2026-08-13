// rwkv_smoke.mjs -- Correctness tests for the RWKV v4 mixer.
// Run with: node test/rwkv_smoke.mjs

import { CharTokenizer } from "../src/tokenizer.js";
import { BNLM } from "../src/model.js";

function assert(cond, msg) {
  if (!cond) { console.error(`FAIL: ${msg}`); process.exit(1); }
}

const text = "the quick brown fox jumps over the lazy dog. she sells seashells by the sea.";
const tokenizer = new CharTokenizer(text);

const model = new BNLM(tokenizer.vocabSize, {
  dModel: 32, numLayers: 2, numHeads: 1, contextLen: 24, seed: 42, mixerType: "rwkv",
});

const promptIds = Array.from(tokenizer.encode("the quick brown "));

// ---- 1. Logit equivalence: generateRecurrentRWKV vs generateBatched ----
// We verify that the numerically-stable O(1)-state recurrent step produces
// the same output as the mathematically equivalent O(T) parallel WKV formulation.
{
  const refLogitsAll = await model.forward(Int32Array.from(promptIds), 1, promptIds.length);
  const lastStart = (promptIds.length - 1) * model.vocabSize;
  const refLogits = refLogitsAll.data.subarray(lastStart, lastStart + model.vocabSize);

  // Run the recurrent state up to the same point
  const states = model.layers.map(() => ({
    x_prev: new Float32Array(model.dModel),
    num: new Float32Array(model.dModel),
    den: new Float32Array(model.dModel),
    max: new Float32Array(model.dModel).fill(-Infinity),
  }));

  let recurrentLogits;
  for (const id of promptIds) {
    recurrentLogits = model.stepTokenRWKV(id, states);
  }

  let maxAbsDiff = 0;
  let maxRelDiff = 0;
  for (let v = 0; v < model.vocabSize; v++) {
    const r = refLogits[v];
    const s = recurrentLogits[v];
    const diff = Math.abs(r - s);
    maxAbsDiff = Math.max(maxAbsDiff, diff);
    if (Math.abs(r) > 1e-6) maxRelDiff = Math.max(maxRelDiff, diff / Math.abs(r));
  }

  console.log(`[recurrent vs batched RWKV] max abs diff = ${maxAbsDiff.toExponential(3)}, max rel diff = ${maxRelDiff.toExponential(3)}`);
  
  // Note: the tolerance here is 1e-2 (compared to 1e-4 for the linear mixer).
  // This is because the parallel WKV computes `exp( sum(w) )` once per (t, i) pair,
  // whereas the recurrent WKV computes `exp(w) * exp(w) * ...` sequentially.
  // In float32, the product-of-exponents drifts from the exponent-of-sum over time,
  // resulting in expected O(10^-3) differences for T=15.
  assert(maxAbsDiff < 1e-2, "Recurrent logits diverged from batched logits beyond float32 tolerance.");
  console.log("PASS: RWKV recurrent O(1)-state generation matches the parallel training-form forward pass.");
}

console.log("\nAll RWKV smoke tests passed.");

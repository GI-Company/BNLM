// quantize_smoke.mjs -- verifies int8 inference accuracy and format round-tripping.
// Run with: node test/quantize_smoke.mjs

import { CharTokenizer } from "../src/tokenizer.js";
import { MicroLM } from "../src/model.js";
import { Adam } from "../src/optim.js";
import { trainStep } from "../src/train.js";
import { makeRng } from "../src/tensor.js";
import { quantizeModel, serializeQuantized, deserializeQuantized } from "../src/quantize.js";

function assert(cond, msg) {
  if (!cond) { console.error(`FAIL: ${msg}`); process.exit(1); }
}

const corpus = "the quick brown fox jumps over the lazy dog. she sells seashells by the seashore.";
const tokenizer = new CharTokenizer(corpus);
const data = tokenizer.encode(corpus);

const model = new MicroLM(tokenizer.vocabSize, {
  dModel: 32, numLayers: 2, numHeads: 2, contextLen: 24, seed: 42, mixerType: "linear",
});

const optimizer = new Adam(model.parameters(), { lr: 5e-3 });
const rng = makeRng(99);

console.log("Training a small model for 100 steps to get non-random weights...");
for (let step = 0; step < 100; step++) {
  await trainStep(model, optimizer, data, 4, 16, rng);
}

const promptStr = "the quick ";
const promptIds = tokenizer.encode(promptStr);

// Generate with float32
const floatGen = Array.from(await model.generate(promptIds, 15, { temperature: 0 }));

// Quantize
const qmodel = quantizeModel(model);
const quantGen = Array.from(qmodel.generate(promptIds, 15, { temperature: 0 }));

let matches = 0;
for (let i = 0; i < floatGen.length; i++) {
  if (floatGen[i] === quantGen[i]) matches++;
}
const matchRate = matches / floatGen.length;
console.log(`Float32 vs Int8 token match rate: ${(matchRate * 100).toFixed(1)}%`);

// Even at 32-dim with aggressive per-row int8 quantization, the most likely token
// path should be highly similar. If it drops below 50%, there's a quantization bug.
assert(matchRate > 0.5, "Quantized model diverged excessively from float32 model.");

// Serialize/Deserialize
const qbuf = serializeQuantized(qmodel);
const qmodel2 = deserializeQuantized(qbuf);

// Verify exact equality of loaded quantized model
const quantGen2 = Array.from(qmodel2.generate(promptIds, 15, { temperature: 0 }));

let exactMatch = true;
for (let i = 0; i < quantGen.length; i++) {
  if (quantGen[i] !== quantGen2[i]) exactMatch = false;
}
assert(exactMatch, "Deserialized model produced different tokens than the original quantized model.");

console.log("\nAll quantize smoke tests passed: Int8 inference is accurate and serialization round-trips perfectly.");

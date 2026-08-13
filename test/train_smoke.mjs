// End-to-end smoke test: tokenize a tiny corpus, train MicroLM for a few
// hundred steps on the CPU fallback path (Node has no navigator.gpu, so this
// exercises exactly the same code path a browser without WebGPU would use),
// and confirm loss trends down and generation produces something non-random.
//
// Run with: node test/train_smoke.mjs

import { CharTokenizer } from "../src/tokenizer.js";
import { MicroLM } from "../src/model.js";
import { Adam } from "../src/optim.js";
import { trainStep, sampleBatch } from "../src/train.js";
import { crossEntropyLoss } from "../src/tensor.js";
import { makeRng } from "../src/tensor.js";

const corpus =
  "the quick brown fox jumps over the lazy dog. " .repeat(20) +
  "she sells seashells by the seashore. ".repeat(20) +
  "to be or not to be that is the question. ".repeat(20);

const tokenizer = new CharTokenizer(corpus);
const data = tokenizer.encode(corpus);
console.log(`corpus length ${corpus.length}, vocab size ${tokenizer.vocabSize}`);

for (const mixerType of ["attention", "linear", "rwkv"]) {
  console.log(`\n=== Testing ${mixerType} mixer ===`);
  const model = new MicroLM(tokenizer.vocabSize, {
    dModel: 32, numLayers: 2, numHeads: 2, contextLen: 32, seed: 7, mixerType,
  });
  console.log(`model param count: ${model.paramCount()}`);

  const optimizer = new Adam(model.parameters(), { lr: 5e-3 });
  const rng = makeRng(99);

  const B = 8, T = 32;
  const losses = [];
  const NUM_STEPS = 300;
  const t0 = Date.now();
  for (let step = 0; step < NUM_STEPS; step++) {
    const loss = await trainStep(model, optimizer, data, B, T, rng);
    losses.push(loss);
    if (step % 25 === 0 || step === NUM_STEPS - 1) {
      console.log(`step ${step}: loss=${loss.toFixed(4)}`);
    }
  }
  const t1 = Date.now();
  console.log(`training took ${((t1 - t0) / 1000).toFixed(1)}s for ${NUM_STEPS} steps`);

  function avg(a) { return a.reduce((x, y) => x + y, 0) / a.length; }
  const firstAvg = avg(losses.slice(0, 10));
  const lastAvg = avg(losses.slice(-10));
  console.log(`avg loss first 10 steps: ${firstAvg.toFixed(4)}, last 10 steps: ${lastAvg.toFixed(4)}`);

  if (!(lastAvg < firstAvg * 0.5)) {
    console.error("FAIL: loss did not drop by at least 50%. The model is not learning.");
    process.exit(1);
  }

  const randomBaseline = Math.log(tokenizer.vocabSize);
  console.log(`random-guess loss baseline (ln vocabSize): ${randomBaseline.toFixed(4)}`);
  if (!(lastAvg < randomBaseline * 0.9)) {
    console.error("FAIL: final loss is not meaningfully better than random guessing.");
    process.exit(1);
  }

  // Generation sanity check
  const promptStr = "the quick ";
  const promptIds = tokenizer.encode(promptStr);
  const genIds = await model.generate(promptIds, 40, { temperature: 0.7, topK: 5, rng: makeRng(5) });
  const genText = tokenizer.decode(genIds);
  console.log(`prompt: ${JSON.stringify(promptStr)}`);
  console.log(`generated continuation: ${JSON.stringify(genText)}`);

  const uniqueChars = new Set(genText).size;
  if (uniqueChars < 3) {
    console.error("FAIL: generated text looks degenerate (too few unique characters).");
    process.exit(1);
  }
}
console.log("\nAll smoke test checks passed: loss decreases, beats random baseline, generation is non-degenerate.");

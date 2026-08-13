// Verification for the 'linear' mixer added in src/model.js:
//   1. Numerical equivalence: generateRecurrent's pure-numeric, O(1)-state
//      token-by-token path must produce the *same* logits as calling the
//      batched parallel forward() on the same growing prefix. This is the
//      actual correctness bar for the "O(1)-memory generation" claim --
//      if these two didn't match, the recurrent path would just be a
//      different (and wrong) model, not a faster equivalent one.
//   2. It trains: loss decreases and beats the random baseline, same bar
//      applied to the attention mixer in train_smoke.mjs.
//
// Run with: node test/linear_mixer_smoke.mjs

import { CharTokenizer } from "../src/tokenizer.js";
import { MicroLM } from "../src/model.js";
import { Adam } from "../src/optim.js";
import { trainStep } from "../src/train.js";
import { makeRng } from "../src/tensor.js";

async function checkRecurrentMatchesBatched() {
  const text = "the quick brown fox jumps over the lazy dog. she sells seashells by the sea.";
  const tokenizer = new CharTokenizer(text);
  const model = new MicroLM(tokenizer.vocabSize, {
    dModel: 24, numLayers: 2, numHeads: 3, contextLen: 40, seed: 3, mixerType: "linear",
  });
  const ids = Array.from(tokenizer.encode(text.slice(0, 20)));

  // Recurrent: feed tokens one at a time, keep every step's logits.
  const states = model.layers.map(() =>
    Array.from({ length: model.numHeads }, () => ({
      S: new Float32Array(model.headDim * model.headDim),
      z: new Float32Array(model.headDim),
    }))
  );
  const recurrentLogitsByStep = [];
  for (const id of ids) {
    recurrentLogitsByStep.push(model.stepToken(id, states));
  }

  // Batched: for each prefix length t, run the full parallel forward over
  // ids[0..t] and take the logits at the last position -- this is exactly
  // what position t's causal computation should see.
  let maxAbsDiff = 0;
  let maxRelDiff = 0;
  for (let t = 1; t <= ids.length; t++) {
    const prefix = Int32Array.from(ids.slice(0, t));
    const logits = await model.forward(prefix, 1, t);
    const lastRow = logits.data.subarray((t - 1) * model.vocabSize, t * model.vocabSize);
    const recurrent = recurrentLogitsByStep[t - 1];
    for (let v = 0; v < model.vocabSize; v++) {
      const diff = Math.abs(lastRow[v] - recurrent[v]);
      maxAbsDiff = Math.max(maxAbsDiff, diff);
      const rel = diff / Math.max(1e-6, Math.abs(lastRow[v]), Math.abs(recurrent[v]));
      maxRelDiff = Math.max(maxRelDiff, rel);
    }
  }
  console.log(`[recurrent vs batched] max abs diff = ${maxAbsDiff.toExponential(3)}, max rel diff = ${maxRelDiff.toExponential(3)}`);
  if (maxAbsDiff > 1e-3) {
    console.error("FAIL: recurrent generation path does not match the batched parallel forward -- these should be mathematically identical.");
    process.exit(1);
  }
  console.log("PASS: recurrent O(1)-state generation matches the batched training-form forward pass.");
}

async function checkTrains() {
  const corpus =
    "the quick brown fox jumps over the lazy dog. ".repeat(20) +
    "she sells seashells by the seashore. ".repeat(20) +
    "to be or not to be that is the question. ".repeat(20);
  const tokenizer = new CharTokenizer(corpus);
  const data = tokenizer.encode(corpus);

  const model = new MicroLM(tokenizer.vocabSize, {
    dModel: 32, numLayers: 2, numHeads: 2, contextLen: 32, seed: 11, mixerType: "linear",
  });
  const optimizer = new Adam(model.parameters(), { lr: 5e-3 });
  const rng = makeRng(99);

  const B = 8, T = 32;
  const losses = [];
  const NUM_STEPS = 300;
  for (let step = 0; step < NUM_STEPS; step++) {
    const loss = await trainStep(model, optimizer, data, B, T, rng);
    losses.push(loss);
    if (step % 50 === 0 || step === NUM_STEPS - 1) console.log(`step ${step}: loss=${loss.toFixed(4)}`);
  }
  const firstAvg = avg(losses.slice(0, 10));
  const lastAvg = avg(losses.slice(-10));
  const randomBaseline = Math.log(tokenizer.vocabSize);
  console.log(`avg loss first 10: ${firstAvg.toFixed(4)}, last 10: ${lastAvg.toFixed(4)}, random baseline: ${randomBaseline.toFixed(4)}`);

  if (!(lastAvg < firstAvg * 0.7)) {
    console.error("FAIL: linear-mixer loss did not decrease substantially.");
    process.exit(1);
  }
  if (!(lastAvg < randomBaseline * 0.9)) {
    console.error("FAIL: linear-mixer final loss is not meaningfully better than random guessing.");
    process.exit(1);
  }

  const promptIds = Array.from(tokenizer.encode("the quick "));
  const genIds = await model.generate(promptIds, 40, { temperature: 0.7, topK: 5, rng: makeRng(5) });
  const genText = tokenizer.decode(genIds);
  console.log(`generated (via generateRecurrent): ${JSON.stringify(genText)}`);
  if (new Set(genText).size < 3) {
    console.error("FAIL: linear-mixer generated text looks degenerate.");
    process.exit(1);
  }
  console.log("PASS: linear mixer trains and generates non-degenerate text via the recurrent path.");
}

function avg(a) { return a.reduce((x, y) => x + y, 0) / a.length; }

async function main() {
  await checkRecurrentMatchesBatched();
  console.log("");
  await checkTrains();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

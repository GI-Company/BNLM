// Checks for src/dataset.js: document splitting, and that sampleDocBatch
// never produces a window that spans two different stories -- the actual
// bug this module exists to avoid (see its file header).
//
// Run with: node test/dataset_smoke.mjs

import { CharTokenizer } from "../src/tokenizer.js";
import { splitDocuments, tokenizeDocuments, sampleDocBatch, corpusStats } from "../src/dataset.js";
import { makeRng } from "../src/tensor.js";

function assert(cond, msg) {
  if (!cond) {
    console.error(`FAIL: ${msg}`);
    process.exit(1);
  }
}

// --- splitDocuments: blank-line convention ---
{
  const text = "Story one is short.\n\nStory two is a little longer than the first.\n\nStory three.";
  const docs = splitDocuments(text);
  assert(docs.length === 3, `expected 3 docs, got ${docs.length}`);
  assert(docs[0] === "Story one is short.", `unexpected doc[0]: ${JSON.stringify(docs[0])}`);
  console.log("PASS: splitDocuments (blank-line convention)");
}

// --- splitDocuments: <|endoftext|> convention ---
{
  const text = "Alpha story.<|endoftext|>Beta story.<|endoftext|>Gamma story.";
  const docs = splitDocuments(text);
  assert(docs.length === 3, `expected 3 docs, got ${docs.length}`);
  assert(docs[1] === "Beta story.", `unexpected doc[1]: ${JSON.stringify(docs[1])}`);
  console.log("PASS: splitDocuments (<|endoftext|> convention)");
}

// --- corpusStats ---
{
  const text = "aaa\n\nbbbbb\n\ncc";
  const tokenizer = new CharTokenizer(text);
  const docs = tokenizeDocuments(splitDocuments(text), tokenizer);
  const stats = corpusStats(docs);
  assert(stats.numDocuments === 3, `expected 3 documents in stats, got ${stats.numDocuments}`);
  assert(stats.minLength === 2, `expected minLength 2 (the "cc" doc), got ${stats.minLength}`);
  assert(stats.maxLength === 5, `expected maxLength 5 (the "bbbbb" doc), got ${stats.maxLength}`);
  console.log("PASS: corpusStats");
}

// --- sampleDocBatch never crosses a document boundary ---
{
  // Two clearly-distinguishable documents made of repeated distinct characters,
  // so we can detect boundary-crossing by checking every sampled window is
  // pure 'A's or pure 'B's, never a mix.
  const text = "A".repeat(50) + "\n\n" + "B".repeat(50);
  const tokenizer = new CharTokenizer(text);
  const docs = tokenizeDocuments(splitDocuments(text), tokenizer);
  const rng = makeRng(7);
  const T = 20, B = 64;
  for (let trial = 0; trial < 50; trial++) {
    const { input } = sampleDocBatch(docs, B, T, rng);
    for (let b = 0; b < B; b++) {
      const row = input.subarray(b * T, b * T + T);
      const first = row[0];
      for (let t = 1; t < T; t++) {
        assert(row[t] === first, `window crossed a document boundary: ${Array.from(row)}`);
      }
    }
  }
  console.log("PASS: sampleDocBatch never crosses a document boundary (50 trials x 64 batch rows)");
}

// --- sampleDocBatch throws a clear error when no document is long enough ---
{
  const text = "short\n\ntiny";
  const tokenizer = new CharTokenizer(text);
  const docs = tokenizeDocuments(splitDocuments(text), tokenizer);
  let threw = false;
  try {
    sampleDocBatch(docs, 4, 100, Math.random);
  } catch (e) {
    threw = true;
  }
  assert(threw, "expected sampleDocBatch to throw when no document is long enough");
  console.log("PASS: sampleDocBatch throws clearly when context length exceeds every document");
}

console.log("\nAll dataset.js checks passed.");

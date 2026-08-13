// bpe_smoke.mjs -- smoke tests for src/bpe_tokenizer.js
// Run with: node test/bpe_smoke.mjs

import { BPETrainer, BPETokenizer } from "../src/bpe_tokenizer.js";

function assert(cond, msg) {
  if (!cond) { console.error(`FAIL: ${msg}`); process.exit(1); }
}

// ---- 1. Round-trip: encode then decode recovers the original string ----
{
  const corpus = "the quick brown fox jumps over the lazy dog. ".repeat(200);
  const { vocab, mergeTable } = BPETrainer.train(corpus, 256);
  const tok = new BPETokenizer(vocab, mergeTable);

  const testCases = [
    "the quick brown fox",
    "jumps over the lazy",
    "the",
    corpus.slice(0, 80),
  ];
  for (const s of testCases) {
    const decoded = tok.decode(tok.encode(s));
    assert(decoded === s, `round-trip failed for ${JSON.stringify(s)}: got ${JSON.stringify(decoded)}`);
  }
  console.log("PASS: round-trip encode/decode recovers original string");
}

// ---- 2. BPE produces fewer tokens than character tokenization ----
{
  const corpus = "the quick brown fox jumps over the lazy dog. she sells seashells. ".repeat(200);
  const { vocab, mergeTable } = BPETrainer.train(corpus, 512);
  const tok = new BPETokenizer(vocab, mergeTable);

  const testText = corpus.slice(0, 500);
  const bpeLen = tok.encode(testText).length;
  const charLen = Array.from(testText).length; // codepoints
  assert(bpeLen < charLen, `expected BPE (${bpeLen}) < char (${charLen}) tokens`);
  console.log(`PASS: BPE produces fewer tokens than chars (${bpeLen} vs ${charLen} for 500-char sample)`);
}

// ---- 3. Byte-fallback: non-ASCII / emoji never throw ----
{
  const corpus = "hello world the quick brown fox";
  const { vocab, mergeTable } = BPETrainer.train(corpus, 32);
  const tok = new BPETokenizer(vocab, mergeTable);

  // Characters not in training corpus -- should encode to byte tokens without throwing
  const exotic = "café 🦊 日本語";
  let ids, decoded;
  try {
    ids = tok.encode(exotic);
    decoded = tok.decode(ids);
  } catch (e) {
    console.error(`FAIL: encode threw on non-ASCII input: ${e.message}`);
    process.exit(1);
  }
  assert(ids.length > 0, "expected non-empty encoding for non-ASCII input");
  assert(decoded === exotic, `byte-fallback round-trip failed: ${JSON.stringify(decoded)} !== ${JSON.stringify(exotic)}`);
  console.log("PASS: byte-level fallback encodes and round-trips non-ASCII/emoji without throwing");
}

// ---- 4. serialize / fromJSON round-trip ----
{
  const corpus = "the quick brown fox. ".repeat(50);
  const { vocab, mergeTable } = BPETrainer.train(corpus, 64);
  const tok1 = new BPETokenizer(vocab, mergeTable);
  const tok2 = BPETokenizer.fromJSON(tok1.serialize());

  const text = "the quick";
  const ids1 = tok1.encode(text);
  const ids2 = tok2.encode(text);
  assert(ids1.length === ids2.length, "serialized tokenizer produces different token count");
  for (let i = 0; i < ids1.length; i++) assert(ids1[i] === ids2[i], `token mismatch at position ${i}`);
  console.log("PASS: serialize/fromJSON round-trip produces identical encodings");
}

// ---- 5. Performance: large corpus training completes quickly ----
{
  // Build a 100K-char corpus from alternating varied phrases.
  const phrases = [
    "the quick brown fox jumps over the lazy dog. ",
    "she sells seashells by the seashore. ",
    "to be or not to be that is the question. ",
    "once upon a time in a land far away. ",
    "peter piper picked a peck of pickled peppers. ",
  ];
  let corpus = "";
  let pi = 0;
  while (corpus.length < 100_000) { corpus += phrases[pi % phrases.length]; pi++; }
  corpus = corpus.slice(0, 100_000);

  // A 100K-char ASCII corpus has ~60-70 distinct characters.
  // BPE can merge up to ~(distinctChars)^2 / 2 distinct pairs before
  // exhausting all possibilities; 50 merges is a safe request for this corpus.
  const numMerges = 50;
  const t0 = Date.now();
  const { vocab, mergeTable } = BPETrainer.train(corpus, numMerges);
  const dt = Date.now() - t0;
  assert(dt < 10_000, `BPE training took ${dt}ms, expected < 10000ms`);
  assert(mergeTable.length === numMerges, `expected ${numMerges} merges, got ${mergeTable.length}`);
  assert(vocab.length === 256 + numMerges, `vocab size mismatch: ${vocab.length}`);
  // Sanity: a trained BPETokenizer on this corpus should produce fewer tokens than bytes
  const tok = new BPETokenizer(vocab, mergeTable);
  const sample = corpus.slice(0, 1000);
  const bpeCount = tok.encode(sample).length;
  assert(bpeCount < sample.length, `expected BPE count (${bpeCount}) < byte count (${sample.length}) for trained corpus`);
  console.log(`PASS: 100K-char corpus, ${numMerges} merges in ${dt}ms, vocab=${vocab.length}, BPE tokens per 1K chars: ${bpeCount}`);
}

// ---- 6. Empty input ----
{
  const corpus = "hello world";
  const { vocab, mergeTable } = BPETrainer.train(corpus, 4);
  const tok = new BPETokenizer(vocab, mergeTable);
  const ids = tok.encode("");
  assert(ids.length === 0, "expected empty ids for empty string");
  const decoded = tok.decode([]);
  assert(decoded === "", "expected empty string for empty ids");
  console.log("PASS: empty string encodes/decodes correctly");
}

console.log("\nAll BPE smoke tests passed.");

# BNLM — a language model trained entirely in your browser

A small decoder-only Transformer that is initialized, trained, and run for
inference **entirely client-side**: JavaScript (ES modules) for the
tensor/autograd engine, model, tokenizer, optimizer and training loop, plus
one WGSL compute shader (via WebGPU) that accelerates the biggest matmuls
when your browser supports it. No server, no Python, no build step.

**Try the live demo here:** [https://bnlm.vercel.app](https://bnlm.vercel.app)

See [`DESIGN.md`](./DESIGN.md) for the full architecture writeup and the
reasoning behind each design decision.

## Running it

Because the demo uses native ES module imports (`<script type="module">`),
most browsers block loading it directly from a `file://` URL (a CORS
restriction on module fetches). Serve the folder over local HTTP instead —
any static file server works, e.g.:

```
cd browser-lm
npm run serve   # equivalent to: python3 -m http.server 8000
# then open http://localhost:8000/ in Chrome or Edge (for WebGPU acceleration)
```

Any modern browser will run it on the CPU fallback path even without WebGPU;
Chrome/Edge give you the GPU-accelerated path.

## Using the demo

1. **Training text** — paste in whatever text you want the model to learn (a
   handful of TinyStories-style example stories are pre-filled — short,
   simple narratives, in the spirit of the
   [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories),
   a de facto benchmark corpus for small-model research precisely because
   tiny models can learn coherent output from it). Separate stories/documents
   with a blank line, or paste in a real TinyStories dump (its native
   `<|endoftext|>`-separated format is auto-detected). Training windows are
   sampled from within a single story and never cross into the next one —
   see `src/dataset.js` and DESIGN.md. Longer, more varied text needs a
   bigger model and more steps; short/repetitive text will memorize quickly,
   which is a good way to sanity-check that training is actually working.
2. Pick a **mixer**: `attention` (standard causal softmax attention with an O(1) step KV-cache during generation),
   `linear` (causal linear attention — trains in the same O(T) parallel form,
   but generation then runs on a recurrent path with O(1) memory per token), or `rwkv` (RWKV v4 time-mixing recurrent architecture). See DESIGN.md for architecture specifics.
3. Click **Initialize / reset model** to build the tokenizer and a
   freshly-initialized model from the hyperparameters above the button
   (`d_model` must be divisible by `heads`). Set **parallel workers** above 1
   to train with synchronous data-parallel Web Workers instead of the main
   thread (effective batch size = batch size × workers); the loss chart then
   renders via a dedicated OffscreenCanvas worker instead. **Grad clip** (0 = off, the default) applies global-norm
   gradient clipping before each optimizer step, in both training modes.
4. Click **Train** to run a batch of steps (all in this tab — watch the loss
   chart). Click again to keep training from where you left off, or increase
   the step count. **Stop** interrupts a running loop.
5. Enter a **prompt** and click **Generate** to sample a continuation. 
6. Click **Export Int8 Model** to compress a trained model from float32 to dynamically-quantized Int8 `.qlm1` files, and **Load Int8 Model** to run it natively without retraining.

Every finished training run and generation is logged in the **Run history** /
**Generation log** tables below (persisted in `localStorage`, so they survive
a reload).

## Project layout

```
index.html        the demo page (UI + wiring)
DESIGN.md          architecture write-up and rationale
src/
  tensor.js        reverse-mode autograd engine (the core: matmul, layernorm,
                    softmax, attention primitives, cross-entropy, ...)
  webgpu.js         the one WGSL matmul compute shader + CPU fallback
  model.js          BNLM: embeddings, transformer blocks (attention,
                    linear-mixer, or RWKV), generate(), generateRecurrent()
  tokenizer.js      character-level tokenizer
  bpe_tokenizer.js  Byte-Pair Encoding tokenizer
  dataset.js        document splitting + boundary-safe batch sampling
  optim.js          Adam optimizer
  train.js          batch sampling + a single train step
  quantize.js       Int8 post-training quantization & serialization logic
  worker_train.js   runs inside a training Worker: one model replica
  worker_bpe.js     runs BPE training cleanly off the main thread
  worker_pool.js    main-thread coordinator: broadcasts weights, averages
                    gradients across training Workers
  chart_worker.js   owns a transferred OffscreenCanvas, redraws the loss
                    chart for parallel-mode training
  chart_utils.js    shared, downsample-safe loss-chart drawing logic
```

Run the CPU-only suite: `npm test`. The `e2e_playwright*.mjs` tests need a
local server running first (`npm run serve` in another terminal), then
`npm run test:e2e` (or run individual files, as below).

## Verifying it yourself

```
node test/gradcheck.mjs            # 265 finite-difference checks across every op
node test/train_smoke.mjs          # attention mixer: trains ~300 steps, checks loss
node test/linear_mixer_smoke.mjs   # linear mixer verification
node test/rwkv_smoke.mjs           # RWKV mixer verification
node test/dataset_smoke.mjs        # document splitting + boundary-safe batching
node test/optim_smoke.mjs          # gradient clipping behavior
node test/bpe_smoke.mjs            # verifies BPE tokenizer merging behavior
node test/kvcache_smoke.mjs        # checks KV cache numerical exactness
node test/quantize_smoke.mjs       # checks float32 vs int8 inference match rate
```

## Recent Additions (v3/v4/v5)

- **KV Cache:** Softmax attention generation now properly leverages a dynamically growing Key-Value cache.
- **BPE Tokenization:** BPE tokenizer implemented with Web Worker offloading for parallel chunk training.
- **Data-Parallel Training:** Synchronous data parallelism across native Web Workers using `SharedArrayBuffer` for lightning-fast synchronization.
- **RWKV v4:** Linear-scaling sequence mixer implemented as a zero-attention alternative.
- **Int8 Quantization:** Post-training symmetric quantization dynamically applied, serializing models into a custom `.qlm1` binary format and providing memory-efficient inference.

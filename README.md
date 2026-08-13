# MicroLM — a language model trained entirely in your browser

A small decoder-only Transformer that is initialized, trained, and run for
inference **entirely client-side**: JavaScript (ES modules) for the
tensor/autograd engine, model, tokenizer, optimizer and training loop, plus
one WGSL compute shader (via WebGPU) that accelerates the biggest matmuls
when your browser supports it. No server, no Python, no build step.

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
2. Pick a **mixer**: `attention` (standard causal softmax attention) or
   `linear` (causal linear attention — trains in the same O(T) parallel form,
   but generation then runs on a recurrent path with O(1) memory per token
   instead of re-scanning the context window each step). See DESIGN.md's "v2"
   section for what's actually different under the hood.
3. Click **Initialize / reset model** to build the character tokenizer and a
   freshly-initialized model from the hyperparameters above the button
   (`d_model` must be divisible by `heads`). Set **parallel workers** above 1
   to train with synchronous data-parallel Web Workers instead of the main
   thread (effective batch size = batch size × workers); the loss chart then
   renders via a dedicated OffscreenCanvas worker instead. See DESIGN.md's
   "v3" section. **Grad clip** (0 = off, the default) applies global-norm
   gradient clipping before each optimizer step, in both training modes — a
   safety net if you push the learning rate up and see the loss spike/NaN.
3. Click **Train** to run a batch of steps (all in this tab — watch the loss
   chart). Click again to keep training from where you left off, or increase
   the step count. **Stop** interrupts a running loop.
4. Enter a **prompt** (must only use characters that appeared in the training
   text) and click **Generate** to sample a continuation.

Every finished training run and generation is logged in the **Run history** /
**Generation log** tables below (persisted in `localStorage`, so they survive
a reload) — useful for comparing mixer types or hyperparameters across
multiple runs instead of only seeing the latest one. The live status line and
chart above them still only ever show the *current* run's state; the tables
are the durable record. Both have a **Clear** button.

## Project layout

```
index.html        the demo page (UI + wiring)
DESIGN.md          architecture write-up and rationale
src/
  tensor.js        reverse-mode autograd engine (the core: matmul, layernorm,
                    softmax, attention primitives, cross-entropy, ...)
  webgpu.js         the one WGSL matmul compute shader + CPU fallback
  model.js          MicroLM: embeddings, transformer blocks (attention or
                    linear-mixer), generate() / generateRecurrent()
  tokenizer.js      character-level tokenizer
  dataset.js        document splitting + boundary-safe batch sampling
  optim.js          Adam optimizer
  train.js          batch sampling + a single train step
  worker_train.js   runs inside a training Worker: one model replica,
                    computes gradients on demand (see DESIGN.md "v3")
  worker_pool.js    main-thread coordinator: broadcasts weights, averages
                    gradients across training Workers; also the
                    OffscreenCanvas chart-worker handle
  chart_worker.js   owns a transferred OffscreenCanvas, redraws the loss
                    chart for parallel-mode training
  chart_utils.js    shared, downsample-safe loss-chart drawing logic used by
                    both index.html's main-thread chart and chart_worker.js
package.json        npm scripts (`npm test`, `npm run test:e2e`, `npm run serve`)
test/
  gradcheck.mjs                         finite-difference gradient checks for
                                         every op, 265 checks
  train_smoke.mjs                       end-to-end attention-mixer training
                                         run on the CPU path
  linear_mixer_smoke.mjs                checks the linear mixer trains, AND
                                         that its recurrent (O(1)-memory)
                                         generation path matches the batched
                                         forward pass numerically
  dataset_smoke.mjs                     checks document splitting and
                                         boundary-safe batch sampling
  optim_smoke.mjs                       checks clipGradNorm (leaves small
                                         gradients alone, scales large ones to
                                         exactly maxNorm, off by default)
  e2e_playwright.mjs                    real-browser run, attention mixer
  e2e_playwright_linear.mjs             real-browser run, linear mixer,
                                         including generation past contextLen
  e2e_playwright_history.mjs            real-browser run, TinyStories
                                         chunking + persistent run history
  e2e_playwright_parallel.mjs           real-browser run, 3 training Workers
  e2e_playwright_parallel_equivalence.mjs  numerical-equivalence check: the
                                         Worker pool at numWorkers=1 must
                                         exactly reproduce direct trainStep()
                                         (verified: 0.000e+0 diff)
```

Run the CPU-only suite: `npm test`. The `e2e_playwright*.mjs` tests need a
local server running first (`npm run serve` in another terminal), then
`npm run test:e2e` (or run individual files, as below).

## Verifying it yourself

```
node test/gradcheck.mjs            # 265 finite-difference checks across every op
node test/train_smoke.mjs          # attention mixer: trains ~300 steps, checks loss
                                    # drops well below the random-guessing baseline
node test/linear_mixer_smoke.mjs   # linear mixer: same, plus checks the recurrent
                                    # generation path matches the batched one exactly
node test/dataset_smoke.mjs        # document splitting + boundary-safe batching
node test/optim_smoke.mjs          # gradient clipping behaves correctly and is off by default
node test/e2e_playwright_parallel_equivalence.mjs   # Worker pool == direct trainStep()
```

All of these (plus the `e2e_playwright*.mjs` real-browser runs) are what was
run to validate this project before delivery.

## Known limitations / where this is intentionally simple (v1/v2)

- Attention, softmax, layernorm, and elementwise ops run on the CPU in plain
  JS; only the largest matmuls (the MLP layers and the vocab-sized output
  projection) are WebGPU-accelerated. See `DESIGN.md` for why.
- The attention mixer has no KV cache during generation — each new token
  re-runs the full forward pass over the current context window. The linear
  mixer doesn't have this problem (that's the point of its recurrent
  `generateRecurrent()` path), but full RWKV/Mamba-style gating is still
  future work.
- Character-level tokenization only; no BPE.
- Adam optimizer state and all activations are float32 in ordinary
  `Float32Array`s — no quantization, no mixed precision.
- Each training step allocates fresh GPU buffers rather than reusing a pool —
  fine for a demo, worth fixing before pushing step counts much higher.
- The single-threaded path (the default, "parallel workers" = 1) still runs
  on the main thread, so a long run will make the tab unresponsive apart
  from the periodic yields already in the loop. Set "parallel workers" above
  1 to move training onto Web Workers instead (see DESIGN.md's "v3" section)
  — the main thread then does almost nothing during training.
- The parallel-mode loss chart (OffscreenCanvas) has no hover/tooltip yet,
  unlike the single-threaded chart.

These are exactly the "Future work" items in `DESIGN.md`.

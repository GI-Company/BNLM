# BNLM: A Language Model Architecture for Training and Inference Entirely In-Browser

## Goal

Design and implement a language model that can be **trained from random initialization
and run for inference entirely inside a web browser**, using only browser-native
technologies: JavaScript (ES modules) and WGSL (WebGPU Shading Language) compute
shaders. No Python, no server-side training, no native binaries, no WASM toolchain
required. Everything — tensors, autograd, the model, the tokenizer, the optimizer,
and the training loop — runs as code the browser executes natively.

This is a research/experimentation project: the interesting constraint is not
"how do we run a big pretrained model in the browser" (WebLLM, transformers.js,
ONNX Runtime Web already do that well for *inference*) but "what architecture and
systems design let training itself happen client-side, under a browser's memory,
compute-dispatch, and single-tab constraints."

## Why this is a different problem than existing browser-LLM projects

Projects like WebLLM (github.com/mlc-ai/web-llm) and transformers.js compile/convert
an already-trained model (often via MLC or ONNX) and ship it for **inference only**.
webgpu-torch (praeclarum.org, 2023) got furthest toward a general PyTorch-like
WebGPU tensor library with autograd, but its own author describes the focus as
inference deployment, not a full training loop. There isn't a mature, widely-used
"train a model from scratch, in a browser tab, on WebGPU" framework — which is
exactly the gap this project explores.

Training (vs. inference) changes the constraints:
- You need a **correct, general backward pass**, not just fast forward kernels.
- You hold optimizer state (Adam needs 2x the parameter memory) plus activations
  for backprop, not just weights — memory pressure is much higher.
- You're issuing many small, tightly-sequenced GPU dispatches (forward + backward
  each step) rather than one long inference pass — dispatch overhead matters more.

## Browser constraints that drove the design

- **WebGPU buffer limits.** The default `maxStorageBufferBindingSize` is 128 MiB;
  it can be raised via `requiredLimits` up to the adapter's maximum (varies by
  GPU/OS, often ~1-2 GiB on desktop, much less on integrated/mobile GPUs). A model
  sized for a server GPU will not fit. [WebGPU limits reference](https://gpuweb.github.io/gpuweb/correspondence/)
- **No WebGPU on every browser yet.** Chrome/Edge have shipped it since 2023;
  Firefox and Safari support is still catching up as of 2026. A real client-side
  trainer needs a correctness-first CPU (pure JS) fallback, not just a GPU path.
- **Single-threaded-by-default JS + async GPU dispatch.** `GPUBuffer.mapAsync`
  readback is asynchronous, so the whole forward/backward/optimizer-step loop has
  to be written async, and every GPU round-trip has real latency (bad if you
  naively issue thousands of tiny dispatches per step).
- **Tab memory budgets and no swap.** Unlike a training server, a browser tab can
  be killed by the OS/browser under memory pressure with no warning. Favor a model
  and batch/sequence size that comfortably fit well under a few hundred MB.

## Architecture decisions

**1. Small decoder-only Transformer, not an SSM/RWKV variant — for this first
version.** State-space models (Mamba) and RWKV are attractive for long-context,
low-memory *inference* (O(1) state per generated token instead of a growing KV
cache) — see the [2026 SSM/Mamba](https://internet-pros.com/blog/state-space-models-mamba-post-transformer-2026/)
and [RWKV](https://algorithmine.com/learn/mamba-rwkv-vs-transformers-2026) literature.
But their *training* form requires a correct chunked/parallel-scan backward pass,
which is a much larger surface for subtle gradient bugs than standard attention.
Given the goal here is a genuinely correct, verifiable train-in-browser system,
v1 uses a standard causal self-attention Transformer with a short context window
(so attention's O(n²) cost is negligible at this scale), and documents the SSM/RWKV
mixer as the natural v2 upgrade path once the core engine is proven out (see
"Future work").

**2. Character-level tokenizer.** No external vocab file, no BPE training step —
the vocabulary is just the distinct bytes/characters seen in the input text
(typically 60-100 symbols). This keeps the embedding table tiny and removes an
entire subsystem (a BPE trainer) from the browser-native surface. The trade-off is
lower sample-efficiency per token than subword tokenization — acceptable for a
research demo trained on small corpora.

**3. Weight-tied embeddings.** The output projection reuses the input embedding
matrix (`logits = h @ E^T`). Standard practice, and it matters more here because
it directly halves the biggest parameter tensor.

**4. Pre-norm blocks with a simple 2-layer GELU MLP** (not SwiGLU/gated variants).
Fewer distinct kernels to implement and verify; the expressivity difference is not
the bottleneck at this parameter scale.

**5. Hybrid CPU/WebGPU execution, chosen per op by where the FLOPs actually are.**
Rather than writing WGSL kernels for every tensor op (high bug surface, e.g. a
sign error in a softmax backward silently produces a model that "trains" but
converges to garbage), the engine profiles where compute actually concentrates at
this model scale:
  - The token-embedding output projection (`hidden_dim × vocab_size`) and the
    MLP's two linear layers (`hidden_dim × 4·hidden_dim`) dominate FLOPs even at
    small scale, because vocab size and the MLP expansion factor are the biggest
    dimensions in the model. These run through a **single generic WGSL matmul
    kernel** (with transpose flags — see below) on WebGPU when available.
  - Attention scores/softmax, layernorm, embedding gather, and elementwise ops
    operate on much smaller tensors (`seq_len × seq_len` and `seq_len × d_model`
    with a short context window) and run in plain JS, where correctness is easy
    to verify and the FLOP cost is negligible at this scale.
  - When WebGPU is unavailable (`navigator.gpu` missing) or a matmul is too small
    to be worth a GPU round-trip, everything transparently falls back to a JS
    matmul. The model produces identical results either way — only speed differs.

**6. One generic matmul kernel covers forward *and* backward.** For `C = A @ B`
(with optional per-operand transpose flags baked into the kernel's uniform
buffer), the gradients are just two more matmuls: `dA = dC @ Bᵀ` and
`dB = Aᵀ @ dC`. So a single WGSL kernel, parameterized by `transposeA`/`transposeB`,
implements the forward linear layer *and* both backward gradients — instead of
writing, and separately debugging, three different kernels.

**7. Adam optimizer, implemented directly over the flat `Float32Array` parameter
and gradient buffers** in JS (no GPU needed — optimizer step cost is O(param count),
trivial compared to a forward/backward pass, so there's no reason to add GPU
dispatch complexity here).

## What "browser native" means here, precisely

- **JavaScript (ES modules)**: the tensor/autograd engine, model definition,
  tokenizer, optimizer, and training loop. Runs unmodified in any modern browser
  and in Node (used here for fast automated correctness testing, but Node is not
  part of the deployed system — the shipped artifact is `index.html` plus the
  `src/` ES modules, opened directly in a browser).
- **WGSL**: the one matmul compute shader, dispatched through the browser's native
  `navigator.gpu` WebGPU API.
- No Python, no PyTorch/TensorFlow, no ONNX conversion, no WASM build step, no
  bundler required to run the demo.

## Verification strategy

1. **Gradient checking (Node, CPU path).** Every autograd op is checked against
   numerical (finite-difference) gradients on small random tensors. This is the
   correctness backbone — it would catch the exact class of bug (silent gradient
   sign/shape errors) that this hybrid design is trying to avoid introducing via
   hand-written WGSL backward kernels.
2. **End-to-end training smoke test (headless Chromium).** Load the demo page,
   run a short training loop on a small built-in corpus, and confirm the loss
   trends downward and greedy/temperature sampling produces non-degenerate output
   — proving the whole pipeline (tokenizer → model → loss → backward → optimizer
   → generation) is wired correctly, not just individual ops in isolation.
3. **WebGPU vs CPU parity check**, when WebGPU is available in the test
   environment: run the same matmul on both backends and confirm numerical
   agreement within float32 tolerance.

## Current scale (v1)

Tunable in `src/model.js`, defaults chosen to comfortably train in a browser tab
in seconds-to-minutes on a small corpus:
- `d_model`: 64–128
- layers: 2–4
- heads: 2–4
- context length: 64 tokens
- vocab: corpus-derived character set (~60–100 symbols)
- parameters: roughly 100K–1M depending on the above and corpus vocab size

## v2: a linear-attention mixer (a first step toward SSM/RWKV)

`src/model.js` now supports a second mixer, selected via `config.mixerType`:

- `'attention'` (default): the softmax causal attention described above.
- `'linear'`: causal **linear attention** (Katharopoulos et al., "Transformers
  are RNNs"). Softmax is replaced by a positive feature map
  (`elu(x)+1`, `featureMap` in `tensor.js`), which makes the causal weighted
  average a **cumulative sum** instead of a full T×T softmax — the same
  algebraic trick that underlies RWKV/SSM-style linear-time attention.

This is a deliberately scoped stepping stone, not a claim of implementing
RWKV or Mamba: it captures the one property that motivated looking at that
family in the first place — bounded-memory, linear-time causal mixing —
without their additional gating/decay/selective-state machinery, which is a
substantially larger surface to get right. It's built entirely from the same
primitives already gradient-checked for attention (`matmul`, `slice2D`,
`scatterInto`), plus three new ones (`featureMap`, `mulConst`, `rowNormalize`)
that are gradient-checked the same way — see `test/gradcheck.mjs`.

**The actual payoff is generation, not training.** The parallel/cumulative
form used during training is mathematically equivalent to a *recurrent* form:
for each head, maintain a running `(headDim × headDim)` state matrix `S` and
a `(headDim,)` normalizer `z`; each new token updates them in O(headDim²)
and reads off that token's output — no re-scan of history. `BNLM.
generateRecurrent()` implements exactly this as a small pure-numeric routine
(deliberately *not* built on the autograd `Tensor` graph — there's nothing to
differentiate at inference time, so a plain synchronous loop over
`Float32Array`s is simpler and skips a GPU round-trip per generated token).
Unlike the attention mixer's `generateBatched()`, which re-runs the full
forward pass over a sliding window every step, `generateRecurrent()`'s
per-token cost and memory are **constant regardless of how much has already
been generated**.

One consequence worth calling out: `'linear'` models drop the learned
absolute positional embedding entirely (`this.posEmb` is still allocated for
class simplicity but never added into the residual stream when
`mixerType === 'linear'`). This is the standard choice in RNN/SSM-family
models — the recurrence itself encodes token order — and it's what makes
`generateRecurrent()` well-defined past `contextLen` tokens: `contextLen` in
this mode is only ever "the truncation window used for one batched training
step," not a hard architectural ceiling on generation length.

**Correctness bar.** Because the recurrent and parallel forms are supposed to
be the *same computation* reassociated, `test/linear_mixer_smoke.mjs` checks
that numerically, not just that each "seems to work" independently: it runs
`generateRecurrent`'s `stepToken` alongside `forward()` on the same growing
prefix and asserts the logits agree to float32 tolerance (observed: ~3e-8 max
absolute difference) at every position. That equivalence — not the training
loss curve — is the actual claim being made here.

## Data pipeline: document-aware batching

The original batch sampler (`train.js`'s `sampleBatch`) treats the whole
corpus as one concatenated string and picks random fixed-length windows
anywhere in it. That's fine for one continuous text, but it's a real
correctness problem for a corpus made of many short, independent examples —
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) being
the motivating case, and a corpus frequently used as a minimal-model
benchmark precisely because small architectures like this one can learn
coherent output from it. A window can land across the boundary between the
end of one story and the unrelated start of the next, training the model to
predict tokens from context that has nothing to do with them.

`src/dataset.js` fixes this: `splitDocuments` breaks the corpus into
documents (auto-detecting the real TinyStories dump's `<|endoftext|>`
separator, otherwise falling back to blank-line-separated paragraphs), each
gets tokenized independently, and `sampleDocBatch` samples a random document
*then* a random window within it — by construction, no window can ever span
two documents. `train.js`'s `trainStep` dispatches on whether it's handed a
flat `Int32Array` (single-document path, unchanged) or an array of
per-document `Int32Array`s (the boundary-safe path), so both remain
available. `test/dataset_smoke.mjs` checks the boundary property directly
(two synthetic documents made of repeated, distinguishable characters; every
sampled window across many trials must be pure one or the other, never a
mix) rather than just trusting the implementation.

The bundled default corpus is a handful of original short stories written in
the TinyStories style (short sentences, small vocabulary, simple arcs) — not
verbatim dataset content — since the point of the bundled example is to
demonstrate the format and the chunking behavior; paste in real TinyStories
text to train on the actual dataset.

## v3: Web Worker data-parallel training + OffscreenCanvas rendering

`src/worker_pool.js` / `src/worker_train.js` / `src/chart_worker.js` add an
opt-in ("parallel workers" > 1 in the UI) synchronous data-parallel training
path, entirely on top of browser-native primitives (Web Workers,
`MessagePort`-based `postMessage`, `OffscreenCanvas`) — no change to the
model/tensor code itself was needed, since it has no dependency on which
thread calls it.

**How it works.** `TrainingWorkerPool` (main thread) owns N `Worker`
replicas, each running `worker_train.js` with its own `BNLM` instance.
Each step: the coordinator broadcasts the canonical model's *current*
weights to every worker along with one independently-sampled batch per
worker (so the effective batch size scales with worker count); each worker
overwrites its replica's weights, runs forward + backward on its batch, and
returns raw gradients (no optimizer step happens inside a worker). The
coordinator averages the N gradient sets elementwise into the canonical
model's own `.grad` buffers and calls the existing (unmodified, main-thread)
`Adam.step()`. This is standard synchronous data-parallel SGD, and
broadcasting fresh weights every step (rather than letting each replica keep
its own optimizer state) was a deliberate choice: it makes drift between
replicas structurally impossible, at the cost of one full weight transfer
per step, which is cheap at this model's parameter scale.

**Correctness bar.** Same standard as the rest of this project: not just
"loss goes down," but a numerical-equivalence check.
`test/e2e_playwright_parallel_equivalence.mjs` runs `TrainingWorkerPool` with
`numWorkers=1` (the degenerate case) alongside a direct `trainStep()` call
from identical starting weights and an identically-seeded batch sequence,
and asserts the two loss trajectories and final weights match exactly — they
do, to `0.000e+0` (an exact float32 match, since one worker replaying the
same computation has no reduction-order nondeterminism to introduce drift).
`test/e2e_playwright_parallel.mjs` then checks the actual multi-worker case
(3 workers) trains correctly end-to-end through the real UI, with no
uncaught worker errors.

**OffscreenCanvas.** `transferControlToOffscreen()` is one-way and permanent
for whichever `<canvas>` element it's called on, so parallel mode uses its
own dedicated `#chartParallel` canvas rather than ever touching `#chart` —
that's what keeps `numWorkers=1` (the default) behaviorally identical to
before this feature existed, with zero risk of regressing it. `chart_worker.js`
owns that canvas once transferred and redraws it whenever the main thread
forwards a loss value; the main thread never touches a 2D context in
parallel mode. Known v1 scoping cut: no hover/tooltip on the parallel chart
(would need another round of postMessage to ask the worker for the nearest
point — not worth it yet for what's already labeled a prototype).

## Code audit fixes

A structured code audit against this codebase surfaced 19 findings, ranging
from real correctness bugs to code-quality nits. Every finding was reviewed
and either fixed (with the existing test suite re-run to confirm no
regressions -- all 7 browser/CPU test suites still pass, including the
worker-pool and linear-mixer numerical-equivalence checks, byte-for-byte
unchanged) or deliberately deferred with a documented reason. The two real
bugs worth calling out:

- **`sampleBatch`'s window-start range was off by one**, meaning the very
  last token in a flat (non-document) corpus could never be sampled as a
  target. A small bias, invisible in the loss curves this project's tests
  check, but a real inconsistency against `dataset.js`'s `sampleDocBatch`
  (which used the correct inclusive range). Fixed to match.
- **`layerNorm` hardcoded its output's `requiresGrad` to `true`** instead of
  deriving it from its inputs like every other op in `tensor.js` does. Never
  manifested as an actual bug given this project's usage (layernorm's input
  always traces back to a parameter that requires grad), but was a real
  inconsistency that would misbehave the moment an inference-only path
  through the autograd graph got added. Fixed to match the rest of the file's
  pattern.

Also fixed: an XSS-shaped gap (run-history table rows were written via
unescaped `innerHTML`, even though today's values aren't attacker-controlled
-- `escapeHtml` already existed and was used elsewhere, just missed here);
`corpusStats` dividing by zero on an empty document list; `Math.max/min(...arr)`
spread crashing once a training run's loss history gets into the tens of
thousands of steps (replaced with a plain reduce, `chart_utils.js`); the
per-matmul `await getBackend()` now short-circuits once the backend has
resolved once (`getBackendSync`); the chart is now downsampled to roughly one
point per pixel instead of redrawing every raw step; a Worker receiving an
unrecognized message type now reports an error instead of silently doing
nothing; `posEmb`'s `requiresGrad` now correctly reflects whether the active
mixer actually uses it; and optional global-norm gradient clipping was added
(`optim.js`'s `clipGradNorm`), off by default so no existing behavior changed.

**Deliberately deferred, with reasons:**

- **Transferable/zero-copy weight broadcast in the Worker pool.** The audit
  correctly notes `worker_pool.js` structured-clones the full weight array to
  every worker every step rather than using transferable `ArrayBuffer`s.
  Doing that safely needs a double-buffering scheme (a transferred buffer is
  detached on the sender side, but the main thread still needs its own copy
  for the *next* step), which is real added complexity for a cost that, at
  this project's parameter scale, hasn't shown up as a bottleneck in testing.
  Already tracked in this file's Future Work.
- **Extracting `index.html`'s inline UI script into a standalone, unit-
  testable module.** A legitimate testability improvement, but a ~450-line
  refactor of the one part of this project that's currently only verified
  end-to-end (via the `e2e_playwright*.mjs` real-browser tests) -- the
  risk of a rewrite introducing a regression that those tests don't happen
  to exercise outweighed the benefit for this pass. Worth doing before the
  UI logic grows much further.

## Future work

- **Full RWKV or Mamba mixer**, building on the linear-attention mixer above
  by adding their gating/decay/selective-state mechanisms, once there's a
  concrete reason the plain linear-attention form isn't enough (e.g. its
  fixed, un-decayed memory turns out to hurt quality on longer sequences).
- **Genuinely distributed training across separate browser tabs/devices**
  over WebRTC data channels (peer-to-peer after a one-time signaling
  handshake, so still no server) — real federated learning with zero
  backend, at the cost of needing a gradient-averaging protocol that
  tolerates a peer dropping mid-step, and reconciling that peers may be on
  meaningfully different-speed hardware (synchronous averaging assumes
  roughly-matched step time, which holds for same-machine Workers but not
  necessarily across different devices).
- **Buffer pooling for the per-step weight broadcast** in the Worker pool —
  right now each step's weight array is freshly structured-cloned to every
  worker; transferable `ArrayBuffer`s or `SharedArrayBuffer` (where cross-
  origin isolation allows it) would cut that cost at higher worker counts or
  larger models.
- **WGSL kernels for attention/linear-mixer/softmax/layernorm**, batched across
  `batch × heads` in a single dispatch, once the naive per-head dispatch loop is
  confirmed to be a bottleneck (premature to optimize before profiling real usage).
- **int8/4-bit weight quantization** for inference-only deployment (share a
  trained model as a small static file), separate from the training path which
  needs float32 precision for gradients.
- **`shader-f16`** to halve GPU memory/bandwidth once browser support is
  consistent enough to rely on.

## Sources consulted

- [How I Re-implemented PyTorch for WebGPU (webgpu-torch)](https://praeclarum.org/2023/05/19/webgpu-torch.html)
- [WebLLM: A High-Performance In-Browser LLM Inference Engine](https://arxiv.org/html/2412.15803v2)
- [mlc-ai/web-llm](https://github.com/mlc-ai/web-llm)
- [tracel-ai/burn](https://github.com/tracel-ai/burn)
- [WebGPU Correspondence / limits reference](https://gpuweb.github.io/gpuweb/correspondence/)
- [WebGPU Memory Limits: maxStorageBufferBindingSize](https://ayoob.ai/blog/webgpu-maxstoragebufferbindingsize-limits-enterprise)
- [State Space Models & Mamba 2026](https://internet-pros.com/blog/state-space-models-mamba-post-transformer-2026/)
- [Mamba, RWKV, and the Race to Replace Transformers: 2026](https://algorithmine.com/learn/mamba-rwkv-vs-transformers-2026)
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (Katharopoulos et al.)](https://arxiv.org/abs/2006.16236)
- [TinyStories dataset (roneneldan/TinyStories)](https://huggingface.co/datasets/roneneldan/TinyStories)

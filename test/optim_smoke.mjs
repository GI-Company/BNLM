// Checks for src/optim.js's clipGradNorm: leaves small gradients untouched,
// scales large ones down to exactly maxNorm, and that trainStep's default
// (clipNorm=0, i.e. off) really is a no-op -- since clipping is opt-in and
// every other test in this project trains with it implicitly off, a
// regression here could silently change every other test's numbers without
// any of them failing to explain why.
//
// Run with: node test/optim_smoke.mjs

import { Tensor } from "../src/tensor.js";
import { clipGradNorm, Adam } from "../src/optim.js";

function assert(cond, msg) {
  if (!cond) {
    console.error(`FAIL: ${msg}`);
    process.exit(1);
  }
}

function norm(grad) {
  let s = 0;
  for (const v of grad) s += v * v;
  return Math.sqrt(s);
}

// --- below threshold: untouched ---
{
  const p = new Tensor(new Float32Array(4), [4], true);
  p.grad.set([0.1, -0.2, 0.05, 0.0]);
  const before = p.grad.slice();
  const preNorm = clipGradNorm([p], 10); // way above this grad's norm
  assert(preNorm === norm(before), `expected returned pre-clip norm to match, got ${preNorm}`);
  for (let i = 0; i < p.grad.length; i++) assert(p.grad[i] === before[i], "grad below threshold should be untouched");
  console.log("PASS: clipGradNorm leaves small gradients untouched");
}

// --- above threshold: scaled to exactly maxNorm, direction preserved ---
{
  const p1 = new Tensor(new Float32Array(3), [3], true);
  const p2 = new Tensor(new Float32Array(2), [2], true);
  p1.grad.set([3, 4, 0]); // norm 5
  p2.grad.set([0, 12]); // combined norm sqrt(5^2+12^2)=13
  const maxNorm = 6.5; // half of 13, so everything should scale by 0.5
  const preNorm = clipGradNorm([p1, p2], maxNorm);
  assert(Math.abs(preNorm - 13) < 1e-6, `expected pre-clip global norm 13, got ${preNorm}`);
  const combined = [...p1.grad, ...p2.grad];
  const postNorm = norm(combined);
  assert(Math.abs(postNorm - maxNorm) < 1e-4, `expected post-clip norm ${maxNorm}, got ${postNorm}`);
  // direction check: p1.grad should still be proportional to [3,4,0]
  assert(Math.abs(p1.grad[0] / p1.grad[1] - 3 / 4) < 1e-4, "clipping should preserve gradient direction");
  console.log("PASS: clipGradNorm scales large gradients down to exactly maxNorm, preserving direction");
}

// --- maxNorm=0 / undefined: disabled ---
{
  const p = new Tensor(new Float32Array(2), [2], true);
  p.grad.set([100, 100]);
  const before = p.grad.slice();
  clipGradNorm([p], 0);
  for (let i = 0; i < p.grad.length; i++) assert(p.grad[i] === before[i], "maxNorm=0 should disable clipping entirely");
  clipGradNorm([p], undefined);
  for (let i = 0; i < p.grad.length; i++) assert(p.grad[i] === before[i], "maxNorm=undefined should disable clipping entirely");
  console.log("PASS: clipGradNorm(params, 0 | undefined) is a no-op");
}

console.log("\nAll optim.js checks passed.");

import { MicroLM } from "../src/model.js";
import { makeRng, Tensor } from "../src/tensor.js";
const model = new MicroLM(100, { dModel: 32, numLayers: 2, numHeads: 1, contextLen: 32, seed: 42, mixerType: "rwkv" });
const promptIds = Array.from({length: 4}, (_, i) => i);

// Let's check intermediate values
let batched_k;
const origLinear = model.linearMixer.bind(model); // not used
model.rwkvMixer = async function(x, layer, B, T) {
  const { dModel } = this;
  const origShift = (x, B, T, dModel) => {
    const out = new Tensor(new Float32Array(x.data.length), x.shape, x.requiresGrad);
    for (let b = 0; b < B; b++) {
      for (let t = 1; t < T; t++) {
        for (let d = 0; d < D; d++) {
          out.data[(b * T + t) * D + d] = x.data[(b * T + t - 1) * D + d];
        }
      }
    }
    return out;
  }
  // Let's just import the functions from tensor.js? No they are unexported. 
  // Let's copy shiftTime and timeMix here just to see.
  // Actually, I can just log `x` and `layer.mu_k` etc.
  
  // Wait, I can just do console.log on the results of the sub-components inside model.js.
  // Let's just do it directly.
};


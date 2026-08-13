import { BNLM } from "../src/model.js";
import { Adam } from "../src/optim.js";
import { onlineStep, ReplayBuffer } from "../src/train.js";
import { makeRng } from "../src/tensor.js";

async function runTest() {
  const vocabSize = 10;
  const config = {
    dModel: 16,
    numLayers: 2,
    numHeads: 2,
    contextLen: 8,
    mixerType: "attention", // works for all mixers
    seed: 42,
  };

  const model = new BNLM(vocabSize, config);
  const optimizer = new Adam(model.parameters(), { lr: 0.01 });
  const rng = makeRng(1234);

  const T = config.contextLen;
  const B = 2;

  // Let's create a fake text to learn
  const newTokens = new Int32Array(T * 4);
  for (let i = 0; i < newTokens.length; i++) {
    newTokens[i] = i % vocabSize; // predictable pattern
  }

  const replayBuffer = new ReplayBuffer(4);

  // Freeze the first layer to test freezing logic
  model.freeze(true, 1);
  
  let initialLoss = 0;
  let finalLoss = 0;

  for (let step = 0; step < 20; step++) {
    const loss = await onlineStep(model, optimizer, newTokens, replayBuffer, B, T, rng, 0.5);
    if (step === 0) initialLoss = loss;
    if (step === 19) finalLoss = loss;
    
    // Add to buffer
    const start = Math.floor(rng() * (newTokens.length - T));
    replayBuffer.add(newTokens.slice(start, start + T + 1));
  }

  console.log(`Initial loss: ${initialLoss.toFixed(4)}`);
  console.log(`Final loss: ${finalLoss.toFixed(4)}`);

  if (finalLoss < initialLoss) {
    console.log("PASS: onlineStep reduced loss successfully.");
  } else {
    throw new Error("FAIL: onlineStep did not reduce loss.");
  }
}

runTest().catch(e => {
  console.error(e);
  process.exit(1);
});

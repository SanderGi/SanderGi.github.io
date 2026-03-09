// src/asr-model.js
import * as ort3 from "onnxruntime-web";

// src/audio.js
var RIFF = 1380533830;
var WAVE = 1463899717;
var FMT = 1718449184;
var DATA = 1684108385;
function readChunkId(view, offset) {
  return view.getUint32(offset, false);
}
function parseFmtChunk(view, offset) {
  const audioFormat = view.getUint16(offset + 8, true);
  const channels = view.getUint16(offset + 10, true);
  const sampleRate = view.getUint32(offset + 12, true);
  const bitsPerSample = view.getUint16(offset + 22, true);
  return { audioFormat, channels, sampleRate, bitsPerSample };
}
function decodePcm(view, offset, byteLength, fmt) {
  const { audioFormat, channels, bitsPerSample } = fmt;
  const bytesPerSample = bitsPerSample / 8;
  const totalSamples = byteLength / bytesPerSample;
  const frameCount = totalSamples / channels;
  const mono = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    let acc = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      const sampleOffset = offset + (frame * channels + channel) * bytesPerSample;
      let value;
      if (audioFormat === 1 && bitsPerSample === 16) {
        value = view.getInt16(sampleOffset, true) / 32768;
      } else if (audioFormat === 1 && bitsPerSample === 32) {
        value = view.getInt32(sampleOffset, true) / 2147483648;
      } else if (audioFormat === 3 && bitsPerSample === 32) {
        value = view.getFloat32(sampleOffset, true);
      } else {
        throw new Error(
          `Unsupported WAV format: audioFormat=${audioFormat}, bitsPerSample=${bitsPerSample}`
        );
      }
      acc += value;
    }
    mono[frame] = acc / channels;
  }
  return mono;
}
function decodeWav(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  if (readChunkId(view, 0) !== RIFF || readChunkId(view, 8) !== WAVE) {
    throw new Error("Invalid WAV file header.");
  }
  let offset = 12;
  let fmt = null;
  let dataOffset = -1;
  let dataSize = 0;
  while (offset + 8 <= view.byteLength) {
    const chunkId = readChunkId(view, offset);
    const chunkSize = view.getUint32(offset + 4, true);
    if (chunkId === FMT) {
      fmt = parseFmtChunk(view, offset);
    } else if (chunkId === DATA) {
      dataOffset = offset + 8;
      dataSize = chunkSize;
      break;
    }
    offset += 8 + chunkSize + chunkSize % 2;
  }
  if (!fmt) {
    throw new Error("WAV file is missing fmt chunk.");
  }
  if (dataOffset < 0) {
    throw new Error("WAV file is missing data chunk.");
  }
  const samples = decodePcm(view, dataOffset, dataSize, fmt);
  return {
    sampleRate: fmt.sampleRate,
    samples
  };
}
function resampleLinear(samples, inputRate, outputRate) {
  if (inputRate === outputRate) {
    return samples;
  }
  const ratio = inputRate / outputRate;
  const outLength = Math.max(1, Math.floor(samples.length / ratio));
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i += 1) {
    const srcPos = i * ratio;
    const left = Math.floor(srcPos);
    const right = Math.min(left + 1, samples.length - 1);
    const alpha = srcPos - left;
    out[i] = samples[left] * (1 - alpha) + samples[right] * alpha;
  }
  return out;
}
function normalize(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) {
    sum += samples[i];
  }
  const mean = sum / samples.length;
  let varianceAcc = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const centered = samples[i] - mean;
    varianceAcc += centered * centered;
  }
  const std = Math.sqrt(varianceAcc / samples.length);
  const denom = std + 1e-5;
  const normalized = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    normalized[i] = (samples[i] - mean) / denom;
  }
  return normalized;
}

// src/models.js
import * as ort from "onnxruntime-web";

// src/utils.js
function int64TensorValues(values) {
  return BigInt64Array.from(values.map((value) => BigInt(value)));
}
function readScalarInt(tensor) {
  if (!tensor || !tensor.data || tensor.data.length === 0) {
    throw new Error("Expected scalar tensor with data.");
  }
  const value = tensor.data[0];
  return typeof value === "bigint" ? Number(value) : value;
}
function firstExistingInputName(session, candidates, fallbackIndex) {
  for (const name of candidates) {
    if (session.inputNames.includes(name)) {
      return name;
    }
  }
  return session.inputNames[fallbackIndex];
}

// src/models.js
function intTensor(type, values, dims) {
  if (type === "int64") {
    return new ort.Tensor("int64", int64TensorValues(values), dims);
  }
  return new ort.Tensor("int32", Int32Array.from(values), dims);
}
function ensureOutput(session, outputMap, index) {
  const name = session.outputNames[index];
  const tensor = outputMap[name];
  if (!tensor) {
    throw new Error(`Missing output tensor at index ${index} (${name}).`);
  }
  return tensor;
}
function valueToNumber(value) {
  return typeof value === "bigint" ? Number(value) : value;
}
function numbersFromTensor(tensor) {
  return Array.from(tensor.data, valueToNumber);
}
function shapeFromMetadata(metadataShape) {
  if (!Array.isArray(metadataShape) || metadataShape.length === 0) {
    return [1];
  }
  return metadataShape.map(
    (dimension) => typeof dimension === "number" && dimension > 0 ? dimension : 1
  );
}
function product(values) {
  return values.reduce((acc, value) => acc * value, 1);
}
function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}
function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}
function buildHannWindow(length) {
  const window = new Float32Array(length);
  if (length === 1) {
    window[0] = 1;
    return window;
  }
  for (let i = 0; i < length; i += 1) {
    window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (length - 1));
  }
  return window;
}
function buildMelFilterBank({ sampleRate, nFft, nMels, fMin, fMax }) {
  const nFreqs = Math.floor(nFft / 2) + 1;
  const bank = Array.from({ length: nMels }, () => new Float32Array(nFreqs));
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);
  const melPoints = new Float32Array(nMels + 2);
  for (let i = 0; i < melPoints.length; i += 1) {
    melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1);
  }
  const hzPoints = Array.from(melPoints, melToHz);
  const bin = hzPoints.map((hz) => Math.floor((nFft + 1) * hz / sampleRate));
  for (let m = 1; m <= nMels; m += 1) {
    const left = bin[m - 1];
    const center = bin[m];
    const right = bin[m + 1];
    for (let k = left; k < center; k += 1) {
      if (k >= 0 && k < nFreqs && center > left) {
        bank[m - 1][k] = (k - left) / (center - left);
      }
    }
    for (let k = center; k < right; k += 1) {
      if (k >= 0 && k < nFreqs && right > center) {
        bank[m - 1][k] = (right - k) / (right - center);
      }
    }
  }
  return bank;
}
function powerSpectrum(frame, nFft) {
  const bins = Math.floor(nFft / 2) + 1;
  const out = new Float32Array(bins);
  for (let k = 0; k < bins; k += 1) {
    let real = 0;
    let imag = 0;
    for (let n = 0; n < nFft; n += 1) {
      const sample = frame[n] ?? 0;
      const angle = 2 * Math.PI * k * n / nFft;
      real += sample * Math.cos(angle);
      imag -= sample * Math.sin(angle);
    }
    out[k] = real * real + imag * imag;
  }
  return out;
}
function logMelSpectrogram(samples, options = {}) {
  const sampleRate = options.sampleRate ?? 16e3;
  const nMels = options.nMels ?? 80;
  const nFft = options.nFft ?? 512;
  const winLength = options.winLength ?? 400;
  const hopLength = options.hopLength ?? 160;
  const fMin = options.fMin ?? 0;
  const fMax = options.fMax ?? sampleRate / 2;
  const logEps = options.logEps ?? 1e-10;
  const preemphasis = options.preemphasis ?? 0.97;
  const normalize2 = options.normalize ?? true;
  if (preemphasis > 0) {
    const emphasized = new Float32Array(samples.length);
    if (samples.length > 0) {
      emphasized[0] = samples[0];
    }
    for (let i = 1; i < samples.length; i += 1) {
      emphasized[i] = samples[i] - preemphasis * samples[i - 1];
    }
    samples = emphasized;
  }
  if (samples.length < winLength) {
    const padded = new Float32Array(winLength);
    padded.set(samples, 0);
    samples = padded;
  }
  const frameCount = 1 + Math.floor((samples.length - winLength) / hopLength);
  const frameWindow = buildHannWindow(winLength);
  const melBank = buildMelFilterBank({ sampleRate, nFft, nMels, fMin, fMax });
  const features = new Float32Array(nMels * frameCount);
  const fftFrame = new Float32Array(nFft);
  for (let t = 0; t < frameCount; t += 1) {
    const start = t * hopLength;
    fftFrame.fill(0);
    for (let i = 0; i < winLength; i += 1) {
      fftFrame[i] = samples[start + i] * frameWindow[i];
    }
    const spectrum = powerSpectrum(fftFrame, nFft);
    for (let m = 0; m < nMels; m += 1) {
      const filter = melBank[m];
      let energy = 0;
      for (let k = 0; k < spectrum.length; k += 1) {
        energy += spectrum[k] * filter[k];
      }
      features[m * frameCount + t] = Math.log(Math.max(logEps, energy));
    }
  }
  if (normalize2) {
    for (let m = 0; m < nMels; m += 1) {
      const offset = m * frameCount;
      let mean = 0;
      for (let t = 0; t < frameCount; t += 1) {
        mean += features[offset + t];
      }
      mean /= frameCount;
      let variance = 0;
      for (let t = 0; t < frameCount; t += 1) {
        const delta = features[offset + t] - mean;
        variance += delta * delta;
      }
      const std = Math.sqrt(variance / frameCount) + 1e-5;
      for (let t = 0; t < frameCount; t += 1) {
        features[offset + t] = (features[offset + t] - mean) / std;
      }
    }
  }
  return {
    features,
    frameCount,
    nMels
  };
}
var PreprocessorModel = class {
  constructor(session) {
    this.session = session;
    this.inputSignalName = firstExistingInputName(session, ["input_signal", "waveforms"], 0);
    this.lengthName = firstExistingInputName(session, ["length", "waveforms_lens"], 1);
    this.lengthType = session.inputMetadata.find((item) => item.name === this.lengthName)?.type ?? "int64";
  }
  async run(audioSamples) {
    const audioTensor = new ort.Tensor("float32", audioSamples, [1, audioSamples.length]);
    const lengthTensor = intTensor(this.lengthType, [audioSamples.length], [1]);
    const outputs = await this.session.run({
      [this.inputSignalName]: audioTensor,
      [this.lengthName]: lengthTensor
    });
    return {
      signal: ensureOutput(this.session, outputs, 0),
      length: ensureOutput(this.session, outputs, 1)
    };
  }
};
var EncoderModel = class {
  constructor(session, options = {}) {
    this.session = session;
    this.options = options;
    this.audioSignalName = firstExistingInputName(session, ["audio_signal", "features", "waveforms"], 0);
    this.lengthName = firstExistingInputName(session, ["length", "features_lens", "waveforms_lens"], 1);
    this.audioMetadata = session.inputMetadata.find((item) => item.name === this.audioSignalName);
    this.lengthMetadata = session.inputMetadata.find((item) => item.name === this.lengthName);
  }
  prepareInputsFromWaveform(samples) {
    const inputRank = this.audioMetadata?.shape?.length ?? 2;
    if (inputRank === 2) {
      const audioTensor2 = new ort.Tensor("float32", samples, [1, samples.length]);
      const lengthType2 = this.lengthMetadata?.type ?? "int64";
      const lengthTensor2 = intTensor(lengthType2, [samples.length], [1]);
      return { signal: audioTensor2, length: lengthTensor2 };
    }
    if (inputRank !== 3) {
      throw new Error(`Unsupported encoder input rank ${inputRank} for '${this.audioSignalName}'.`);
    }
    const nMelsFromShape = this.audioMetadata?.shape?.[1];
    const fe = this.options.config?.feature_extraction_params;
    const configNels = this.options.config?.features_size ?? fe?.n_mels;
    const nMels = Number.isFinite(nMelsFromShape) ? nMelsFromShape : configNels ?? 80;
    const isGigaam = this.options.config?.model_type === "gigaam";
    const sampleRate = this.options.sampleRate ?? fe?.sample_rate ?? 16e3;
    const nFft = fe?.n_fft ?? 512;
    const winLength = fe?.window_size ? Math.round(Number(fe.window_size) * sampleRate) : 400;
    const hopLength = fe?.window_stride ? Math.round(Number(fe.window_stride) * sampleRate) : 160;
    const mel = logMelSpectrogram(samples, {
      sampleRate,
      nMels,
      nFft,
      winLength,
      hopLength,
      preemphasis: fe?.preemphasis_coefficient ?? (isGigaam ? 0 : 0.97),
      normalize: !isGigaam
    });
    const audioTensor = new ort.Tensor("float32", mel.features, [1, mel.nMels, mel.frameCount]);
    const lengthType = this.lengthMetadata?.type ?? "int64";
    const lengthTensor = intTensor(lengthType, [mel.frameCount], [1]);
    return { signal: audioTensor, length: lengthTensor };
  }
  async run(processedSignalTensor, processedLengthTensor) {
    const outputs = await this.session.run({
      [this.audioSignalName]: processedSignalTensor,
      [this.lengthName]: processedLengthTensor
    });
    const encoded = ensureOutput(this.session, outputs, 0);
    const encodedLength = ensureOutput(this.session, outputs, 1);
    if (encoded.dims.length === 3 && encoded.dims[0] === 1) {
      return {
        encodedData: encoded.data,
        encodedDims: encoded.dims,
        encodedLayout: "BCT",
        encodedLength: readScalarInt(encodedLength)
      };
    }
    throw new Error(
      `Unexpected encoder output shape: [${encoded.dims.join(", ")}]. Expected [1, C, T].`
    );
  }
};
var CtcAcousticModel = class extends EncoderModel {
  constructor(session, options = {}) {
    super(session, options);
    this.vocabSize = options.vocabSize;
  }
  async run(processedSignalTensor, processedLengthTensor) {
    const outputs = await this.session.run({
      [this.audioSignalName]: processedSignalTensor,
      [this.lengthName]: processedLengthTensor
    });
    const logits = ensureOutput(this.session, outputs, 0);
    const lengthTensor = this.session.outputNames.length > 1 ? outputs[this.session.outputNames[1]] : null;
    if (logits.dims.length !== 3) {
      throw new Error(
        `Unexpected CTC output shape: [${logits.dims.join(", ")}]. Expected rank-3 [1, T, V] or [1, V, T].`
      );
    }
    let d1;
    let d2;
    if (logits.dims[0] === 1) {
      d1 = logits.dims[1];
      d2 = logits.dims[2];
    } else if (logits.dims[1] === 1) {
      d1 = logits.dims[0];
      d2 = logits.dims[2];
    } else {
      throw new Error(`Unexpected CTC batch layout: [${logits.dims.join(", ")}].`);
    }
    let layout = "BTV";
    if (this.vocabSize && d1 === this.vocabSize && d2 !== this.vocabSize) {
      layout = "BVT";
    } else if (this.vocabSize && d2 === this.vocabSize) {
      layout = "BTV";
    }
    return {
      encodedData: logits.data,
      encodedDims: [1, d1, d2],
      encodedLayout: layout,
      encodedLength: lengthTensor ? readScalarInt(lengthTensor) : layout === "BVT" ? d2 : d1
    };
  }
};
var DecoderTransducerModel = class {
  constructor(session, options = {}) {
    this.session = session;
    this.decoderKind = options.decoderKind ?? "tdt";
    this.vocabSize = options.vocabSize;
    this.encoderOutputsName = firstExistingInputName(session, ["encoder_outputs"], 0);
    this.targetsName = firstExistingInputName(session, ["targets"], 1);
    this.targetLengthName = session.inputNames.includes("target_length") ? "target_length" : null;
    this.targetsType = session.inputMetadata.find((item) => item.name === this.targetsName)?.type ?? "int32";
    this.targetLengthType = this.targetLengthName ? session.inputMetadata.find((item) => item.name === this.targetLengthName)?.type ?? "int32" : "int32";
    this.stateInputNames = session.inputNames.filter(
      (name) => name !== this.encoderOutputsName && name !== this.targetsName && name !== this.targetLengthName
    );
    this.stateInputMetadata = this.stateInputNames.map((name) => {
      const meta = session.inputMetadata.find((item) => item.name === name);
      if (!meta) {
        throw new Error(`Missing input metadata for decoder state '${name}'.`);
      }
      return meta;
    });
    const stateOutputCandidates = session.outputNames.filter((name) => /state/i.test(name));
    this.stateOutputNames = stateOutputCandidates.length > 0 ? stateOutputCandidates : session.outputNames.slice(1).filter((name) => !/length/i.test(name));
  }
  initialStates() {
    const states = /* @__PURE__ */ new Map();
    for (const meta of this.stateInputMetadata) {
      const shape = shapeFromMetadata(meta.shape);
      const length = product(shape);
      if (meta.type !== "float32") {
        throw new Error(`Unsupported decoder state type '${meta.type}' for '${meta.name}'.`);
      }
      states.set(meta.name, new ort.Tensor("float32", new Float32Array(length), shape));
    }
    return states;
  }
  resolveNextStates(outputs, currentStates) {
    const nextStates = /* @__PURE__ */ new Map();
    for (let i = 0; i < this.stateInputNames.length; i += 1) {
      const inputName = this.stateInputNames[i];
      const outputName = this.stateOutputNames[i];
      if (outputName && outputs[outputName]) {
        nextStates.set(inputName, outputs[outputName]);
      } else {
        nextStates.set(inputName, currentStates.get(inputName));
      }
    }
    return nextStates;
  }
  argmax(data, start, end) {
    let maxValue = -Infinity;
    let maxIndex = start;
    for (let i = start; i < end; i += 1) {
      const value = data[i];
      if (value > maxValue) {
        maxValue = value;
        maxIndex = i;
      }
    }
    return maxIndex;
  }
  rnntCandidates(mainOutputTensor) {
    if (mainOutputTensor.type === "int32" || mainOutputTensor.type === "int64") {
      return numbersFromTensor(mainOutputTensor).map((token) => ({ token, duration: 0 }));
    }
    const flat = mainOutputTensor.data;
    return [{ token: this.argmax(flat, 0, flat.length), duration: 0 }];
  }
  tdtCandidates(mainOutputTensor) {
    if (!this.vocabSize || this.vocabSize <= 0) {
      throw new Error("TDT decoder requires a positive vocabSize.");
    }
    const flat = mainOutputTensor.data;
    if (flat.length < this.vocabSize) {
      throw new Error(`TDT logits length ${flat.length} is smaller than vocabSize ${this.vocabSize}.`);
    }
    const token = this.argmax(flat, 0, this.vocabSize);
    let duration = 0;
    if (flat.length > this.vocabSize) {
      duration = this.argmax(flat, this.vocabSize, flat.length) - this.vocabSize;
    }
    return [{ token, duration }];
  }
  async predict(encoderFrameData, encoderFrameDims, token, states) {
    const feeds = {
      [this.encoderOutputsName]: new ort.Tensor("float32", encoderFrameData, encoderFrameDims),
      [this.targetsName]: intTensor(this.targetsType, [token], [1, 1])
    };
    if (this.targetLengthName) {
      feeds[this.targetLengthName] = intTensor(this.targetLengthType, [1], [1]);
    }
    for (const stateName of this.stateInputNames) {
      const state = states.get(stateName);
      if (!state) {
        throw new Error(`Missing decoder state '${stateName}'.`);
      }
      feeds[stateName] = state;
    }
    const outputs = await this.session.run(feeds);
    const mainOutput = ensureOutput(this.session, outputs, 0);
    const nextStates = this.resolveNextStates(outputs, states);
    const candidates = this.decoderKind === "rnnt" ? this.rnntCandidates(mainOutput) : this.tdtCandidates(mainOutput);
    return { candidates, nextStates };
  }
};
var TransducerGreedyDecoder = class {
  constructor(model, options = {}) {
    this.model = model;
    this.maxSymbols = options.maxSymbols ?? 10;
    this.blankTokenId = options.blankTokenId ?? 0;
    this.defaultDuration = options.defaultDuration ?? 1;
  }
  frameAt(encodedData, encodedDims, encodedLayout, t) {
    if (encodedLayout === "BCT") {
      const channels = encodedDims[1];
      const time = encodedDims[2];
      if (t >= time) {
        throw new Error(`Time index ${t} out of range for encoder output time=${time}.`);
      }
      const frame = new Float32Array(channels);
      for (let c = 0; c < channels; c += 1) {
        frame[c] = encodedData[c * time + t];
      }
      return { data: frame, dims: [1, channels, 1] };
    }
    throw new Error(`Unsupported encoder layout: ${encodedLayout}`);
  }
  async decode(encodedData, encodedDims, encodedLayout, encodedLength) {
    const tokenIds = [];
    const tokenFrames = [];
    let currentToken = this.blankTokenId;
    let states = this.model.initialStates();
    let t = 0;
    const timeSteps = encodedLayout === "BCT" ? encodedDims[2] : 0;
    const limit = Math.min(encodedLength, timeSteps);
    while (t < limit) {
      const frame = this.frameAt(encodedData, encodedDims, encodedLayout, t);
      let nextT = t + this.defaultDuration;
      const emittedIndexes = [];
      for (let n = 0; n < this.maxSymbols; n += 1) {
        const { candidates, nextStates } = await this.model.predict(
          frame.data,
          frame.dims,
          currentToken,
          states
        );
        let consumed = false;
        for (const candidate of candidates) {
          const token = valueToNumber(candidate.token);
          const duration = valueToNumber(candidate.duration);
          if (token !== this.blankTokenId) {
            const emissionIndex = tokenIds.length;
            tokenIds.push(token);
            tokenFrames.push({ startFrame: t, endFrame: t + this.defaultDuration });
            emittedIndexes.push(emissionIndex);
            currentToken = token;
            states = nextStates;
          }
          if (duration > 0) {
            nextT = t + duration;
            consumed = true;
            break;
          }
        }
        if (consumed) {
          break;
        }
      }
      for (const emissionIndex of emittedIndexes) {
        tokenFrames[emissionIndex].endFrame = nextT;
      }
      t = nextT;
    }
    return { tokenIds, tokenFrames, totalFrames: limit };
  }
};
var CtcGreedyDecoder = class {
  constructor(options = {}) {
    this.blankTokenId = options.blankTokenId ?? 0;
  }
  argmaxAt(data, start, size) {
    let best = 0;
    let bestValue = -Infinity;
    for (let i = 0; i < size; i += 1) {
      const value = data[start + i];
      if (value > bestValue) {
        bestValue = value;
        best = i;
      }
    }
    return best;
  }
  async decode(encodedData, encodedDims, encodedLayout, encodedLength) {
    const tokenIds = [];
    const tokenFrames = [];
    let previous = this.blankTokenId;
    const timeSteps = encodedLayout === "BVT" ? encodedDims[2] : encodedDims[1];
    const vocabSize = encodedLayout === "BVT" ? encodedDims[1] : encodedDims[2];
    const limit = Math.min(encodedLength, timeSteps);
    for (let t = 0; t < limit; t += 1) {
      let token;
      if (encodedLayout === "BVT") {
        let best = 0;
        let bestValue = -Infinity;
        for (let v = 0; v < vocabSize; v += 1) {
          const value = encodedData[v * timeSteps + t];
          if (value > bestValue) {
            bestValue = value;
            best = v;
          }
        }
        token = best;
      } else {
        token = this.argmaxAt(encodedData, t * vocabSize, vocabSize);
      }
      if (token === this.blankTokenId) {
        previous = this.blankTokenId;
        continue;
      }
      if (token === previous) {
        const last = tokenFrames[tokenFrames.length - 1];
        if (last) {
          last.endFrame = t + 1;
        }
        continue;
      }
      tokenIds.push(token);
      tokenFrames.push({ startFrame: t, endFrame: t + 1 });
      previous = token;
    }
    return { tokenIds, tokenFrames, totalFrames: limit };
  }
};

// src/whisper.js
import * as ort2 from "onnxruntime-web";
function hzToMel2(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}
function melToHz2(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}
function buildHannWindow2(length) {
  const window = new Float32Array(length);
  if (length === 1) {
    window[0] = 1;
    return window;
  }
  for (let i = 0; i < length; i += 1) {
    window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (length - 1));
  }
  return window;
}
function buildMelFilterBank2({ sampleRate, nFft, nMels, fMin, fMax }) {
  const nFreqs = Math.floor(nFft / 2) + 1;
  const bank = Array.from({ length: nMels }, () => new Float32Array(nFreqs));
  const melMin = hzToMel2(fMin);
  const melMax = hzToMel2(fMax);
  const melPoints = new Float32Array(nMels + 2);
  for (let i = 0; i < melPoints.length; i += 1) {
    melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1);
  }
  const hzPoints = Array.from(melPoints, melToHz2);
  const bin = hzPoints.map((hz) => Math.floor((nFft + 1) * hz / sampleRate));
  for (let m = 1; m <= nMels; m += 1) {
    const left = bin[m - 1];
    const center = bin[m];
    const right = bin[m + 1];
    for (let k = left; k < center; k += 1) {
      if (k >= 0 && k < nFreqs && center > left) {
        bank[m - 1][k] = (k - left) / (center - left);
      }
    }
    for (let k = center; k < right; k += 1) {
      if (k >= 0 && k < nFreqs && right > center) {
        bank[m - 1][k] = (right - k) / (right - center);
      }
    }
  }
  return bank;
}
function powerSpectrum2(frame, nFft) {
  const bins = Math.floor(nFft / 2) + 1;
  const out = new Float32Array(bins);
  for (let k = 0; k < bins; k += 1) {
    let real = 0;
    let imag = 0;
    for (let n = 0; n < nFft; n += 1) {
      const sample = frame[n] ?? 0;
      const angle = 2 * Math.PI * k * n / nFft;
      real += sample * Math.cos(angle);
      imag -= sample * Math.sin(angle);
    }
    out[k] = real * real + imag * imag;
  }
  return out;
}
function whisperLogMelSpectrogram(samples, options = {}) {
  const sampleRate = options.sampleRate ?? 16e3;
  const nMels = options.nMels ?? 80;
  const nFft = 400;
  const hopLength = 160;
  const winLength = 400;
  const chunkSamples = 30 * sampleRate;
  const pad = Math.floor(nFft / 2);
  if (samples.length < chunkSamples) {
    const padded2 = new Float32Array(chunkSamples);
    padded2.set(samples, 0);
    samples = padded2;
  } else if (samples.length > chunkSamples) {
    samples = samples.slice(0, chunkSamples);
  }
  const padded = new Float32Array(samples.length + 2 * pad);
  padded.set(samples, pad);
  const rawFrameCount = 1 + Math.floor((padded.length - nFft) / hopLength);
  const frameCount = Math.max(1, rawFrameCount - 1);
  const frameWindow = buildHannWindow2(winLength);
  const melBank = buildMelFilterBank2({ sampleRate, nFft, nMels, fMin: 0, fMax: sampleRate / 2 });
  const mel = new Float32Array(nMels * frameCount);
  const fftFrame = new Float32Array(nFft);
  for (let t = 0; t < frameCount; t += 1) {
    const start = t * hopLength;
    fftFrame.fill(0);
    for (let i = 0; i < winLength; i += 1) {
      fftFrame[i] = padded[start + i] * frameWindow[i];
    }
    const spectrum = powerSpectrum2(fftFrame, nFft);
    for (let m = 0; m < nMels; m += 1) {
      const filter = melBank[m];
      let energy = 0;
      for (let k = 0; k < spectrum.length; k += 1) {
        energy += spectrum[k] * filter[k];
      }
      mel[m * frameCount + t] = Math.max(1e-10, energy);
    }
  }
  let maxLog = -Infinity;
  for (let i = 0; i < mel.length; i += 1) {
    const logValue = Math.log10(mel[i]);
    mel[i] = logValue;
    if (logValue > maxLog) {
      maxLog = logValue;
    }
  }
  const floor = maxLog - 8;
  for (let i = 0; i < mel.length; i += 1) {
    mel[i] = (Math.max(mel[i], floor) + 4) / 4;
  }
  return new ort2.Tensor("float32", mel, [1, nMels, frameCount]);
}
function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i += 1) bs.push(i);
  for (let i = 161; i <= 172; i += 1) bs.push(i);
  for (let i = 174; i <= 255; i += 1) bs.push(i);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b += 1) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const unicode = cs.map((value) => String.fromCharCode(value));
  const out = /* @__PURE__ */ new Map();
  for (let i = 0; i < bs.length; i += 1) {
    out.set(bs[i], unicode[i]);
  }
  return out;
}
function argmax(values) {
  let idx = 0;
  let max = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] > max) {
      max = values[i];
      idx = i;
    }
  }
  return idx;
}
function hasAnyState(stateMap) {
  for (const value of stateMap.values()) {
    if (value.data.length > 0) {
      return true;
    }
  }
  return false;
}
function intTensorFor(type, values, dims) {
  if (type === "int64") {
    return new ort2.Tensor("int64", BigInt64Array.from(values.map((x) => BigInt(x))), dims);
  }
  return new ort2.Tensor("int32", Int32Array.from(values), dims);
}
function boolTensor(values, dims) {
  return new ort2.Tensor("bool", Uint8Array.from(values.map((v) => v ? 1 : 0)), dims);
}
function tokenIdsToText(tokenIds, vocabById, byteDecoder) {
  let text = "";
  for (const id of tokenIds) {
    const token = vocabById.get(id);
    if (token && !token.startsWith("<|")) {
      text += token;
    }
  }
  const bytes = [];
  for (const ch of text) {
    const value = byteDecoder.get(ch);
    if (value != null) {
      bytes.push(value);
    }
  }
  return new TextDecoder("utf-8", { fatal: false }).decode(Uint8Array.from(bytes)).replace(/^ /, "");
}
var WhisperBaseModel = class {
  constructor({ config, vocab, addedTokens }) {
    this.config = config;
    this.tokens = { ...vocab, ...addedTokens };
    this.vocabById = /* @__PURE__ */ new Map();
    for (const [token, id] of Object.entries(this.tokens)) {
      this.vocabById.set(Number(id), token);
    }
    this.bosTokenId = this.tokens["<|startoftranscript|>"];
    this.eosTokenId = this.tokens["<|endoftext|>"];
    this.transcribeTokenId = this.tokens["<|transcribe|>"];
    this.notimestampsTokenId = this.tokens["<|notimestamps|>"];
    this.transcribeInput = [[
      this.bosTokenId,
      this.eosTokenId,
      this.transcribeTokenId,
      this.notimestampsTokenId
    ]];
    this.detectLangInput = [[this.bosTokenId]];
    const unicodeMap = bytesToUnicode();
    this.byteDecoder = /* @__PURE__ */ new Map();
    for (const [k, v] of unicodeMap.entries()) {
      this.byteDecoder.set(v, k);
    }
    this.sampleRate = 16e3;
    this.nMels = Number(config.features_size ?? config.num_mel_bins ?? 80);
  }
  _prepareFeatures(samples, sampleRate) {
    let prepared = samples;
    if (!(prepared instanceof Float32Array)) {
      prepared = Float32Array.from(prepared);
    }
    if (sampleRate !== this.sampleRate) {
      prepared = resampleLinear(prepared, sampleRate, this.sampleRate);
    }
    return whisperLogMelSpectrogram(prepared, { sampleRate: this.sampleRate, nMels: this.nMels });
  }
  _decodeTokens(tokens) {
    return tokenIdsToText(tokens, this.vocabById, this.byteDecoder);
  }
  async _recognizeFeatures(inputFeatures, options = {}) {
    let prompt = this.transcribeInput.map((row) => row.slice());
    if (options.language) {
      const languageToken = this.tokens[`<|${options.language}|>`];
      if (languageToken != null) {
        prompt[0][1] = languageToken;
      }
    } else {
      const detected = await this._decoding(inputFeatures, this.detectLangInput, 3);
      if (detected.length > 0 && detected[0].length > 1) {
        prompt[0][1] = detected[0][1];
      }
    }
    const tokenMatrix = await this._decoding(inputFeatures, prompt, options.maxLength ?? 448);
    const tokenIds = tokenMatrix[0] ?? [];
    return {
      tokenIds,
      tokenFrames: [],
      words: [],
      text: this._decodeTokens(tokenIds)
    };
  }
  async transcribeSamples(samples, sampleRate = this.sampleRate, options = {}) {
    const features = this._prepareFeatures(samples, sampleRate);
    return this._recognizeFeatures(features, options);
  }
  async transcribeWavBuffer(arrayBuffer, options = {}) {
    const decoded = decodeWav(arrayBuffer);
    return this.transcribeSamples(decoded.samples, decoded.sampleRate, options);
  }
};
var WhisperOrtModel = class extends WhisperBaseModel {
  constructor({ config, vocab, addedTokens, session }) {
    super({ config, vocab, addedTokens });
    this.session = session;
    this.inputMetadata = new Map(session.inputMetadata.map((meta) => [meta.name, meta]));
    this.outputName = session.outputNames.includes("sequences") ? "sequences" : session.outputNames[0];
  }
  _paramTensor(name, value) {
    const meta = this.inputMetadata.get(name);
    if (!meta) {
      return null;
    }
    if (meta.type === "float" || meta.type === "float32") {
      return new ort2.Tensor("float32", Float32Array.from([value]), [1]);
    }
    if (meta.type === "int64") {
      return intTensorFor("int64", [value], [1]);
    }
    return intTensorFor("int32", [value], [1]);
  }
  async _decoding(inputFeatures, tokens, maxLength = 448) {
    const decoderInput = Int32Array.from(tokens.flat());
    const decoderTensor = new ort2.Tensor("int32", decoderInput, [tokens.length, tokens[0].length]);
    const feeds = {
      input_features: inputFeatures,
      decoder_input_ids: decoderTensor
    };
    const optionalParams = {
      max_length: maxLength,
      min_length: 0,
      num_beams: 1,
      num_return_sequences: 1,
      length_penalty: 1,
      repetition_penalty: 1
    };
    for (const [key, value] of Object.entries(optionalParams)) {
      const tensor = this._paramTensor(key, value);
      if (tensor) {
        feeds[key] = tensor;
      }
    }
    const outputs = await this.session.run(feeds);
    const sequences = outputs[this.outputName];
    if (!sequences) {
      throw new Error("Whisper ORT decoding did not return sequences output.");
    }
    const data = Array.from(sequences.data, (x) => Number(x));
    const shape = sequences.dims;
    if (shape.length === 3) {
      const [batch, beam, length] = shape;
      const out = [];
      for (let b = 0; b < batch; b += 1) {
        const start = b * beam * length;
        out.push(data.slice(start, start + length));
      }
      return out;
    }
    if (shape.length === 2) {
      const [batch, length] = shape;
      const out = [];
      for (let b = 0; b < batch; b += 1) {
        out.push(data.slice(b * length, (b + 1) * length));
      }
      return out;
    }
    throw new Error(`Unexpected whisper-ort sequences shape: [${shape.join(", ")}].`);
  }
};
var WhisperHfModel = class extends WhisperBaseModel {
  constructor({ config, vocab, addedTokens, encoderSession, decoderSession }) {
    super({ config, vocab, addedTokens });
    this.encoderSession = encoderSession;
    this.decoderSession = decoderSession;
    this.decoderInputMeta = new Map(decoderSession.inputMetadata.map((meta) => [meta.name, meta]));
    this.decoderOutputNames = decoderSession.outputNames;
    this.inputIdName = decoderSession.inputNames.includes("input_ids") ? "input_ids" : decoderSession.inputNames[0];
    this.encoderHiddenName = decoderSession.inputNames.includes("encoder_hidden_states") ? "encoder_hidden_states" : decoderSession.inputNames.find((name) => name.includes("encoder"));
    this.useCacheBranchName = decoderSession.inputNames.includes("use_cache_branch") ? "use_cache_branch" : null;
    this.pastInputNames = decoderSession.inputNames.filter((name) => name.startsWith("past_key_values."));
  }
  _emptyStateTensor(meta) {
    const shape = (meta.shape ?? []).map((dim) => {
      if (typeof dim === "number") {
        return dim >= 0 ? dim : 1;
      }
      if (typeof dim === "string") {
        const lowered = dim.toLowerCase();
        if (lowered.includes("past") || lowered.includes("sequence")) {
          return 0;
        }
      }
      return 1;
    });
    const size = shape.length === 0 ? 0 : shape.reduce((acc, v) => acc * v, 1);
    return new ort2.Tensor("float32", new Float32Array(size), shape);
  }
  _createState() {
    const state = /* @__PURE__ */ new Map();
    for (const name of this.pastInputNames) {
      const meta = this.decoderInputMeta.get(name);
      if (!meta) {
        continue;
      }
      state.set(name, this._emptyStateTensor(meta));
    }
    return state;
  }
  _decoderInputTensor(tokens, useCache) {
    const width = useCache ? 1 : tokens[0].length;
    const data = new Int32Array(tokens.length * width);
    for (let b = 0; b < tokens.length; b += 1) {
      if (useCache) {
        data[b] = tokens[b][tokens[b].length - 1];
      } else {
        data.set(tokens[b], b * width);
      }
    }
    const meta = this.decoderInputMeta.get(this.inputIdName);
    const type = meta?.type === "int64" ? "int64" : "int32";
    if (type === "int64") {
      return new ort2.Tensor("int64", BigInt64Array.from(data, (x) => BigInt(x)), [tokens.length, width]);
    }
    return new ort2.Tensor("int32", data, [tokens.length, width]);
  }
  async _encode(inputFeatures) {
    const inputName = this.encoderSession.inputNames[0];
    const outputs = await this.encoderSession.run({ [inputName]: inputFeatures });
    return outputs[this.encoderSession.outputNames[0]];
  }
  async _decodeStep(tokens, state, encoderOut) {
    const useCache = hasAnyState(state);
    const feeds = {
      [this.inputIdName]: this._decoderInputTensor(tokens, useCache),
      [this.encoderHiddenName]: encoderOut
    };
    if (this.useCacheBranchName) {
      feeds[this.useCacheBranchName] = boolTensor([useCache], [1]);
    }
    for (const [name, value] of state.entries()) {
      feeds[name] = value;
    }
    const outputs = await this.decoderSession.run(feeds);
    const logits = outputs[this.decoderOutputNames[0]];
    const nextState = /* @__PURE__ */ new Map();
    for (const inputName of this.pastInputNames) {
      const outputName = inputName.replace("past_key_values.", "present.");
      const prev = state.get(inputName);
      const candidate = outputs[outputName] ?? prev;
      nextState.set(inputName, candidate && candidate.data.length > 0 ? candidate : prev);
    }
    return { logits, nextState };
  }
  async _decoding(inputFeatures, tokens, maxLength = 448) {
    const encoderOut = await this._encode(inputFeatures);
    let state = this._createState();
    let outputTokens = tokens.map((row) => row.slice());
    for (let step = outputTokens[0].length; step < maxLength; step += 1) {
      const { logits, nextState } = await this._decodeStep(outputTokens, state, encoderOut);
      state = nextState;
      const dims = logits.dims;
      const vocabSize = dims[dims.length - 1];
      const seqLen = dims[dims.length - 2];
      const batch = dims[0];
      for (let b = 0; b < batch; b += 1) {
        if (outputTokens[b][outputTokens[b].length - 1] === this.eosTokenId) {
          outputTokens[b].push(this.eosTokenId);
          continue;
        }
        const offset = b * seqLen * vocabSize + (seqLen - 1) * vocabSize;
        const next = argmax(logits.data.slice(offset, offset + vocabSize));
        outputTokens[b].push(next);
      }
      if (outputTokens.every((row) => row[row.length - 1] === this.eosTokenId)) {
        break;
      }
    }
    return outputTokens;
  }
};

// src/asr-model.js
function configureOrtWeb(options = {}) {
  if (options.numThreads != null) {
    ort3.env.wasm.numThreads = options.numThreads;
  }
  if (options.wasmPaths) {
    ort3.env.wasm.wasmPaths = options.wasmPaths;
  }
  if (options.simd != null) {
    ort3.env.wasm.simd = options.simd;
  }
  if (options.proxy != null) {
    ort3.env.wasm.proxy = options.proxy;
  }
}
function parseVocabulary(vocabularyText) {
  const lines = vocabularyText.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const indexed = lines.every((line) => /^(.*)\s+(\d+)$/.test(line));
  if (indexed) {
    const parsed = lines.map((line) => {
      const match = line.match(/^(.*)\s+(\d+)$/);
      return { token: match[1], id: Number(match[2]) };
    });
    const maxId = parsed.reduce((acc, item) => Math.max(acc, item.id), 0);
    const tokens = new Array(maxId + 1);
    for (const item of parsed) {
      tokens[item.id] = item.token;
    }
    return tokens;
  }
  return lines;
}
function isControlToken(token) {
  return token && token.startsWith("<") && token.endsWith(">");
}
function decodeTokenPiece(token) {
  return token.replaceAll("\u2581", " ");
}
function decodeText(tokens, tokenIds) {
  return tokenIds.map((tokenId) => tokens[tokenId]).filter((token) => token && !isControlToken(token)).map((token) => decodeTokenPiece(token)).join("").trim();
}
function wordTimestamps(tokens, tokenIds, tokenFrames, secondsPerFrame) {
  const words = [];
  let current = null;
  const closeCurrent = () => {
    if (!current) {
      return;
    }
    const cleaned = current.text.trim();
    if (cleaned) {
      words.push({
        word: cleaned,
        start: Number((current.startFrame * secondsPerFrame).toFixed(3)),
        end: Number((current.endFrame * secondsPerFrame).toFixed(3))
      });
    }
    current = null;
  };
  for (let i = 0; i < tokenIds.length; i += 1) {
    const token = tokens[tokenIds[i]];
    const frame = tokenFrames[i];
    if (!token || !frame || isControlToken(token)) {
      continue;
    }
    const startsNewWord = token.startsWith("\u2581");
    const piece = decodeTokenPiece(token);
    if (startsNewWord) {
      closeCurrent();
      current = {
        text: piece.trimStart(),
        startFrame: frame.startFrame,
        endFrame: frame.endFrame
      };
      continue;
    }
    if (!current) {
      current = {
        text: piece,
        startFrame: frame.startFrame,
        endFrame: frame.endFrame
      };
    } else {
      current.text += piece;
      current.endFrame = frame.endFrame;
    }
  }
  closeCurrent();
  return words;
}
function detectBlankTokenId(tokens) {
  const preferred = ["<blk>", "<blank>"];
  for (const name of preferred) {
    const index = tokens.findIndex((token) => token === name);
    if (index >= 0) {
      return index;
    }
  }
  return tokens.length - 1;
}
var AsrModel = class {
  constructor({ preprocessor, encoder, decoder, tokens, sampleRate = 16e3 }) {
    this.preprocessor = preprocessor;
    this.encoder = encoder;
    this.decoder = decoder;
    this.tokens = tokens;
    this.sampleRate = sampleRate;
  }
  async transcribeSamples(samples, sampleRate = this.sampleRate) {
    let prepared = samples;
    if (!(prepared instanceof Float32Array)) {
      prepared = Float32Array.from(prepared);
    }
    if (sampleRate !== this.sampleRate) {
      prepared = resampleLinear(prepared, sampleRate, this.sampleRate);
    }
    if (this.preprocessor) {
      prepared = normalize(prepared);
    }
    const encoderInputs = this.preprocessor ? await this.preprocessor.run(prepared) : this.encoder.prepareInputsFromWaveform(prepared);
    const encoded = await this.encoder.run(encoderInputs.signal, encoderInputs.length);
    const decoded = await this.decoder.decode(
      encoded.encodedData,
      encoded.encodedDims,
      encoded.encodedLayout,
      encoded.encodedLength
    );
    const tokenIds = decoded.tokenIds;
    const tokenFrames = decoded.tokenFrames;
    const secondsPerFrame = encoded.encodedLength > 0 ? prepared.length / this.sampleRate / encoded.encodedLength : 0;
    return {
      tokenIds,
      tokenFrames,
      words: wordTimestamps(this.tokens, tokenIds, tokenFrames, secondsPerFrame),
      text: decodeText(this.tokens, tokenIds)
    };
  }
  async transcribeWavBuffer(arrayBuffer) {
    const decoded = decodeWav(arrayBuffer);
    return this.transcribeSamples(decoded.samples, decoded.sampleRate);
  }
};
function intTensorFor2(type, values, dims) {
  if (type === "int64") {
    return new ort3.Tensor("int64", BigInt64Array.from(values.map((x) => BigInt(x))), dims);
  }
  return new ort3.Tensor("int32", Int32Array.from(values), dims);
}
function zerosTensor(type, shape) {
  const size = shape.reduce((acc, value) => acc * value, 1);
  if (type === "float32") {
    return new ort3.Tensor("float32", new Float32Array(size), shape);
  }
  if (type === "float16") {
    return new ort3.Tensor("float16", new Uint16Array(size), shape);
  }
  throw new Error(`Unsupported tensor init type '${type}'.`);
}
function argmaxSlice(data, start, length) {
  let bestIndex = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < length; i += 1) {
    const value = data[start + i];
    if (value > bestValue) {
      bestValue = value;
      bestIndex = i;
    }
  }
  return bestIndex;
}
function firstExistingName(names, candidates, fallbackIndex = 0) {
  for (const candidate of candidates) {
    if (names.includes(candidate)) {
      return candidate;
    }
  }
  return names[fallbackIndex];
}
function metaForName(session, name) {
  return session.inputMetadata.find((meta) => meta.name === name) ?? session.outputMetadata.find((meta) => meta.name === name) ?? null;
}
function shapeFromMeta(meta, options = {}) {
  const batch = options.batch ?? 1;
  return (meta?.shape ?? []).map((dim, idx) => {
    if (typeof dim === "number") {
      return dim >= 0 ? dim : 1;
    }
    if (idx === 0 || String(dim).toLowerCase().includes("batch")) {
      return batch;
    }
    if (String(dim).toLowerCase().includes("time")) {
      return 1;
    }
    return 1;
  });
}
var NemoAedModel = class {
  constructor({ config, tokens, encoderSession, decoderSession }) {
    this.config = config;
    this.tokens = tokens;
    this.tokenToId = /* @__PURE__ */ new Map();
    for (let i = 0; i < tokens.length; i += 1) {
      const token = tokens[i];
      if (token != null && !this.tokenToId.has(token)) {
        this.tokenToId.set(token, i);
      }
    }
    this.encoderSession = encoderSession;
    this.decoderSession = decoderSession;
    this.encoderHelper = new EncoderModel(encoderSession, { config, sampleRate: 16e3 });
    this.sampleRate = 16e3;
    this.maxSequenceLength = config?.max_sequence_length ?? 1024;
    this.encoderOutputEmbeddingsName = encoderSession.outputNames.includes("encoder_embeddings") ? "encoder_embeddings" : encoderSession.outputNames[0];
    this.encoderOutputMaskName = encoderSession.outputNames.includes("encoder_mask") ? "encoder_mask" : encoderSession.outputNames[1];
    this.decoderInputIdName = decoderSession.inputNames.includes("input_ids") ? "input_ids" : decoderSession.inputNames[0];
    this.decoderEncoderEmbeddingsName = decoderSession.inputNames.includes("encoder_embeddings") ? "encoder_embeddings" : decoderSession.inputNames[1];
    this.decoderEncoderMaskName = decoderSession.inputNames.includes("encoder_mask") ? "encoder_mask" : decoderSession.inputNames[2];
    this.decoderMemsName = decoderSession.inputNames.includes("decoder_mems") ? "decoder_mems" : decoderSession.inputNames[3];
    this.decoderInputIdType = decoderSession.inputMetadata.find((meta) => meta.name === this.decoderInputIdName)?.type ?? "int32";
    const memMeta = decoderSession.inputMetadata.find((meta) => meta.name === this.decoderMemsName);
    if (!memMeta) {
      throw new Error("Decoder input metadata for decoder_mems is missing.");
    }
    this.decoderMemsType = memMeta.type;
    this.decoderMemsShapeTemplate = memMeta.shape;
    this.logitsName = decoderSession.outputNames.includes("logits") ? "logits" : decoderSession.outputNames[0];
    this.decoderHiddenStatesName = decoderSession.outputNames.includes("decoder_hidden_states") ? "decoder_hidden_states" : decoderSession.outputNames[1];
  }
  canaryPrefix(options = {}) {
    const fallbackLanguage = options.language ?? "en";
    const targetLanguage = options.targetLanguage ?? "en";
    const pnc = options.pnc ?? "yes";
    const values = [
      "<|startofcontext|>",
      "<|startoftranscript|>",
      "<|emo:undefined|>",
      `<|${fallbackLanguage}|>`,
      `<|${targetLanguage}|>`,
      pnc === "yes" ? "<|pnc|>" : "<|nopnc|>",
      "<|noitn|>",
      "<|notimestamp|>",
      "<|nodiarize|>"
    ];
    return values.map((token) => {
      const id = this.tokenToId.get(token);
      if (id == null) {
        throw new Error(`Required Canary token not found in vocab: ${token}`);
      }
      return id;
    });
  }
  initialMems(batchSize) {
    const shape = this.decoderMemsShapeTemplate.map((dim, index) => {
      if (typeof dim === "number") {
        return dim;
      }
      if (index === 1) {
        return batchSize;
      }
      if (index === 2) {
        return 0;
      }
      return 1;
    });
    return zerosTensor(this.decoderMemsType, shape);
  }
  async transcribeSamples(samples, sampleRate = this.sampleRate, options = {}) {
    let prepared = samples;
    if (!(prepared instanceof Float32Array)) {
      prepared = Float32Array.from(prepared);
    }
    if (sampleRate !== this.sampleRate) {
      prepared = resampleLinear(prepared, sampleRate, this.sampleRate);
    }
    const encoderInputs = this.encoderHelper.prepareInputsFromWaveform(prepared);
    const encoderOutputs = await this.encoderSession.run({
      [this.encoderHelper.audioSignalName]: encoderInputs.signal,
      [this.encoderHelper.lengthName]: encoderInputs.length
    });
    const encoderEmbeddings = encoderOutputs[this.encoderOutputEmbeddingsName];
    const encoderMask = encoderOutputs[this.encoderOutputMaskName];
    if (!encoderEmbeddings || !encoderMask) {
      throw new Error("Canary encoder outputs are missing required tensors.");
    }
    const prefix = this.canaryPrefix(options);
    const batchTokens = [prefix.slice()];
    const prefixLength = batchTokens[0].length;
    const eosId = this.tokenToId.get("<|endoftext|>");
    if (eosId == null) {
      throw new Error("Canary vocab is missing <|endoftext|> token.");
    }
    let decoderMems = this.initialMems(batchTokens.length);
    while (batchTokens[0].length < this.maxSequenceLength) {
      const inputIds = decoderMems.dims[2] === 0 ? batchTokens.flat() : batchTokens.map((row) => row[row.length - 1]);
      const sequenceLength = decoderMems.dims[2] === 0 ? batchTokens[0].length : 1;
      const inputTensor = intTensorFor2(this.decoderInputIdType, inputIds, [batchTokens.length, sequenceLength]);
      const decoderOutputs = await this.decoderSession.run({
        [this.decoderInputIdName]: inputTensor,
        [this.decoderEncoderEmbeddingsName]: encoderEmbeddings,
        [this.decoderEncoderMaskName]: encoderMask,
        [this.decoderMemsName]: decoderMems
      });
      const logits = decoderOutputs[this.logitsName];
      const nextMems = decoderOutputs[this.decoderHiddenStatesName];
      if (!logits || !nextMems) {
        throw new Error("Canary decoder outputs are missing logits or decoder state.");
      }
      decoderMems = nextMems;
      const [batchSize, seq, vocab] = logits.dims;
      let allEos = true;
      for (let b = 0; b < batchSize; b += 1) {
        const offset = b * seq * vocab + (seq - 1) * vocab;
        const nextToken = argmaxSlice(logits.data, offset, vocab);
        batchTokens[b].push(nextToken);
        if (nextToken !== eosId) {
          allEos = false;
        }
      }
      if (allEos) {
        break;
      }
    }
    const tokenIds = batchTokens[0].slice(prefixLength).filter((id) => this.tokens[id] && !this.tokens[id].startsWith("<|"));
    return {
      tokenIds,
      tokenFrames: [],
      words: [],
      text: decodeText(this.tokens, tokenIds)
    };
  }
  async transcribeWavBuffer(arrayBuffer, options = {}) {
    const decoded = decodeWav(arrayBuffer);
    return this.transcribeSamples(decoded.samples, decoded.sampleRate, options);
  }
};
var GigaamRnntModel = class {
  constructor({ config, tokens, encoderSession, decoderSession, jointSession }) {
    this.config = config;
    this.tokens = tokens;
    this.encoderSession = encoderSession;
    this.decoderSession = decoderSession;
    this.jointSession = jointSession;
    this.sampleRate = 16e3;
    this.encoderHelper = new EncoderModel(encoderSession, { config, sampleRate: this.sampleRate });
    this.blankTokenId = detectBlankTokenId(tokens);
    this.maxTokensPerStep = config?.max_tokens_per_step ?? 3;
    this.decoderTargetName = firstExistingName(decoderSession.inputNames, ["targets", "target", "tokens"], 0);
    this.decoderTargetLengthName = decoderSession.inputNames.includes("target_length") ? "target_length" : null;
    this.decoderTargetType = metaForName(decoderSession, this.decoderTargetName)?.type ?? "int32";
    this.decoderTargetLengthType = this.decoderTargetLengthName ? metaForName(decoderSession, this.decoderTargetLengthName)?.type ?? "int32" : "int32";
    this.decoderStateInputNames = decoderSession.inputNames.filter(
      (name) => name !== this.decoderTargetName && name !== this.decoderTargetLengthName
    );
    this.decoderVectorOutputName = decoderSession.outputNames.find(
      (name) => /dec|pred|output/i.test(name)
    ) ?? decoderSession.outputNames[0];
    this.decoderStateOutputNames = decoderSession.outputNames.filter(
      (name) => name !== this.decoderVectorOutputName && !/length/i.test(name)
    );
    this.jointEncoderInputName = firstExistingName(
      jointSession.inputNames,
      ["encoder_outputs", "encoder_output", "enc_out", "encoder"],
      0
    );
    this.jointDecoderInputName = jointSession.inputNames.find((name) => name !== this.jointEncoderInputName) ?? jointSession.inputNames[1];
    this.jointOutputName = jointSession.outputNames[0];
    this.jointEncoderShape = metaForName(jointSession, this.jointEncoderInputName)?.shape ?? [1, 1, 1];
    this.jointDecoderShape = metaForName(jointSession, this.jointDecoderInputName)?.shape ?? [1, 1, 1];
  }
  initialDecoderStates(batchSize = 1) {
    const states = /* @__PURE__ */ new Map();
    for (const inputName of this.decoderStateInputNames) {
      const meta = metaForName(this.decoderSession, inputName);
      if (!meta || meta.type !== "float32") {
        continue;
      }
      const shape = shapeFromMeta(meta, { batch: batchSize });
      states.set(inputName, zerosTensor("float32", shape));
    }
    return states;
  }
  adaptJointInput(tensor, targetShape) {
    if (!Array.isArray(targetShape) || targetShape.length !== 3 || tensor.dims.length !== 3) {
      return tensor;
    }
    const expected1 = typeof targetShape[1] === "number" ? targetShape[1] : tensor.dims[1];
    const expected2 = typeof targetShape[2] === "number" ? targetShape[2] : tensor.dims[2];
    if (tensor.dims[1] === expected1 && tensor.dims[2] === expected2) {
      return tensor;
    }
    if (tensor.dims[1] === expected2 && tensor.dims[2] === expected1) {
      const [b, d1, d2] = tensor.dims;
      const out = new Float32Array(b * d2 * d1);
      for (let bi = 0; bi < b; bi += 1) {
        for (let i = 0; i < d1; i += 1) {
          for (let j = 0; j < d2; j += 1) {
            const src = bi * d1 * d2 + i * d2 + j;
            const dst = bi * d2 * d1 + j * d1 + i;
            out[dst] = tensor.data[src];
          }
        }
      }
      return new ort3.Tensor("float32", out, [b, d2, d1]);
    }
    return tensor;
  }
  runJointArgmax(encoderFrameTensor, decoderVectorTensor) {
    const enc = this.adaptJointInput(encoderFrameTensor, this.jointEncoderShape);
    const dec = this.adaptJointInput(decoderVectorTensor, this.jointDecoderShape);
    return this.jointSession.run({
      [this.jointEncoderInputName]: enc,
      [this.jointDecoderInputName]: dec
    }).then((outputs) => {
      const logits = outputs[this.jointOutputName];
      if (!logits) {
        throw new Error("GigaAM RNNT joint output tensor is missing.");
      }
      return argmaxSlice(logits.data, 0, logits.data.length);
    });
  }
  async transcribeSamples(samples, sampleRate = this.sampleRate) {
    let prepared = samples;
    if (!(prepared instanceof Float32Array)) {
      prepared = Float32Array.from(prepared);
    }
    if (sampleRate !== this.sampleRate) {
      prepared = resampleLinear(prepared, sampleRate, this.sampleRate);
    }
    const encoderInputs = this.encoderHelper.prepareInputsFromWaveform(prepared);
    const encoded = await this.encoderHelper.run(encoderInputs.signal, encoderInputs.length);
    const [batch, channels, time] = encoded.encodedDims;
    if (batch !== 1) {
      throw new Error(`GigaAM RNNT currently expects batch=1, got ${batch}.`);
    }
    const secondsPerFrame = encoded.encodedLength > 0 ? prepared.length / this.sampleRate / encoded.encodedLength : 0;
    let states = this.initialDecoderStates(1);
    let currentToken = this.blankTokenId;
    const tokenIds = [];
    const tokenFrames = [];
    const limit = Math.min(encoded.encodedLength, time);
    for (let t = 0; t < limit; t += 1) {
      const frameData = new Float32Array(channels);
      for (let c = 0; c < channels; c += 1) {
        frameData[c] = encoded.encodedData[c * time + t];
      }
      const frameTensor = new ort3.Tensor("float32", frameData, [1, channels, 1]);
      for (let n = 0; n < this.maxTokensPerStep; n += 1) {
        const feeds = {
          [this.decoderTargetName]: intTensorFor2(this.decoderTargetType, [currentToken], [1, 1])
        };
        if (this.decoderTargetLengthName) {
          feeds[this.decoderTargetLengthName] = intTensorFor2(this.decoderTargetLengthType, [1], [1]);
        }
        for (const [name, tensor] of states.entries()) {
          feeds[name] = tensor;
        }
        const decoderOutputs = await this.decoderSession.run(feeds);
        const decoderVector = decoderOutputs[this.decoderVectorOutputName];
        if (!decoderVector) {
          throw new Error("GigaAM RNNT decoder vector output is missing.");
        }
        const nextStates = new Map(states);
        for (const inputName of this.decoderStateInputNames) {
          const exact = decoderOutputs[inputName];
          if (exact) {
            nextStates.set(inputName, exact);
            continue;
          }
          const normalizedInput = inputName.replace(/\.\d+$/, "");
          const outputName = this.decoderStateOutputNames.find(
            (name) => name === inputName || name.replace(/\.\d+$/, "") === normalizedInput
          );
          if (outputName && decoderOutputs[outputName]) {
            nextStates.set(inputName, decoderOutputs[outputName]);
          }
        }
        const nextToken = await this.runJointArgmax(frameTensor, decoderVector);
        if (nextToken === this.blankTokenId) {
          break;
        }
        tokenIds.push(nextToken);
        tokenFrames.push({ startFrame: t, endFrame: t + 1 });
        currentToken = nextToken;
        states = nextStates;
      }
    }
    return {
      tokenIds,
      tokenFrames,
      words: wordTimestamps(this.tokens, tokenIds, tokenFrames, secondsPerFrame),
      text: decodeText(this.tokens, tokenIds)
    };
  }
  async transcribeWavBuffer(arrayBuffer) {
    const decoded = decodeWav(arrayBuffer);
    return this.transcribeSamples(decoded.samples, decoded.sampleRate);
  }
};
var ToneCtcModel = class {
  constructor({ config, tokens, session }) {
    this.config = config;
    this.tokens = tokens;
    this.session = session;
    this.sampleRate = Number(config?.feature_extraction_params?.sample_rate) || Number(config?.sample_rate) || 8e3;
    this.blankTokenId = Number.isInteger(config?.pad_token_id) ? config.pad_token_id : detectBlankTokenId(tokens);
    this.signalName = session.inputNames[0];
    this.stateName = session.inputNames[1];
    this.logitsName = session.outputNames[0];
    this.nextStateName = session.outputNames[1];
    this.signalMeta = metaForName(session, this.signalName);
    this.stateMeta = metaForName(session, this.stateName);
    this.chunkSamples = typeof this.signalMeta?.shape?.[1] === "number" ? this.signalMeta.shape[1] : 2400;
    this.stateType = this.stateMeta?.type ?? "float16";
    this.stateShape = shapeFromMeta(this.stateMeta, { batch: 1 });
  }
  toIntPcm(samples) {
    const out = new Int32Array(samples.length);
    for (let i = 0; i < samples.length; i += 1) {
      const v = Math.max(-1, Math.min(1, samples[i]));
      out[i] = Math.round(v * 32767);
    }
    return out;
  }
  decodeCtcGreedy(tokens, tokenFrames, sequence, frameStartIndex) {
    let prev = this.blankTokenId;
    for (let i = 0; i < sequence.length; i += 1) {
      const token = sequence[i];
      const t = frameStartIndex + i;
      if (token === this.blankTokenId) {
        prev = this.blankTokenId;
        continue;
      }
      if (token === prev) {
        const last = tokenFrames[tokenFrames.length - 1];
        if (last) {
          last.endFrame = t + 1;
        }
        continue;
      }
      tokens.push(token);
      tokenFrames.push({ startFrame: t, endFrame: t + 1 });
      prev = token;
    }
  }
  async transcribeSamples(samples, sampleRate = this.sampleRate) {
    let prepared = samples;
    if (!(prepared instanceof Float32Array)) {
      prepared = Float32Array.from(prepared);
    }
    if (sampleRate !== this.sampleRate) {
      prepared = resampleLinear(prepared, sampleRate, this.sampleRate);
    }
    const pcm = this.toIntPcm(prepared);
    let state = zerosTensor(this.stateType, this.stateShape);
    const tokenIds = [];
    const tokenFrames = [];
    let frameOffset = 0;
    let consumedSamples = 0;
    while (consumedSamples < pcm.length) {
      const end = Math.min(consumedSamples + this.chunkSamples, pcm.length);
      const chunk = new Int32Array(this.chunkSamples);
      chunk.set(pcm.subarray(consumedSamples, end), 0);
      const signal = new ort3.Tensor("int32", chunk, [1, this.chunkSamples, 1]);
      const outputs = await this.session.run({
        [this.signalName]: signal,
        [this.stateName]: state
      });
      const logprobs = outputs[this.logitsName];
      const nextState = outputs[this.nextStateName];
      if (!logprobs || !nextState) {
        throw new Error("Tone CTC model outputs are missing logprobs or state_next.");
      }
      state = nextState;
      const [batch, timeSteps, vocabSize] = logprobs.dims;
      if (batch !== 1) {
        throw new Error(`Tone CTC currently expects batch=1, got ${batch}.`);
      }
      const frameTokens = [];
      for (let t = 0; t < timeSteps; t += 1) {
        const offset = t * vocabSize;
        frameTokens.push(argmaxSlice(logprobs.data, offset, vocabSize));
      }
      this.decodeCtcGreedy(tokenIds, tokenFrames, frameTokens, frameOffset);
      frameOffset += timeSteps;
      consumedSamples = end;
    }
    const secondsPerFrame = frameOffset > 0 ? prepared.length / this.sampleRate / frameOffset : 0;
    return {
      tokenIds,
      tokenFrames,
      words: wordTimestamps(this.tokens, tokenIds, tokenFrames, secondsPerFrame),
      text: decodeText(this.tokens, tokenIds)
    };
  }
  async transcribeWavBuffer(arrayBuffer) {
    const decoded = decodeWav(arrayBuffer);
    return this.transcribeSamples(decoded.samples, decoded.sampleRate);
  }
};
var SherpaTransducerModel = class {
  constructor({ config, tokens, encoderSession, decoderSession, joinerSession }) {
    this.config = config ?? {};
    this.tokens = tokens;
    this.encoderSession = encoderSession;
    this.decoderSession = decoderSession;
    this.joinerSession = joinerSession;
    this.sampleRate = Number(this.config.sample_rate) || 16e3;
    this.blankTokenId = detectBlankTokenId(tokens);
    this.maxSymbolsPerFrame = this.config.max_tokens_per_step ?? 10;
    this.encoderInputName = encoderSession.inputNames[0];
    this.encoderLengthName = encoderSession.inputNames.length > 1 ? encoderSession.inputNames[1] : null;
    this.encoderInputMeta = metaForName(encoderSession, this.encoderInputName);
    this.encoderLengthMeta = this.encoderLengthName ? metaForName(encoderSession, this.encoderLengthName) : null;
    this.encoderOutputName = encoderSession.outputNames[0];
    this.encoderLengthOutName = encoderSession.outputNames.length > 1 ? encoderSession.outputNames[1] : null;
    this.decoderInputName = decoderSession.inputNames[0];
    this.decoderInputMeta = metaForName(decoderSession, this.decoderInputName);
    this.decoderOutputName = decoderSession.outputNames[0];
    const contextDim = this.decoderInputMeta?.shape?.[1];
    this.contextSize = typeof contextDim === "number" && contextDim > 0 ? contextDim : 1;
    this.decoderInputType = this.decoderInputMeta?.type ?? "int64";
    this.joinerEncName = joinerSession.inputNames[0];
    this.joinerDecName = joinerSession.inputNames[1];
    this.joinerOutputName = joinerSession.outputNames[0];
    this.joinerEncShape = metaForName(joinerSession, this.joinerEncName)?.shape ?? [1, 1, 1];
    this.joinerDecShape = metaForName(joinerSession, this.joinerDecName)?.shape ?? [1, 1, 1];
    this.encoderFeatureConfig = {
      model_type: "sherpa-transducer",
      features_size: Number(this.encoderInputMeta?.shape?.[1]) || 80,
      feature_extraction_params: {
        sample_rate: this.sampleRate,
        n_mels: Number(this.encoderInputMeta?.shape?.[1]) || 80,
        n_fft: 400,
        window_size: 0.025,
        window_stride: 0.01,
        preemphasis_coefficient: 0.97
      }
    };
    this.encoderHelper = new EncoderModel(encoderSession, {
      config: this.encoderFeatureConfig,
      sampleRate: this.sampleRate
    });
  }
  adaptTensorToShape(tensor, targetShape) {
    if (!Array.isArray(targetShape)) {
      return tensor;
    }
    if (targetShape.length !== tensor.dims.length) {
      if (tensor.dims.length === 3 && targetShape.length === 2 && tensor.dims[0] === 1) {
        const flat = Float32Array.from(tensor.data);
        if (tensor.dims[1] === 1) {
          return new ort3.Tensor("float32", flat, [1, tensor.dims[2]]);
        }
        if (tensor.dims[2] === 1) {
          return new ort3.Tensor("float32", flat, [1, tensor.dims[1]]);
        }
      }
      if (tensor.dims.length === 2 && targetShape.length === 3 && tensor.dims[0] === 1) {
        const d = tensor.dims[1];
        if (targetShape[1] === 1 || targetShape[2] === d) {
          return new ort3.Tensor("float32", Float32Array.from(tensor.data), [1, 1, d]);
        }
        return new ort3.Tensor("float32", Float32Array.from(tensor.data), [1, d, 1]);
      }
      return tensor;
    }
    if (tensor.dims.every((d, i) => typeof targetShape[i] !== "number" || targetShape[i] < 0 || d === targetShape[i])) {
      return tensor;
    }
    if (tensor.dims.length === 3 && tensor.dims[1] === targetShape[2] && tensor.dims[2] === targetShape[1]) {
      const [b, d1, d2] = tensor.dims;
      const out = new Float32Array(b * d2 * d1);
      for (let bi = 0; bi < b; bi += 1) {
        for (let i = 0; i < d1; i += 1) {
          for (let j = 0; j < d2; j += 1) {
            out[bi * d2 * d1 + j * d1 + i] = tensor.data[bi * d1 * d2 + i * d2 + j];
          }
        }
      }
      return new ort3.Tensor("float32", out, [b, d2, d1]);
    }
    return tensor;
  }
  makeDecoderInput(context) {
    return intTensorFor2(this.decoderInputType, context, [1, this.contextSize]);
  }
  async runEncoder(samples) {
    const inputRank = this.encoderInputMeta?.shape?.length ?? 2;
    if (inputRank === 2) {
      const signal2 = new ort3.Tensor("float32", samples, [1, samples.length]);
      const lenType = this.encoderLengthMeta?.type ?? "int64";
      const length = intTensorFor2(lenType, [samples.length], [1]);
      const outputs2 = await this.encoderSession.run({
        [this.encoderInputName]: signal2,
        ...this.encoderLengthName ? { [this.encoderLengthName]: length } : {}
      });
      return outputs2;
    }
    const prepared = this.encoderHelper.prepareInputsFromWaveform(samples);
    let signal = prepared.signal;
    const inputShape = this.encoderInputMeta?.shape ?? [];
    if (signal.dims.length === 3 && inputShape.length === 3) {
      const featureBins = signal.dims[1];
      const secondDim = inputShape[1];
      const thirdDim = inputShape[2];
      const expectsBtf = typeof thirdDim === "number" && thirdDim === featureBins && secondDim !== featureBins;
      if (expectsBtf) {
        const time = signal.dims[2];
        const transposed = new Float32Array(time * featureBins);
        for (let f = 0; f < featureBins; f += 1) {
          for (let t = 0; t < time; t += 1) {
            transposed[t * featureBins + f] = signal.data[f * time + t];
          }
        }
        signal = new ort3.Tensor("float32", transposed, [1, time, featureBins]);
      }
    }
    const outputs = await this.encoderSession.run({
      [this.encoderInputName]: signal,
      ...this.encoderLengthName ? { [this.encoderLengthName]: prepared.length } : {}
    });
    return outputs;
  }
  async transcribeSamples(samples, sampleRate = this.sampleRate) {
    let prepared = samples;
    if (!(prepared instanceof Float32Array)) {
      prepared = Float32Array.from(prepared);
    }
    if (sampleRate !== this.sampleRate) {
      prepared = resampleLinear(prepared, sampleRate, this.sampleRate);
    }
    const encoderOutputs = await this.runEncoder(prepared);
    const enc = encoderOutputs[this.encoderOutputName];
    const encLen = this.encoderLengthOutName ? encoderOutputs[this.encoderLengthOutName] : null;
    if (!enc) {
      throw new Error("Sherpa encoder output is missing.");
    }
    let channels;
    let time;
    let bctData;
    if (enc.dims.length !== 3) {
      throw new Error(`Unexpected Sherpa encoder output shape: [${enc.dims.join(", ")}].`);
    }
    if (enc.dims[0] !== 1) {
      throw new Error(`Sherpa transducer currently expects batch=1, got ${enc.dims[0]}.`);
    }
    if (typeof this.joinerEncShape?.[1] === "number" && enc.dims[1] === this.joinerEncShape[1]) {
      channels = enc.dims[1];
      time = enc.dims[2];
      bctData = enc.data;
    } else {
      time = enc.dims[1];
      channels = enc.dims[2];
      bctData = new Float32Array(channels * time);
      for (let t = 0; t < time; t += 1) {
        for (let c = 0; c < channels; c += 1) {
          bctData[c * time + t] = enc.data[t * channels + c];
        }
      }
    }
    const encodedLength = encLen ? Number(Array.from(encLen.data, (v) => typeof v === "bigint" ? Number(v) : v)[0]) : time;
    const limit = Math.min(time, encodedLength);
    const tokenIds = [];
    const tokenFrames = [];
    const context = new Array(this.contextSize).fill(this.blankTokenId);
    for (let t = 0; t < limit; t += 1) {
      const frame = new Float32Array(channels);
      for (let c = 0; c < channels; c += 1) {
        frame[c] = bctData[c * time + t];
      }
      let encTensor = new ort3.Tensor("float32", frame, [1, channels, 1]);
      encTensor = this.adaptTensorToShape(encTensor, this.joinerEncShape);
      for (let n = 0; n < this.maxSymbolsPerFrame; n += 1) {
        const decOutputs = await this.decoderSession.run({
          [this.decoderInputName]: this.makeDecoderInput(context)
        });
        let decTensor = decOutputs[this.decoderOutputName];
        if (!decTensor) {
          throw new Error("Sherpa decoder output is missing.");
        }
        decTensor = this.adaptTensorToShape(decTensor, this.joinerDecShape);
        const jointOutputs = await this.joinerSession.run({
          [this.joinerEncName]: encTensor,
          [this.joinerDecName]: decTensor
        });
        const logits = jointOutputs[this.joinerOutputName];
        if (!logits) {
          throw new Error("Sherpa joiner output is missing.");
        }
        const nextToken = argmaxSlice(logits.data, 0, logits.data.length);
        if (nextToken === this.blankTokenId) {
          break;
        }
        tokenIds.push(nextToken);
        tokenFrames.push({ startFrame: t, endFrame: t + 1 });
        context.shift();
        context.push(nextToken);
      }
    }
    const secondsPerFrame = limit > 0 ? prepared.length / this.sampleRate / limit : 0;
    return {
      tokenIds,
      tokenFrames,
      words: wordTimestamps(this.tokens, tokenIds, tokenFrames, secondsPerFrame),
      text: decodeText(this.tokens, tokenIds)
    };
  }
  async transcribeWavBuffer(arrayBuffer) {
    const decoded = decodeWav(arrayBuffer);
    return this.transcribeSamples(decoded.samples, decoded.sampleRate);
  }
};
async function createAsrModel({
  modelType,
  decoderKind,
  config,
  preprocessorModel,
  encoderModel,
  decoderModel,
  decoderJointModel,
  whisperModel,
  vocabularyText,
  vocabJson,
  addedTokensJson,
  sessionOptions,
  decoderOptions
} = {}) {
  if (!modelType || !decoderKind) {
    throw new Error(
      "createAsrModel expects modelType and decoderKind."
    );
  }
  const configuredSampleRate = Number(config?.feature_extraction_params?.sample_rate) || Number(config?.sample_rate) || 16e3;
  if (decoderKind === "whisper-ort") {
    if (!whisperModel || !vocabJson) {
      throw new Error("whisper-ort requires whisperModel and vocabJson.");
    }
    const session = await ort3.InferenceSession.create(whisperModel, sessionOptions);
    return new WhisperOrtModel({
      config,
      vocab: JSON.parse(vocabJson),
      addedTokens: addedTokensJson ? JSON.parse(addedTokensJson) : {},
      session
    });
  }
  if (decoderKind === "whisper-hf") {
    if (!encoderModel || !decoderJointModel || !vocabJson) {
      throw new Error("whisper requires encoderModel, decoderJointModel, and vocabJson.");
    }
    const [encoderSession2, decoderSession2] = await Promise.all([
      ort3.InferenceSession.create(encoderModel, sessionOptions),
      ort3.InferenceSession.create(decoderJointModel, sessionOptions)
    ]);
    return new WhisperHfModel({
      config,
      vocab: JSON.parse(vocabJson),
      addedTokens: addedTokensJson ? JSON.parse(addedTokensJson) : {},
      encoderSession: encoderSession2,
      decoderSession: decoderSession2
    });
  }
  if (decoderKind === "aed") {
    if (!encoderModel || !decoderJointModel || !vocabularyText) {
      throw new Error("nemo-conformer-aed requires encoderModel, decoderModel, and vocabularyText.");
    }
    const [encoderSession2, decoderSession2] = await Promise.all([
      ort3.InferenceSession.create(encoderModel, sessionOptions),
      ort3.InferenceSession.create(decoderJointModel, sessionOptions)
    ]);
    const tokens2 = parseVocabulary(vocabularyText);
    return new NemoAedModel({
      config,
      tokens: tokens2,
      encoderSession: encoderSession2,
      decoderSession: decoderSession2
    });
  }
  if (decoderKind === "gigaam-rnnt") {
    if (!encoderModel || !decoderModel || !decoderJointModel || !vocabularyText) {
      throw new Error(
        "gigaam rnnt requires encoderModel, decoderModel, jointModel, and vocabularyText."
      );
    }
    const [encoderSession2, decoderSession2, jointSession] = await Promise.all([
      ort3.InferenceSession.create(encoderModel, sessionOptions),
      ort3.InferenceSession.create(decoderModel, sessionOptions),
      ort3.InferenceSession.create(decoderJointModel, sessionOptions)
    ]);
    const tokens2 = parseVocabulary(vocabularyText);
    return new GigaamRnntModel({
      config,
      tokens: tokens2,
      encoderSession: encoderSession2,
      decoderSession: decoderSession2,
      jointSession
    });
  }
  if (decoderKind === "tone-ctc") {
    if (!encoderModel || !vocabularyText) {
      throw new Error("tone-ctc requires model.onnx and vocabulary.");
    }
    const session = await ort3.InferenceSession.create(encoderModel, sessionOptions);
    const tokens2 = parseVocabulary(vocabularyText);
    return new ToneCtcModel({
      config,
      tokens: tokens2,
      session
    });
  }
  if (decoderKind === "sherpa-transducer") {
    if (!encoderModel || !decoderModel || !decoderJointModel || !vocabularyText) {
      throw new Error("sherpa-transducer requires encoder, decoder, joiner, and tokens.");
    }
    const [encoderSession2, decoderSession2, joinerSession] = await Promise.all([
      ort3.InferenceSession.create(encoderModel, sessionOptions),
      ort3.InferenceSession.create(decoderModel, sessionOptions),
      ort3.InferenceSession.create(decoderJointModel, sessionOptions)
    ]);
    const tokens2 = parseVocabulary(vocabularyText);
    return new SherpaTransducerModel({
      config,
      tokens: tokens2,
      encoderSession: encoderSession2,
      decoderSession: decoderSession2,
      joinerSession
    });
  }
  if (!encoderModel || !vocabularyText) {
    throw new Error("createAsrModel expects encoderModel and vocabularyText for non-whisper models.");
  }
  if (decoderKind !== "ctc" && !decoderJointModel) {
    throw new Error("Transducer models require decoderJointModel.");
  }
  const sessionPromises = [ort3.InferenceSession.create(encoderModel, sessionOptions)];
  if (decoderJointModel) {
    sessionPromises.push(ort3.InferenceSession.create(decoderJointModel, sessionOptions));
  }
  if (preprocessorModel) {
    sessionPromises.unshift(ort3.InferenceSession.create(preprocessorModel, sessionOptions));
  }
  const sessions = await Promise.all(sessionPromises);
  const preprocessorSession = preprocessorModel ? sessions[0] : null;
  const encoderSession = preprocessorModel ? sessions[1] : sessions[0];
  const decoderSession = preprocessorModel ? sessions[2] : sessions[1];
  const tokens = parseVocabulary(vocabularyText);
  const blankTokenId = detectBlankTokenId(tokens);
  const maxSymbols = config?.max_tokens_per_step ?? decoderOptions?.maxSymbols ?? 10;
  if (decoderKind === "ctc") {
    const blankTokenId2 = Number.isInteger(config?.pad_token_id) ? config.pad_token_id : detectBlankTokenId(tokens);
    return new AsrModel({
      preprocessor: preprocessorSession ? new PreprocessorModel(preprocessorSession) : null,
      encoder: new CtcAcousticModel(encoderSession, {
        config,
        sampleRate: configuredSampleRate,
        vocabSize: tokens.length
      }),
      decoder: new CtcGreedyDecoder({
        blankTokenId: blankTokenId2
      }),
      tokens,
      sampleRate: configuredSampleRate
    });
  }
  return new AsrModel({
    preprocessor: preprocessorSession ? new PreprocessorModel(preprocessorSession) : null,
    encoder: new EncoderModel(encoderSession, { config, sampleRate: configuredSampleRate }),
    decoder: new TransducerGreedyDecoder(
      new DecoderTransducerModel(decoderSession, {
        decoderKind,
        vocabSize: tokens.length
      }),
      {
        blankTokenId,
        maxSymbols,
        ...decoderOptions
      }
    ),
    tokens,
    sampleRate: configuredSampleRate
  });
}

// src/model-types.js
var MODEL_TYPES = {
  "nemo-conformer-tdt": {
    decoderKind: "tdt",
    preprocessor: "nemo128.onnx",
    encoder: "encoder-model.onnx",
    decoderJoint: "decoder_joint-model.onnx",
    vocabCandidates: ["tokens.txt", "vocab.txt"]
  },
  "nemo-conformer-rnnt": {
    decoderKind: "rnnt",
    preprocessor: null,
    encoder: "encoder-model.onnx",
    decoderJoint: "decoder_joint-model.onnx",
    vocabCandidates: ["vocab.txt", "tokens.txt"]
  },
  "nemo-conformer-ctc": {
    decoderKind: "ctc",
    preprocessor: null,
    encoder: "model.onnx",
    decoderJoint: null,
    vocabCandidates: ["vocab.txt", "tokens.txt"]
  },
  "nemo-conformer-aed": {
    decoderKind: "aed",
    preprocessor: null,
    encoder: "encoder-model.onnx",
    decoderJoint: "decoder-model.onnx",
    vocabCandidates: ["vocab.txt", "tokens.txt"]
  },
  "nemo-conformer": {
    decoderKind: "nemo-conformer",
    preprocessor: null,
    encoder: "encoder-model.onnx",
    decoderJoint: "decoder_joint-model.onnx",
    vocabCandidates: ["vocab.txt", "tokens.txt"]
  },
  gigaam: {
    decoderKind: "gigaam",
    preprocessor: null,
    encoder: null,
    decoderJoint: null,
    vocabCandidates: ["v3_vocab.txt", "v2_vocab.txt", "vocab.txt", "tokens.txt"]
  },
  "whisper-ort": {
    decoderKind: "whisper-ort",
    preprocessor: null,
    encoder: null,
    decoderJoint: null,
    whisperModelPattern: /_beamsearch(?:\\.int8)?\\.onnx$/
  },
  whisper: {
    decoderKind: "whisper-hf",
    preprocessor: null,
    encoder: "onnx/encoder_model.onnx",
    decoderJoint: "onnx/decoder_model_merged.onnx",
    vocabCandidates: []
  },
  "tone-ctc": {
    decoderKind: "tone-ctc",
    preprocessor: null,
    encoder: "model.onnx",
    decoderJoint: null,
    vocabCandidates: ["vocab.txt", "tokens.txt", "vocab.json"]
  }
};
function parseConfigText(configText) {
  let parsed;
  try {
    parsed = JSON.parse(configText);
  } catch (error) {
    throw new Error(`Invalid config.json: ${error}`);
  }
  if (!parsed || typeof parsed !== "object") {
    throw new Error("Invalid config.json: expected a JSON object.");
  }
  return parsed;
}
function detectModelType(config) {
  const modelType = config?.model_type;
  if (!modelType || typeof modelType !== "string") {
    const architectures = Array.isArray(config?.architectures) ? config.architectures : [];
    if (architectures.includes("ToneForCTC")) {
      return { modelType: "tone-ctc", spec: MODEL_TYPES["tone-ctc"] };
    }
    throw new Error("config.json is missing string field 'model_type'.");
  }
  const spec = MODEL_TYPES[modelType];
  if (!spec) {
    throw new Error(`Unsupported model_type: ${modelType}`);
  }
  return { modelType, spec };
}
function toneVocabularyTextFromConfig(config) {
  const vocab = config?.decoder_params?.vocabulary;
  if (!Array.isArray(vocab) || vocab.length === 0) {
    return null;
  }
  const lines = vocab.map((token, index) => {
    const normalized = token === " " ? "\u2581" : token;
    return `${normalized} ${index}`;
  });
  const pad = config?.pad_token_id;
  if (typeof pad === "number" && pad >= vocab.length) {
    lines.push(`<blank> ${pad}`);
  }
  return `${lines.join("\n")}
`;
}

// src/vad.js
import * as ort4 from "onnxruntime-web";
function firstExistingName2(names, candidates, fallbackIndex = 0) {
  for (const candidate of candidates) {
    if (names.includes(candidate)) {
      return candidate;
    }
  }
  return names[fallbackIndex];
}
function int64Scalar(value) {
  return new ort4.Tensor("int64", new BigInt64Array([BigInt(value)]), []);
}
function joinText(parts) {
  return parts.filter((text) => typeof text === "string" && text.trim().length > 0).join(" ").replace(/\s+/g, " ").trim();
}
function toFloat32(samples) {
  if (samples instanceof Float32Array) {
    return samples;
  }
  return Float32Array.from(samples);
}
var SileroVadModel = class {
  constructor(session, options = {}) {
    this.session = session;
    this.sampleRate = Number(options.sampleRate) || 16e3;
    this.threshold = options.threshold ?? 0.5;
    this.negThreshold = options.negThreshold ?? Math.max(0.05, this.threshold - 0.15);
    this.minSpeechMs = options.minSpeechMs ?? 250;
    this.minSilenceMs = options.minSilenceMs ?? 700;
    this.speechPadMs = options.speechPadMs ?? 200;
    this.windowSamples = Number(options.windowSamples) || (this.sampleRate === 8e3 ? 256 : 512);
    this.inputName = firstExistingName2(session.inputNames, ["input", "x"], 0);
    this.stateInputName = firstExistingName2(session.inputNames, ["state"], 1);
    this.srInputName = firstExistingName2(session.inputNames, ["sr", "sample_rate"], 2);
    this.outputName = firstExistingName2(session.outputNames, ["output", "prob", "speech_prob"], 0);
    this.stateOutputName = firstExistingName2(session.outputNames, ["stateN", "state", "new_state"], 1);
  }
  initialState() {
    return new ort4.Tensor("float32", new Float32Array(2 * 128), [2, 1, 128]);
  }
  async speechProbabilities(samples, sampleRate = this.sampleRate) {
    let prepared = toFloat32(samples);
    if (sampleRate !== this.sampleRate) {
      prepared = resampleLinear(prepared, sampleRate, this.sampleRate);
    }
    const probs = [];
    let state = this.initialState();
    let offset = 0;
    while (offset < prepared.length) {
      const end = Math.min(offset + this.windowSamples, prepared.length);
      const chunk = new Float32Array(this.windowSamples);
      chunk.set(prepared.subarray(offset, end), 0);
      const outputs = await this.session.run({
        [this.inputName]: new ort4.Tensor("float32", chunk, [1, this.windowSamples]),
        [this.stateInputName]: state,
        [this.srInputName]: int64Scalar(this.sampleRate)
      });
      const prob = outputs[this.outputName];
      const nextState = outputs[this.stateOutputName];
      if (!prob || !nextState) {
        throw new Error("Silero VAD outputs are missing probability or next state.");
      }
      probs.push(Number(prob.data[0] ?? 0));
      state = nextState;
      offset = end;
    }
    return { probs, processedSamples: prepared.length };
  }
  /**
   * Returns speech segments in input sample-rate coordinates.
   * Segment bounds are [start, end), in samples.
   */
  async detectSpeechSegments(samples, sampleRate = this.sampleRate, overrides = {}) {
    const threshold = overrides.threshold ?? this.threshold;
    const negThreshold = overrides.negThreshold ?? this.negThreshold;
    const minSpeechSamples = Math.round((overrides.minSpeechMs ?? this.minSpeechMs) / 1e3 * this.sampleRate);
    const minSilenceSamples = Math.round((overrides.minSilenceMs ?? this.minSilenceMs) / 1e3 * this.sampleRate);
    const speechPadSamples = Math.round((overrides.speechPadMs ?? this.speechPadMs) / 1e3 * this.sampleRate);
    const prepared = toFloat32(samples);
    const ratioToVad = sampleRate / this.sampleRate;
    const { probs, processedSamples } = await this.speechProbabilities(prepared, sampleRate);
    const maxVadSamples = sampleRate === this.sampleRate ? prepared.length : Math.round(processedSamples * ratioToVad);
    const raw = [];
    let activeStart = -1;
    let pendingEnd = -1;
    for (let i = 0; i < probs.length; i += 1) {
      const frameStart = i * this.windowSamples;
      const p = probs[i];
      if (p >= threshold) {
        if (activeStart < 0) {
          activeStart = frameStart;
        }
        pendingEnd = -1;
        continue;
      }
      if (activeStart < 0) {
        continue;
      }
      if (p <= negThreshold && pendingEnd < 0) {
        pendingEnd = frameStart;
      }
      if (pendingEnd >= 0 && frameStart - pendingEnd >= minSilenceSamples) {
        if (pendingEnd - activeStart >= minSpeechSamples) {
          raw.push({ start: activeStart, end: pendingEnd });
        }
        activeStart = -1;
        pendingEnd = -1;
      }
    }
    if (activeStart >= 0) {
      const end = probs.length * this.windowSamples;
      if (end - activeStart >= minSpeechSamples) {
        raw.push({ start: activeStart, end });
      }
    }
    if (raw.length === 0) {
      return [];
    }
    const padded = raw.map((segment) => ({
      start: Math.max(0, segment.start - speechPadSamples),
      end: segment.end + speechPadSamples
    }));
    const merged = [];
    for (const segment of padded) {
      const last = merged[merged.length - 1];
      if (!last || segment.start > last.end + minSilenceSamples) {
        merged.push({ ...segment });
      } else {
        last.end = Math.max(last.end, segment.end);
      }
    }
    return merged.map((segment) => {
      const startSec = segment.start / this.sampleRate;
      const endSec = segment.end / this.sampleRate;
      const start = Math.max(0, Math.round(startSec * sampleRate));
      const end = Math.min(maxVadSamples, Math.round(endSec * sampleRate));
      return {
        start,
        end: Math.max(start + 1, end),
        startSec: start / sampleRate,
        endSec: Math.max(start + 1, end) / sampleRate
      };
    });
  }
};
var VadChunkedAsrModel = class {
  constructor(baseModel, vadModel, options = {}) {
    this.baseModel = baseModel;
    this.vadModel = vadModel;
    this.options = options;
    this.sampleRate = baseModel.sampleRate ?? vadModel.sampleRate ?? 16e3;
  }
  async transcribeSamples(samples, sampleRate = this.sampleRate, options = {}) {
    const prepared = toFloat32(samples);
    const vadOptions = options.vadOptions ?? this.options;
    const segments = await this.vadModel.detectSpeechSegments(prepared, sampleRate, vadOptions);
    if (segments.length === 0) {
      return {
        tokenIds: [],
        tokenFrames: [],
        words: [],
        text: "",
        segments: []
      };
    }
    const tokenIds = [];
    const words = [];
    const texts = [];
    for (const segment of segments) {
      const chunk = prepared.subarray(segment.start, segment.end);
      const chunkResult = await this.baseModel.transcribeSamples(chunk, sampleRate, options);
      tokenIds.push(...chunkResult.tokenIds ?? []);
      if (chunkResult.text) {
        texts.push(chunkResult.text);
      }
      for (const word of chunkResult.words ?? []) {
        words.push({
          word: word.word,
          start: Number((word.start + segment.startSec).toFixed(3)),
          end: Number((word.end + segment.startSec).toFixed(3))
        });
      }
    }
    return {
      tokenIds,
      tokenFrames: [],
      words,
      text: joinText(texts),
      segments
    };
  }
  async transcribeWavBuffer(arrayBuffer, options = {}) {
    const decoded = decodeWav(arrayBuffer);
    return this.transcribeSamples(decoded.samples, decoded.sampleRate, options);
  }
};
function withVadModel(asrModel, vadModel, options = {}) {
  if (!vadModel || typeof vadModel.detectSpeechSegments !== "function") {
    throw new Error("Invalid VAD model: expected detectSpeechSegments(samples, sampleRate, options).");
  }
  return new VadChunkedAsrModel(asrModel, vadModel, options);
}
async function createSileroVadModel({
  modelPath,
  sessionOptions,
  options = {}
} = {}) {
  if (!modelPath) {
    throw new Error("createSileroVadModel expects modelPath.");
  }
  const session = await ort4.InferenceSession.create(modelPath, sessionOptions);
  return new SileroVadModel(session, options);
}

// src/browser.js
function modelFilenameCandidates(filename, quantization = "int8") {
  if (!filename.endsWith(".onnx")) {
    return [filename];
  }
  if (quantization !== "int8") {
    return [filename];
  }
  const dotInt8 = filename.replace(/\.onnx$/, ".int8.onnx");
  const underscoreInt8 = filename.replace(/\.onnx$/, "_int8.onnx");
  return [.../* @__PURE__ */ new Set([dotInt8, underscoreInt8, filename])];
}
function joinUrl(baseUrl, file) {
  const withSlash = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const pageHref = typeof globalThis.location?.href === "string" ? globalThis.location.href : "http://localhost/";
  const resolvedBase = new URL(withSlash, pageHref);
  return new URL(file, resolvedBase).toString();
}
async function fetchText(url, fetchImpl, headers) {
  const response = await fetchImpl(url, { headers });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}
async function fetchTextOptional(url, fetchImpl, headers) {
  const response = await fetchImpl(url, { headers });
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}
async function probeUrl(url, fetchImpl, headers) {
  const head = await fetchImpl(url, { method: "HEAD", headers });
  if (head.ok) {
    return true;
  }
  if (head.status === 404) {
    return false;
  }
  const get = await fetchImpl(url, { headers });
  if (get.ok) {
    return true;
  }
  if (get.status === 404) {
    return false;
  }
  throw new Error(`Failed to probe ${url}: ${get.status} ${get.statusText}`);
}
async function resolveModelUrl(baseUrl, filename, options) {
  const { fetchImpl, headers, quantization } = options;
  const candidates = modelFilenameCandidates(filename, quantization);
  for (const candidate of candidates) {
    const url = joinUrl(baseUrl, candidate);
    if (await probeUrl(url, fetchImpl, headers)) {
      return url;
    }
  }
  return joinUrl(baseUrl, candidates[candidates.length - 1]);
}
async function resolveFirstExistingUrl(baseUrl, candidates, fetchImpl, headers) {
  for (const candidate of candidates) {
    const url = joinUrl(baseUrl, candidate);
    if (await probeUrl(url, fetchImpl, headers)) {
      return url;
    }
  }
  throw new Error(`Missing file. Checked: ${candidates.join(", ")}`);
}
async function fetchFirstExistingText(baseUrl, candidates, fetchImpl, headers) {
  const url = await resolveFirstExistingUrl(baseUrl, candidates, fetchImpl, headers);
  return fetchText(url, fetchImpl, headers);
}
function vocabJsonToText(vocabJsonText) {
  const parsed = JSON.parse(vocabJsonText);
  const entries = Object.entries(parsed).filter(([, id]) => Number.isInteger(id)).sort((a, b) => a[1] - b[1]);
  if (entries.length === 0) {
    throw new Error("Invalid vocab.json mapping.");
  }
  return `${entries.map(([token, id]) => `${token} ${id}`).join("\n")}
`;
}
async function listHuggingFaceRepoFiles(repoId, revision, endpoint, fetchImpl, headers) {
  const base = endpoint.replace(/\/$/, "");
  const url = `${base}/api/models/${repoId}/tree/${encodeURIComponent(revision)}?recursive=1`;
  const response = await fetchImpl(url, { headers });
  if (!response.ok) {
    throw new Error(`Failed to list Hugging Face repo files: ${response.status} ${response.statusText}`);
  }
  const payload = await response.json();
  return new Set(payload.map((item) => item.path).filter((path) => typeof path === "string"));
}
function pickWhisperBeamsearchFile(files, quantization) {
  const list = [...files].filter((path) => /_beamsearch(?:\.int8)?\.onnx$/.test(path));
  if (list.length === 0) {
    return null;
  }
  if (quantization === "int8") {
    const int8 = list.find((name) => /(?:\.int8|_int8)\.onnx$/.test(name));
    if (int8) {
      return int8;
    }
  }
  return list.find((name) => !/(?:\.int8|_int8)\.onnx$/.test(name)) ?? list[0];
}
async function resolveGigaamFromBase(baseUrl, config, quantization, fetchImpl, headers) {
  const version = config.version ?? "v2";
  const vocabName = `${version}_vocab.txt`;
  const vocab = await fetchFirstExistingText(baseUrl, [vocabName, "vocab.txt", "tokens.txt"], fetchImpl, headers);
  const rnntPrefixes = [`${version}_rnnt`, `${version}_e2e_rnnt`];
  for (const prefix of rnntPrefixes) {
    const encoder = await resolveFirstExistingUrl(
      baseUrl,
      modelFilenameCandidates(`${prefix}_encoder.onnx`, quantization),
      fetchImpl,
      headers
    ).catch(() => null);
    const decoder = await resolveFirstExistingUrl(
      baseUrl,
      modelFilenameCandidates(`${prefix}_decoder.onnx`, quantization),
      fetchImpl,
      headers
    ).catch(() => null);
    const joint = await resolveFirstExistingUrl(
      baseUrl,
      modelFilenameCandidates(`${prefix}_joint.onnx`, quantization),
      fetchImpl,
      headers
    ).catch(() => null);
    if (encoder && decoder && joint) {
      return {
        mode: "rnnt",
        encoder,
        decoder,
        joint,
        vocabularyText: vocab
      };
    }
  }
  const ctcCandidates = [`${version}_ctc.onnx`, `${version}_e2e_ctc.onnx`];
  for (const candidate of ctcCandidates) {
    const model = await resolveFirstExistingUrl(
      baseUrl,
      modelFilenameCandidates(candidate, quantization),
      fetchImpl,
      headers
    ).catch(() => null);
    if (model) {
      return {
        mode: "ctc",
        ctcModel: model,
        vocabularyText: vocab
      };
    }
  }
  throw new Error("Could not resolve GigaAM RNNT or CTC model files.");
}
async function resolveNemoConformerFromBase(baseUrl, quantization, fetchImpl, headers) {
  const encoder = await resolveFirstExistingUrl(
    baseUrl,
    modelFilenameCandidates("encoder-model.onnx", quantization),
    fetchImpl,
    headers
  ).catch(() => null);
  const decoderJoint = await resolveFirstExistingUrl(
    baseUrl,
    modelFilenameCandidates("decoder_joint-model.onnx", quantization),
    fetchImpl,
    headers
  ).catch(() => null);
  if (encoder && decoderJoint) {
    return { mode: "rnnt", encoder, decoderJoint };
  }
  const ctcModel = await resolveFirstExistingUrl(
    baseUrl,
    modelFilenameCandidates("model.onnx", quantization),
    fetchImpl,
    headers
  ).catch(() => null);
  if (ctcModel) {
    return { mode: "ctc", ctcModel };
  }
  throw new Error("Could not resolve nemo-conformer RNNT or CTC artifacts.");
}
async function resolveSherpaFromBase(baseUrl, quantization, fetchImpl, headers) {
  const amOnnxProbe = modelFilenameCandidates("am-onnx/encoder.onnx", quantization);
  const amProbe = modelFilenameCandidates("am/encoder.onnx", quantization);
  const amDir = await resolveFirstExistingUrl(baseUrl, [...amOnnxProbe, ...amProbe], fetchImpl, headers).then((url) => url.includes("/am-onnx/") ? "am-onnx" : "am").catch(() => null);
  if (!amDir) {
    return null;
  }
  const encoder = await resolveFirstExistingUrl(
    baseUrl,
    modelFilenameCandidates(`${amDir}/encoder.onnx`, quantization),
    fetchImpl,
    headers
  );
  const decoder = await resolveFirstExistingUrl(
    baseUrl,
    modelFilenameCandidates(`${amDir}/decoder.onnx`, quantization),
    fetchImpl,
    headers
  );
  const joiner = await resolveFirstExistingUrl(
    baseUrl,
    modelFilenameCandidates(`${amDir}/joiner.onnx`, quantization),
    fetchImpl,
    headers
  );
  const tokensPath = await resolveFirstExistingUrl(baseUrl, ["lang/tokens.txt", "tokens.txt"], fetchImpl, headers);
  const tokensText = await fetchText(tokensPath, fetchImpl, headers);
  return { encoder, decoder, joiner, tokensText };
}
function vadModelCandidates(quantization = "int8") {
  if (quantization && quantization !== "none" && quantization !== "float32" && quantization !== "fp32") {
    return [`onnx/model_${quantization}.onnx`, "onnx/model.onnx", "model.onnx"];
  }
  return ["onnx/model.onnx", "onnx/model_int8.onnx", "model.onnx"];
}
function resolveVadOption(options = {}) {
  return options.vadModel ?? options.vad ?? null;
}
function attachVadIfProvided(asrModel, options = {}) {
  const vadModel = resolveVadOption(options);
  if (!vadModel) {
    return asrModel;
  }
  return withVadModel(asrModel, vadModel, options.vadOptions);
}
async function createAsrModelWithVad(params, options = {}) {
  const asrModel = await createAsrModel(params);
  return attachVadIfProvided(asrModel, options);
}
async function loadLocalVadModel(baseUrl, options = {}) {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new Error("No fetch implementation available.");
  }
  const quantization = options.quantization ?? "int8";
  const headers = options.headers;
  const modelUrl = await resolveFirstExistingUrl(baseUrl, vadModelCandidates(quantization), fetchImpl, headers);
  return createSileroVadModel({
    modelPath: modelUrl,
    sessionOptions: options.sessionOptions,
    options: options.vadOptions
  });
}
async function loadLocalModel(baseUrl, options = {}) {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new Error("No fetch implementation available.");
  }
  const quantization = options.quantization ?? "int8";
  const headers = options.headers;
  const configText = await fetchTextOptional(joinUrl(baseUrl, "config.json"), fetchImpl, headers);
  if (!configText) {
    const sherpa = await resolveSherpaFromBase(baseUrl, quantization, fetchImpl, headers);
    if (!sherpa) {
      throw new Error("Could not detect model type: missing config.json and no sherpa am-onnx/am files.");
    }
    return createAsrModelWithVad({
      modelType: "sherpa-transducer",
      decoderKind: "sherpa-transducer",
      config: { sample_rate: options.sampleRate ?? 16e3, max_tokens_per_step: 10 },
      encoderModel: sherpa.encoder,
      decoderModel: sherpa.decoder,
      decoderJointModel: sherpa.joiner,
      vocabularyText: sherpa.tokensText,
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions
    }, options);
  }
  const config = parseConfigText(configText);
  const { modelType, spec } = detectModelType(config);
  if (spec.decoderKind === "whisper-ort") {
    const modelCandidates = options.whisperModelCandidates ?? [
      "whisper-base_beamsearch.int8.onnx",
      "whisper-base_beamsearch.onnx"
    ];
    const whisperModel = await resolveFirstExistingUrl(baseUrl, modelCandidates, fetchImpl, headers);
    const vocabJson = await fetchText(joinUrl(baseUrl, "vocab.json"), fetchImpl, headers);
    const addedTokensJson = await probeUrl(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers) ? await fetchText(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers) : "{}";
    return createAsrModelWithVad({
      modelType,
      decoderKind: spec.decoderKind,
      config,
      whisperModel,
      vocabJson,
      addedTokensJson,
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions
    }, options);
  }
  if (spec.decoderKind === "whisper-hf") {
    const [encoderModel2, decoderJointModel2, vocabJson, hasAddedTokens] = await Promise.all([
      resolveModelUrl(baseUrl, spec.encoder, { fetchImpl, headers, quantization }),
      resolveModelUrl(baseUrl, spec.decoderJoint, { fetchImpl, headers, quantization }),
      fetchText(joinUrl(baseUrl, "vocab.json"), fetchImpl, headers),
      probeUrl(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers)
    ]);
    return createAsrModelWithVad({
      modelType,
      decoderKind: spec.decoderKind,
      config,
      encoderModel: encoderModel2,
      decoderJointModel: decoderJointModel2,
      vocabJson,
      addedTokensJson: hasAddedTokens ? await fetchText(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers) : "{}",
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions
    }, options);
  }
  if (spec.decoderKind === "gigaam") {
    const artifacts = await resolveGigaamFromBase(baseUrl, config, quantization, fetchImpl, headers);
    if (artifacts.mode === "rnnt") {
      return createAsrModelWithVad({
        modelType,
        decoderKind: "gigaam-rnnt",
        config,
        encoderModel: artifacts.encoder,
        decoderModel: artifacts.decoder,
        decoderJointModel: artifacts.joint,
        vocabularyText: artifacts.vocabularyText,
        sessionOptions: options.sessionOptions,
        decoderOptions: options.decoderOptions
      }, options);
    }
    return createAsrModelWithVad({
      modelType,
      decoderKind: "ctc",
      config,
      encoderModel: artifacts.ctcModel,
      vocabularyText: artifacts.vocabularyText,
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions
    }, options);
  }
  if (spec.decoderKind === "nemo-conformer") {
    const artifacts = await resolveNemoConformerFromBase(baseUrl, quantization, fetchImpl, headers);
    const vocabularyText2 = await fetchFirstExistingText(baseUrl, spec.vocabCandidates, fetchImpl, headers);
    if (artifacts.mode === "rnnt") {
      return createAsrModelWithVad({
        modelType,
        decoderKind: "rnnt",
        config,
        encoderModel: artifacts.encoder,
        decoderJointModel: artifacts.decoderJoint,
        vocabularyText: vocabularyText2,
        sessionOptions: options.sessionOptions,
        decoderOptions: options.decoderOptions
      }, options);
    }
    return createAsrModelWithVad({
      modelType,
      decoderKind: "ctc",
      config,
      encoderModel: artifacts.ctcModel,
      vocabularyText: vocabularyText2,
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions
    }, options);
  }
  const [encoderModel, decoderJointModel, vocabularyText, preprocessorModel] = await Promise.all([
    resolveModelUrl(baseUrl, spec.encoder, { fetchImpl, headers, quantization }),
    spec.decoderJoint ? resolveModelUrl(baseUrl, spec.decoderJoint, { fetchImpl, headers, quantization }) : Promise.resolve(null),
    (async () => {
      const toneText = toneVocabularyTextFromConfig(config);
      if (toneText) {
        return toneText;
      }
      const vocab = await fetchFirstExistingText(baseUrl, spec.vocabCandidates, fetchImpl, headers);
      return spec.vocabCandidates.includes("vocab.json") && vocab.trim().startsWith("{") ? vocabJsonToText(vocab) : vocab;
    })(),
    spec.preprocessor ? resolveModelUrl(baseUrl, spec.preprocessor, { fetchImpl, headers, quantization }) : Promise.resolve(null)
  ]);
  return createAsrModelWithVad({
    modelType,
    decoderKind: spec.decoderKind,
    config,
    preprocessorModel,
    encoderModel,
    decoderJointModel,
    vocabularyText,
    sessionOptions: options.sessionOptions,
    decoderOptions: options.decoderOptions
  }, options);
}
async function loadHuggingfaceModel(repoId, options = {}) {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new Error("No fetch implementation available.");
  }
  const revision = options.revision ?? "main";
  const endpoint = (options.endpoint ?? "https://huggingface.co").replace(/\/$/, "");
  const headers = options.hfToken ? { ...options.headers ?? {}, Authorization: `Bearer ${options.hfToken}` } : options.headers;
  const baseUrl = `${endpoint}/${repoId}/resolve/${encodeURIComponent(revision)}/`;
  if (options.skipRepoListing) {
    return loadLocalModel(baseUrl, { ...options, headers, fetch: fetchImpl });
  }
  const configText = await fetchTextOptional(joinUrl(baseUrl, "config.json"), fetchImpl, headers);
  if (!configText) {
    return loadLocalModel(baseUrl, { ...options, headers, fetch: fetchImpl });
  }
  const config = parseConfigText(configText);
  const { modelType, spec } = detectModelType(config);
  if (spec.decoderKind === "whisper-ort") {
    const repoFiles = await listHuggingFaceRepoFiles(repoId, revision, endpoint, fetchImpl, headers);
    const whisperModelPath = pickWhisperBeamsearchFile(repoFiles, options.quantization ?? "int8");
    if (!whisperModelPath) {
      throw new Error("Could not find whisper-ort beamsearch ONNX in Hugging Face repo.");
    }
    return createAsrModelWithVad({
      modelType,
      decoderKind: spec.decoderKind,
      config,
      whisperModel: joinUrl(baseUrl, whisperModelPath),
      vocabJson: await fetchText(joinUrl(baseUrl, "vocab.json"), fetchImpl, headers),
      addedTokensJson: repoFiles.has("added_tokens.json") ? await fetchText(joinUrl(baseUrl, "added_tokens.json"), fetchImpl, headers) : "{}",
      sessionOptions: options.sessionOptions,
      decoderOptions: options.decoderOptions
    }, options);
  }
  return loadLocalModel(baseUrl, { ...options, headers, fetch: fetchImpl });
}
async function loadHuggingfaceVadModel(repoId, options = {}) {
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (!fetchImpl) {
    throw new Error("No fetch implementation available.");
  }
  const revision = options.revision ?? "main";
  const endpoint = (options.endpoint ?? "https://huggingface.co").replace(/\/$/, "");
  const headers = options.hfToken ? { ...options.headers ?? {}, Authorization: `Bearer ${options.hfToken}` } : options.headers;
  const baseUrl = `${endpoint}/${repoId}/resolve/${encodeURIComponent(revision)}/`;
  return loadLocalVadModel(baseUrl, { ...options, headers, fetch: fetchImpl });
}
export {
  configureOrtWeb,
  loadHuggingfaceModel,
  loadHuggingfaceVadModel,
  loadLocalModel,
  loadLocalVadModel
};
//# sourceMappingURL=browser.js.map

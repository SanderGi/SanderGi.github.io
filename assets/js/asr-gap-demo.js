import {
  configureOrtWeb,
  loadHuggingfaceModel,
} from "/cdn/onnx-asr-web/dist/browser.js";

(function () {
  const root = document.querySelector("[data-asr-demo]");
  if (!root) return;

  configureOrtWeb({
    wasmPaths: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/",
  });

  const canvas = root.querySelector("[data-asr-waveform]");
  const ctx = canvas.getContext("2d");
  const recordButton = root.querySelector("[data-asr-record]");
  const stopButton = root.querySelector("[data-asr-stop]");
  const runButton = root.querySelector("[data-asr-run]");
  const clearButton = root.querySelector("[data-asr-clear]");
  const status = root.querySelector("[data-asr-status]");
  const audio = root.querySelector("[data-asr-audio]");
  const results = root.querySelector("[data-asr-results]");
  const modelInputs = Array.from(root.querySelectorAll("[data-asr-model]"));
  const defaultModelId = modelInputs.find((input) => input.checked)?.value;

  const modelCache = new Map();
  let audioContext = null;
  let analyser = null;
  let mediaStream = null;
  let source = null;
  let processor = null;
  let muteNode = null;
  let chunks = [];
  let recordingSampleRate = 16000;
  let recordedSamples = null;
  let animationId = null;
  let userInteracted = false;

  function setStatus(message) {
    status.textContent = message;
  }

  function setBusy(busy) {
    recordButton.disabled = busy;
    stopButton.disabled = true;
    runButton.disabled = busy || !recordedSamples;
    clearButton.disabled = busy || !recordedSamples;
    for (const input of modelInputs) input.disabled = busy;
  }

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawIdleWave();
  }

  function drawIdleWave() {
    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.strokeStyle = "#12847e";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let x = 0; x <= width; x += 8) {
      const y = height / 2 + Math.sin(x / 28) * 9;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawLiveWave() {
    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteTimeDomainData(data);
    ctx.clearRect(0, 0, width, height);

    ctx.save();
    ctx.fillStyle = "rgba(255, 253, 248, 0.34)";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#ff8a65";
    ctx.lineWidth = 3.5;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let i = 0; i < data.length; i += 1) {
      const x = (i / (data.length - 1)) * width;
      const y = (data[i] / 255) * height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
    animationId = requestAnimationFrame(drawLiveWave);
  }

  function flattenChunks(parts) {
    const length = parts.reduce((total, part) => total + part.length, 0);
    const merged = new Float32Array(length);
    let offset = 0;
    for (const part of parts) {
      merged.set(part, offset);
      offset += part.length;
    }
    return merged;
  }

  function encodeWav(samples, sampleRate) {
    const bytesPerSample = 2;
    const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
    const view = new DataView(buffer);
    const writeString = (offset, value) => {
      for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
    };
    writeString(0, "RIFF");
    view.setUint32(4, 36 + samples.length * bytesPerSample, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * bytesPerSample, true);
    view.setUint16(32, bytesPerSample, true);
    view.setUint16(34, 16, true);
    writeString(36, "data");
    view.setUint32(40, samples.length * bytesPerSample, true);
    let offset = 44;
    for (const sample of samples) {
      const clamped = Math.max(-1, Math.min(1, sample));
      view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
      offset += 2;
    }
    return buffer;
  }

  async function startRecording() {
    try {
      userInteracted = true;
      results.innerHTML = "";
      recordedSamples = null;
      audio.hidden = true;
      audio.removeAttribute("src");
      chunks = [];
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      audioContext = new AudioContext();
      recordingSampleRate = audioContext.sampleRate;
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      source = audioContext.createMediaStreamSource(mediaStream);
      processor = audioContext.createScriptProcessor(4096, 1, 1);
      muteNode = audioContext.createGain();
      muteNode.gain.value = 0;
      processor.onaudioprocess = (event) => {
        chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      source.connect(analyser);
      source.connect(processor);
      processor.connect(muteNode);
      muteNode.connect(audioContext.destination);
      recordButton.disabled = true;
      stopButton.disabled = false;
      runButton.disabled = true;
      clearButton.disabled = true;
      setStatus("Recording... speak naturally, or try noise/accented/atypical speech.");
      drawLiveWave();
    } catch (error) {
      setStatus(error?.message || "Could not access the microphone.");
      cleanupRecording();
      drawIdleWave();
    }
  }

  function cleanupRecording() {
    if (animationId) cancelAnimationFrame(animationId);
    animationId = null;
    if (processor) processor.disconnect();
    if (muteNode) muteNode.disconnect();
    if (source) source.disconnect();
    if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
    if (audioContext) audioContext.close();
    processor = null;
    muteNode = null;
    source = null;
    mediaStream = null;
    audioContext = null;
  }

  function stopRecording() {
    recordedSamples = flattenChunks(chunks);
    const duration = recordedSamples.length / recordingSampleRate;
    cleanupRecording();
    if (!recordedSamples.length) {
      recordedSamples = null;
      recordButton.disabled = false;
      stopButton.disabled = true;
      runButton.disabled = true;
      clearButton.disabled = true;
      setStatus("No audio was captured. Try recording again for a little longer.");
      drawIdleWave();
      return;
    }
    const wav = encodeWav(recordedSamples, recordingSampleRate);
    audio.src = URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
    audio.hidden = false;
    recordButton.disabled = false;
    stopButton.disabled = true;
    runButton.disabled = recordedSamples.length === 0;
    clearButton.disabled = recordedSamples.length === 0;
    setStatus(`Recorded ${duration.toFixed(1)} seconds. Choose models, then transcribe.`);
    drawIdleWave();
  }

  function selectedModels() {
    return modelInputs
      .filter((input) => input.checked)
      .map((input) => ({
        id: input.value,
        label: input.closest("label").textContent.trim(),
      }));
  }

  async function loadModel(modelId) {
    if (!modelCache.has(modelId)) {
      modelCache.set(modelId, loadHuggingfaceModel(modelId, {
        sessionOptions: { executionProviders: ["wasm"] },
        quantization: "int8",
      }));
    }
    return modelCache.get(modelId);
  }

  function renderResult(label, body, error) {
    const card = document.createElement("div");
    card.className = `asr-result${error ? " error" : ""}`;
    const heading = document.createElement("h4");
    heading.textContent = label;
    const text = document.createElement("p");
    text.textContent = body;
    card.append(heading, text);
    results.append(card);
  }

  async function transcribe() {
    userInteracted = true;
    const models = selectedModels();
    if (!recordedSamples) {
      setStatus("Record audio first.");
      return;
    }
    if (!models.length) {
      setStatus("Choose at least one model.");
      return;
    }

    results.innerHTML = "";
    setBusy(true);
    try {
      for (const modelInfo of models) {
        setStatus(`Loading ${modelInfo.label}...`);
        const model = await loadModel(modelInfo.id);
        setStatus(`Running ${modelInfo.label}...`);
        const result = await model.transcribeSamples(recordedSamples, recordingSampleRate);
        renderResult(modelInfo.label, result.text || "(No transcript returned.)");
      }
      setStatus("Finished. Try changing the noise, accent, or speaking style and compare again.");
    } catch (error) {
      renderResult("ASR demo error", error?.message || String(error), true);
      setStatus("Something went wrong while loading or running a model.");
    } finally {
      setBusy(false);
    }
  }

  function clear() {
    userInteracted = true;
    cleanupRecording();
    recordedSamples = null;
    chunks = [];
    results.innerHTML = "";
    audio.hidden = true;
    audio.removeAttribute("src");
    recordButton.disabled = false;
    stopButton.disabled = true;
    runButton.disabled = true;
    clearButton.disabled = true;
    setStatus("Grant microphone access to record a short clip.");
    drawIdleWave();
  }

  recordButton.addEventListener("click", startRecording);
  stopButton.addEventListener("click", stopRecording);
  runButton.addEventListener("click", transcribe);
  clearButton.addEventListener("click", clear);
  modelInputs.forEach((input) => {
    input.addEventListener("change", () => {
      userInteracted = true;
    });
  });
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  const preloadDefaultModel = () => {
    if (!defaultModelId) return;
    loadModel(defaultModelId)
      .then(() => {
        if (!userInteracted && !recordedSamples) {
          setStatus("Default Parakeet model is ready. Grant microphone access to record a short clip.");
        }
      })
      .catch(() => {
        if (!userInteracted && !recordedSamples) {
          setStatus("Grant microphone access to record a short clip.");
        }
      });
  };

  if ("requestIdleCallback" in window) {
    window.requestIdleCallback(preloadDefaultModel, { timeout: 1200 });
  } else {
    window.setTimeout(preloadDefaultModel, 700);
  }
})();

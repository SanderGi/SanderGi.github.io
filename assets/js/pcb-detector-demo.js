(function () {
  const root = document.querySelector("[data-pcb-demo]");
  if (!root || !window.ort) return;

  const modelUrl = "/assets/models/pcb-detector/pcb-seg-yolo11n.onnx";
  const inputSize = 640;
  const scoreThreshold = 0.3;
  const maskThreshold = 0.45;
  const iouThreshold = 0.45;
  const maxDetections = 6;

  const image = root.querySelector("[data-pcb-image]");
  const canvas = root.querySelector("[data-pcb-overlay]");
  const label = root.querySelector("[data-pcb-label]");
  const score = root.querySelector("[data-pcb-score]");
  const note = root.querySelector("[data-pcb-note]");
  const meter = root.querySelector("[data-pcb-meter]");
  const samples = Array.from(root.querySelectorAll("[data-pcb-src]"));
  const upload = root.querySelector("[data-pcb-upload]");

  let sessionPromise = null;
  let objectUrl = null;
  let runId = 0;

  ort.env.wasm.wasmPaths = "/assets/js/vendor/";
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  function setStatus(nextLabel, nextScore, nextNote, confidence) {
    label.textContent = nextLabel;
    score.textContent = nextScore;
    note.textContent = nextNote;
    meter.style.width = `${Math.max(0, Math.min(100, confidence * 100))}%`;
  }

  function getSession() {
    if (!sessionPromise) {
      setStatus("Loading PCB detector", "Preparing the model in your browser.", "The 11 MB ONNX model runs locally with ONNX Runtime Web.", 0.12);
      sessionPromise = ort.InferenceSession.create(modelUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
      });
    }
    return sessionPromise;
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      image.onload = () => {
        requestAnimationFrame(() => resolve());
      };
      image.onerror = reject;
      image.src = src;
    });
  }

  function preprocess() {
    const sourceWidth = image.naturalWidth;
    const sourceHeight = image.naturalHeight;
    const ratio = Math.min(inputSize / sourceWidth, inputSize / sourceHeight);
    const targetWidth = Math.round(sourceWidth * ratio);
    const targetHeight = Math.round(sourceHeight * ratio);
    const padX = Math.floor((inputSize - targetWidth) / 2);
    const padY = Math.floor((inputSize - targetHeight) / 2);

    const scratch = document.createElement("canvas");
    scratch.width = inputSize;
    scratch.height = inputSize;
    const ctx = scratch.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "rgb(114, 114, 114)";
    ctx.fillRect(0, 0, inputSize, inputSize);
    ctx.drawImage(image, padX, padY, targetWidth, targetHeight);

    const rgba = ctx.getImageData(0, 0, inputSize, inputSize).data;
    const data = new Float32Array(3 * inputSize * inputSize);
    const plane = inputSize * inputSize;
    for (let i = 0, p = 0; i < rgba.length; i += 4, p += 1) {
      data[p] = rgba[i] / 255;
      data[plane + p] = rgba[i + 1] / 255;
      data[2 * plane + p] = rgba[i + 2] / 255;
    }

    return {
      tensor: new ort.Tensor("float32", data, [1, 3, inputSize, inputSize]),
      meta: { sourceWidth, sourceHeight, ratio, padX, padY }
    };
  }

  function sigmoid(value) {
    return 1 / (1 + Math.exp(-value));
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
    const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
    return intersection / Math.max(1e-6, areaA + areaB - intersection);
  }

  function nonMaxSuppression(detections) {
    const kept = [];
    const sorted = detections.sort((a, b) => b.confidence - a.confidence);
    for (const detection of sorted) {
      if (kept.every((other) => iou(detection, other) < iouThreshold)) {
        kept.push(detection);
      }
      if (kept.length >= maxDetections) break;
    }
    return kept;
  }

  function parseDetections(results, meta) {
    const tensors = Object.values(results);
    const detectionTensor = tensors.find((tensor) => tensor.dims.length === 3);
    const protoTensor = tensors.find((tensor) => tensor.dims.length === 4);
    if (!detectionTensor || !protoTensor) return { detections: [], proto: null };

    const dims = detectionTensor.dims;
    const channelsFirst = dims[1] < dims[2];
    const channels = channelsFirst ? dims[1] : dims[2];
    const anchors = channelsFirst ? dims[2] : dims[1];
    const protoChannels = protoTensor.dims[1];
    const classCount = Math.max(1, channels - 4 - protoChannels);
    const coeffStart = 4 + classCount;
    const detections = [];

    function value(anchor, channel) {
      return channelsFirst
        ? detectionTensor.data[channel * anchors + anchor]
        : detectionTensor.data[anchor * channels + channel];
    }

    for (let anchor = 0; anchor < anchors; anchor += 1) {
      let confidence = 0;
      for (let cls = 0; cls < classCount; cls += 1) {
        confidence = Math.max(confidence, value(anchor, 4 + cls));
      }
      if (confidence < scoreThreshold) continue;

      const cx = value(anchor, 0);
      const cy = value(anchor, 1);
      const width = value(anchor, 2);
      const height = value(anchor, 3);
      const x1 = clamp((cx - width / 2 - meta.padX) / meta.ratio, 0, meta.sourceWidth);
      const y1 = clamp((cy - height / 2 - meta.padY) / meta.ratio, 0, meta.sourceHeight);
      const x2 = clamp((cx + width / 2 - meta.padX) / meta.ratio, 0, meta.sourceWidth);
      const y2 = clamp((cy + height / 2 - meta.padY) / meta.ratio, 0, meta.sourceHeight);
      if (x2 - x1 < 4 || y2 - y1 < 4) continue;

      const coeffs = new Float32Array(protoChannels);
      for (let i = 0; i < protoChannels; i += 1) {
        coeffs[i] = value(anchor, coeffStart + i);
      }
      detections.push({ x1, y1, x2, y2, confidence, coeffs });
    }

    return {
      detections: nonMaxSuppression(detections),
      proto: protoTensor
    };
  }

  function drawMask(ctx, detection, proto, meta, scaleX, scaleY) {
    const protoChannels = proto.dims[1];
    const protoHeight = proto.dims[2];
    const protoWidth = proto.dims[3];
    const width = canvas.width;
    const height = canvas.height;
    const left = Math.max(0, Math.floor(detection.x1 * scaleX));
    const top = Math.max(0, Math.floor(detection.y1 * scaleY));
    const right = Math.min(width, Math.ceil(detection.x2 * scaleX));
    const bottom = Math.min(height, Math.ceil(detection.y2 * scaleY));
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    for (let y = top; y < bottom; y += 1) {
      const sourceY = y / scaleY;
      const modelY = sourceY * meta.ratio + meta.padY;
      const protoY = clamp(Math.floor((modelY / inputSize) * protoHeight), 0, protoHeight - 1);
      for (let x = left; x < right; x += 1) {
        const sourceX = x / scaleX;
        const modelX = sourceX * meta.ratio + meta.padX;
        const protoX = clamp(Math.floor((modelX / inputSize) * protoWidth), 0, protoWidth - 1);
        const protoOffset = protoY * protoWidth + protoX;
        let maskValue = 0;
        for (let c = 0; c < protoChannels; c += 1) {
          maskValue += detection.coeffs[c] * proto.data[c * protoWidth * protoHeight + protoOffset];
        }
        if (sigmoid(maskValue) > maskThreshold) {
          const index = (y * width + x) * 4;
          data[index] = 12;
          data[index + 1] = 140;
          data[index + 2] = 132;
          data[index + 3] = Math.max(data[index + 3], 112);
        }
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }

  function drawResults(detections, proto, meta) {
    const rect = image.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / meta.sourceWidth;
    const scaleY = canvas.height / meta.sourceHeight;

    for (const detection of detections) {
      drawMask(ctx, detection, proto, meta, scaleX, scaleY);
    }

    ctx.lineWidth = Math.max(3, 3 * dpr);
    ctx.font = `${Math.round(13 * dpr)}px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
    ctx.textBaseline = "top";

    detections.forEach((detection, index) => {
      const x = detection.x1 * scaleX;
      const y = detection.y1 * scaleY;
      const w = (detection.x2 - detection.x1) * scaleX;
      const h = (detection.y2 - detection.y1) * scaleY;
      const text = `PCB ${Math.round(detection.confidence * 100)}%`;
      const textWidth = ctx.measureText(text).width + 14 * dpr;
      const textHeight = 24 * dpr;

      ctx.strokeStyle = index % 2 ? "#ef8f72" : "#0c8c84";
      ctx.fillStyle = "rgba(255, 253, 248, 0.92)";
      ctx.strokeRect(x, y, w, h);
      ctx.fillRect(x, Math.max(0, y - textHeight), textWidth, textHeight);
      ctx.fillStyle = "#23313d";
      ctx.fillText(text, x + 7 * dpr, Math.max(0, y - textHeight) + 5 * dpr);
    });
  }

  async function runDetection(src, sourceButton) {
    const currentRun = ++runId;
    samples.forEach((button) => button.classList.toggle("active", button === sourceButton));
    setStatus("Scanning for PCBs", "Running segmentation locally.", "This is the same kind of specialized detector the agents can call instead of guessing from scratch.", 0.28);

    try {
      await loadImage(src);
      if (currentRun !== runId) return;
      const session = await getSession();
      const { tensor, meta } = preprocess();
      const results = await session.run({ [session.inputNames[0]]: tensor });
      if (currentRun !== runId) return;
      const { detections, proto } = parseDetections(results, meta);
      drawResults(detections, proto, meta);

      if (detections.length) {
        const best = detections[0].confidence;
        setStatus(
          detections.length === 1 ? "1 PCB found" : `${detections.length} PCBs found`,
          `Best match: ${Math.round(best * 100)}% confidence.`,
          "Masks mark the board pixels; boxes show the region passed forward to downstream LCA tools.",
          best
        );
      } else {
        setStatus(
          "No PCB detected",
          "The model did not find a board above the confidence threshold.",
          "Try another preset or upload a clearer electronics image.",
          0
        );
      }
    } catch (error) {
      console.error(error);
      setStatus("Demo could not run", "Something went wrong while loading the model or image.", "Refreshing the page usually clears transient browser runtime issues.", 0);
    }
  }

  samples.forEach((button) => {
    button.addEventListener("click", () => {
      runDetection(button.dataset.pcbSrc, button);
    });
  });

  upload.addEventListener("change", () => {
    const file = upload.files && upload.files[0];
    if (!file) return;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    runDetection(objectUrl, null);
  });

  runDetection(samples[0].dataset.pcbSrc, samples[0]);
})();

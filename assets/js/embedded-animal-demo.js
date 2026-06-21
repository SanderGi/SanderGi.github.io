(() => {
  const root = document.getElementById("embedded-animal-demo");
  if (!root || !window.ort) return;

  const modelUrl = "/assets/models/embedded-arena/animal-detector.onnx";
  const preview = root.querySelector("[data-edge-preview]");
  const label = root.querySelector("[data-edge-label]");
  const score = root.querySelector("[data-edge-score]");
  const note = root.querySelector("[data-edge-note]");
  const meter = root.querySelector("[data-edge-meter]");
  const upload = root.querySelector("[data-edge-upload]");
  const sampleButtons = Array.from(root.querySelectorAll("[data-edge-sample]"));

  let sessionPromise;
  let currentObjectUrl;

  ort.env.wasm.wasmPaths = "/assets/js/vendor/";
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  function setStatus(title, details = "", meterWidth = 0) {
    label.textContent = title;
    score.textContent = details;
    meter.style.width = `${Math.max(0, Math.min(100, meterWidth))}%`;
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("Could not load the selected image."));
      image.src = src;
    });
  }

  function softmax(values) {
    const max = Math.max(...values);
    const exps = values.map((value) => Math.exp(value - max));
    const total = exps.reduce((sum, value) => sum + value, 0);
    return exps.map((value) => value / total);
  }

  function imageToTensor(image) {
    const size = 224;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(image, 0, 0, size, size);
    const { data } = ctx.getImageData(0, 0, size, size);
    const tensor = new Float32Array(3 * size * size);
    const plane = size * size;

    for (let i = 0; i < plane; i += 1) {
      tensor[i] = data[i * 4] / 255;
      tensor[plane + i] = data[i * 4 + 1] / 255;
      tensor[plane * 2 + i] = data[i * 4 + 2] / 255;
    }

    return new ort.Tensor("float32", tensor, [1, 3, size, size]);
  }

  async function getSession() {
    if (!sessionPromise) {
      sessionPromise = ort.InferenceSession.create(modelUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
      });
    }
    return sessionPromise;
  }

  async function classify(src) {
    try {
      setStatus("Running detector...", "Resizing image and running the compressed model.", 12);
      note.textContent = "Inference runs locally in your browser. No image is uploaded.";

      const [session, image] = await Promise.all([getSession(), loadImage(src)]);
      const inputName = session.inputNames[0];
      const outputName = session.outputNames[0];
      const feeds = { [inputName]: imageToTensor(image) };

      const start = performance.now();
      const outputs = await session.run(feeds);
      const elapsed = performance.now() - start;
      const raw = Array.from(outputs[outputName].data).slice(0, 2);
      const probabilities = softmax(raw);
      const animalConfidence = probabilities[0];
      const isAnimal = animalConfidence >= probabilities[1];
      const confidence = isAnimal ? animalConfidence : probabilities[1];

      setStatus(
        isAnimal ? "Animal detected" : "No animal detected",
        `${Math.round(confidence * 100)}% confidence - ${elapsed.toFixed(0)} ms browser inference`,
        animalConfidence * 100
      );
      note.textContent = isAnimal
        ? "A field node would send a LoRa alert instead of waiting for a farmer to inspect every camera."
        : "The node can stay quiet, avoiding alerts for wind, foliage, or empty field motion.";
    } catch (error) {
      setStatus("Detector unavailable", "The model could not run in this browser.", 0);
      note.textContent = error.message || "Try a different browser or refresh the page.";
    }
  }

  function setImage(src, activeButton = null) {
    if (currentObjectUrl && currentObjectUrl !== src) {
      URL.revokeObjectURL(currentObjectUrl);
      currentObjectUrl = null;
    }
    sampleButtons.forEach((button) => button.classList.toggle("active", button === activeButton));
    preview.src = src;
    classify(src);
  }

  sampleButtons.forEach((button) => {
    button.addEventListener("click", () => {
      setImage(button.dataset.edgeSample, button);
    });
  });

  upload.addEventListener("change", () => {
    const file = upload.files && upload.files[0];
    if (!file) return;
    if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = URL.createObjectURL(file);
    setImage(currentObjectUrl);
  });

  setStatus("Model loading...", "Preparing the browser runtime.", 6);
  classify(preview.src);
})();

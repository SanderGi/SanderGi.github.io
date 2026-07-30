(function () {
  "use strict";

  const root = document.querySelector("[data-spectrogram-tutorial]");
  if (!root) return;

  const stage = root.querySelector("[data-spectrogram-stage]");
  const nav = root.querySelector("[data-spectrogram-nav]");
  const previousButton = root.querySelector("[data-spectrogram-previous]");
  const nextButton = root.querySelector("[data-spectrogram-next]");
  const audio = root.querySelector("[data-spectrogram-audio]");
  const exercises = window.SPECTROGRAM_EXERCISES || [];
  const reduceMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;

  const phonemeBase = "/images/demo/spectrogram-tutorial/phonemes/";
  const soundImageBase =
    "/images/demo/spectrogram-tutorial/sound-spectrograms/";
  const phonemeAudio = {
    p: "p.m4a",
    b: "b.m4a",
    t: "t.m4a",
    d: "d.m4a",
    k: "k.m4a",
    ɡ: "g.m4a",
    m: "m.m4a",
    n: "n.m4a",
    ŋ: "ng.m4a",
    f: "f.m4a",
    v: "v.m4a",
    θ: "th-voiceless.m4a",
    ð: "th-buzzed.m4a",
    s: "s.m4a",
    z: "z.m4a",
    ʃ: "sh.m4a",
    ʒ: "zsh.m4a",
    h: "h.m4a",
    tʃ: "ch.m4a",
    dʒ: "j.m4a",
    ɹ: "r.m4a",
    j: "y.m4a",
    w: "w.m4a",
    l: "l.m4a",
    ɾ: "tap.m4a",
    i: "ee.m4a",
    ɪ: "ih.m4a",
    eɪ: "ei.m4a",
    ɛ: "eh.m4a",
    æ: "ah.m4a",
    ɑ: "aa.m4a",
    ɔ: "aw.m4a",
    ʌ: "uh.m4a",
    u: "oo.m4a",
    ʊ: "foot.mp3",
    ə: "about.mp3",
    oʊ: "oh.m4a",
    aɪ: "ay.m4a",
    aʊ: "ow.m4a",
    ɔɪ: "oy.m4a",
    "ə˞": "ur.m4a",
    "ɜ˞": "er.m4a",
  };

  const consonantRows = [
    {
      label: "Plosive",
      note: "Stop the air, then release it.",
      cells: [["p", "b"], [], [], ["t", "d"], [], [], ["k", "ɡ"], []],
    },
    {
      label: "Nasal",
      note: "Send the air through the nose.",
      cells: [["m"], [], [], ["n"], [], [], ["ŋ"], []],
    },
    {
      label: "Fricative",
      note: "Squeeze air through a narrow gap.",
      cells: [
        [],
        ["f", "v"],
        ["θ", "ð"],
        ["s", "z"],
        ["ʃ", "ʒ"],
        [],
        [],
        ["h"],
      ],
    },
    {
      label: "Affricate",
      note: "A stop released into friction.",
      cells: [[], [], [], [], ["tʃ", "dʒ"], [], [], []],
    },
    {
      label: "Approximant",
      note: "Bring articulators close, without friction.",
      cells: [["w"], [], [], ["ɹ"], [], ["j"], [], []],
    },
    {
      label: "Lateral",
      note: "Let air pass around the tongue.",
      cells: [[], [], [], ["l"], [], [], [], []],
    },
    {
      label: "Tap / flap",
      note: "A very quick tongue touch, as in US water.",
      cells: [[], [], [], ["ɾ"], [], [], [], []],
    },
  ];
  const consonantPlaces = [
    "Bilabial",
    "Labiodental",
    "Dental",
    "Alveolar",
    "Postalveolar",
    "Palatal",
    "Velar",
    "Glottal",
  ];

  const vowels = [
    { symbol: "i", example: "fleece", left: 9, top: 12, f1: 280, f2: 2230 },
    { symbol: "ɪ", example: "kit", left: 28, top: 26, f1: 370, f2: 2090 },
    { symbol: "eɪ", example: "face", left: 20, top: 42, f1: 405, f2: 2080 },
    { symbol: "ɛ", example: "dress", left: 31, top: 56, f1: 600, f2: 1930 },
    { symbol: "æ", example: "trap", left: 39, top: 80, f1: 860, f2: 1550 },
    { symbol: "ʌ", example: "strut", left: 56, top: 62, f1: 680, f2: 1310 },
    { symbol: "ɑ", example: "palm", left: 82, top: 80, f1: 830, f2: 1170 },
    { symbol: "ɔ", example: "thought", left: 84, top: 57, f1: 560, f2: 820 },
    { symbol: "oʊ", example: "goat", left: 79, top: 40, f1: 430, f2: 980 },
    { symbol: "ʊ", example: "foot", left: 69, top: 29, f1: 400, f2: 1100 },
    { symbol: "ə", example: "about", left: 55, top: 47, f1: 500, f2: 1500 },
    { symbol: "u", example: "goose", left: 77, top: 12, f1: 330, f2: 1260 },
  ];

  const lessonMeta = [
    ["The IPA map", "IPA"],
    ["Spelling and dialect", "Dialect"],
    ["Sound is pressure", "Pressure"],
    ["From air to waveform", "Wave"],
    ["Inside a complex sound", "Spectrum"],
    ["Stacking time", "Spectrogram"],
    ["Wideband and log-mel", "Mel"],
    ["The formant trick", "Formants"],
    ["Find the formants", "F1 / F2"],
    ["Sound signatures", "Patterns"],
    ["Find the word", "Practice"],
  ];

  let currentLesson = 0;
  let cleanup = [];
  let endlessIndex = 0;
  let endlessCorrect = 0;
  let endlessAttempts = 0;

  function setStatus() {}

  function soundImage(symbol) {
    const filename = phonemeAudio[symbol];
    if (!filename) return "";
    return soundImageBase + filename.replace(/\.(m4a|mp3)$/i, ".webp");
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function listen(target, type, handler, options) {
    target.addEventListener(type, handler, options);
    cleanup.push(function () {
      target.removeEventListener(type, handler, options);
    });
  }

  function stopLesson() {
    cleanup.forEach(function (dispose) {
      try {
        dispose();
      } catch (error) {
        console.warn(error);
      }
    });
    cleanup = [];
    audio.pause();
    audio.removeAttribute("src");
    stage.classList.remove("is-practice");
    stage.classList.remove("is-formant-quiz");
  }

  function playPhoneme(symbol, button) {
    const filename = phonemeAudio[symbol];
    if (!filename) {
      setStatus("/" + symbol + "/ is demonstrated by its word example.");
      return;
    }
    root.querySelectorAll(".is-sounding").forEach(function (item) {
      item.classList.remove("is-sounding");
    });
    if (button) button.classList.add("is-sounding");
    audio.src = phonemeBase + filename;
    audio
      .play()
      .then(function () {
        setStatus("Playing /" + symbol + "/.");
      })
      .catch(function () {
        if (button) button.classList.remove("is-sounding");
        setStatus("Select the symbol again to play /" + symbol + "/.");
      });
    audio.onended = function () {
      if (button) button.classList.remove("is-sounding");
    };
  }

  function bindPhonemeButtons(container) {
    container.querySelectorAll("[data-ipa]").forEach(function (button) {
      listen(button, "click", function () {
        playPhoneme(button.dataset.ipa, button);
      });
    });
  }

  function canvasRenderer(canvas, draw) {
    function render() {
      const bounds = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.round(bounds.width * ratio));
      canvas.height = Math.max(1, Math.round(bounds.height * ratio));
      const context = canvas.getContext("2d");
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw(context, bounds.width, bounds.height);
    }
    render();
    listen(window, "resize", render);
    return render;
  }

  function roundedRect(context, x, y, width, height, radius) {
    context.beginPath();
    if (context.roundRect) {
      context.roundRect(x, y, width, height, radius);
      return;
    }
    const corner = Math.min(radius, width / 2, height / 2);
    context.moveTo(x + corner, y);
    context.lineTo(x + width - corner, y);
    context.quadraticCurveTo(x + width, y, x + width, y + corner);
    context.lineTo(x + width, y + height - corner);
    context.quadraticCurveTo(
      x + width,
      y + height,
      x + width - corner,
      y + height,
    );
    context.lineTo(x + corner, y + height);
    context.quadraticCurveTo(x, y + height, x, y + height - corner);
    context.lineTo(x, y + corner);
    context.quadraticCurveTo(x, y, x + corner, y);
    context.closePath();
  }

  function axisLabel(context, text, x, y, align) {
    context.save();
    context.fillStyle = "#53636d";
    context.font = "600 12px system-ui, sans-serif";
    context.textAlign = align || "center";
    context.fillText(text, x, y);
    context.restore();
  }

  function renderIpaLesson() {
    const headers = consonantPlaces
      .map(function (place) {
        return '<span class="ipa-place">' + place + "</span>";
      })
      .join("");
    const rows = consonantRows
      .map(function (row) {
        const cells = row.cells
          .map(function (symbols) {
            return (
              '<span class="ipa-cell">' +
              symbols
                .map(function (symbol) {
                  const playable = Boolean(phonemeAudio[symbol]);
                  return (
                    '<button type="button" class="ipa-symbol" data-ipa="' +
                    symbol +
                    '" aria-label="Play /' +
                    symbol +
                    '/ sound"' +
                    (playable ? "" : " disabled") +
                    ">" +
                    symbol +
                    "</button>"
                  );
                })
                .join("") +
              "</span>"
            );
          })
          .join("");
        return (
          '<span class="ipa-manner" title="' +
          escapeHtml(row.note) +
          '">' +
          row.label +
          "</span>" +
          cells
        );
      })
      .join("");
    const vowelButtons = vowels
      .map(function (vowel) {
        return (
          '<button type="button" class="ipa-vowel" data-ipa="' +
          vowel.symbol +
          '" style="--left:' +
          vowel.left +
          "%;--top:" +
          vowel.top +
          '%" aria-label="Play /' +
          vowel.symbol +
          "/ as in " +
          vowel.example +
          '"><b>' +
          vowel.symbol +
          "</b><small>" +
          vowel.example +
          "</small></button>"
        );
      })
      .join("");

    stage.innerHTML =
      '<div class="spectrogram-lesson-heading">' +
      '<span class="spectrogram-lesson-number">01</span>' +
      "<div><h3>How to describe sounds</h3>" +
      "<p>The International Phonetic Alphabet (IPA) gives each speech sound its own symbol based on how it is produced. " +
      "That lets us describe pronunciation one-to-one where English spelling is more arbitrary.</p></div>" +
      "</div>" +
      '<div class="ipa-chart-block">' +
      '<div class="ipa-chart-intro"><h4>Consonants</h4><p>Sounds where air is partially or completely blocked by the lips, teeth, or tongue. <b>Rows</b> describe how air is shaped. ' +
      "<b>Columns</b> describe where the mouth makes the constriction. For symbols in pairs, the right is voiced meaning the vocal cord vibrates. Select a symbol to hear it.</p></div>" +
      '<div class="ipa-table-scroll"><div class="ipa-consonant-grid" role="group" aria-label="English consonant chart">' +
      '<span class="ipa-corner">How / where</span>' +
      headers +
      rows +
      "</div></div>" +
      '<div class="ipa-consonant-glossary">' +
      '<div><b>How air moves</b><div><span><strong style="font-weight:700">Plosive</strong> Seal, then pop.</span><span><strong style=\"font-weight:700\">Fricative</strong> Noisy airflow.</span><span><strong style=\"font-weight:700\">Nasal</strong> Air through the nose.</span><span><strong style="font-weight:700">Affricate</strong> Stop + fricative.</span><span><strong style="font-weight:700">Approximant</strong> Almost vowel.</span><span><strong style="font-weight:700">Lateral</strong> Tongue block.</span><span><strong style=\"font-weight:700\">Tap</strong> Quick tongue touch.</span></div></div>' +
      '<div><b>Where it happens</b><div><span><strong style="font-weight:700">Bilabial</strong> Both lips.</span><span><strong style="font-weight:700">Labiodental</strong> Bottom lip to top teeth.</span><span><strong style=\"font-weight:700\">Dental</strong> Tongue at teeth.</span><span><strong style=\"font-weight:700\">Alveolar</strong> Tongue near alveolar ridge.</span><span><strong style="font-weight:700">Postalveolar</strong> Tongue behind ridge.</span><span><strong style="font-weight:700">Palatal</strong> Tongue-middle raised to roof.</span><span><strong style=\"font-weight:700\">Velar</strong> Tongue at soft palate.</span><span><strong style=\"font-weight:700\">Glottal</strong> In the throat.</span></div></div>' +
      "</div>" +
      "</div>" +
      '<div style="height:0.3em"></div><div class="ipa-vowel-layout">' +
      "<div><h4>Vowels</h4><p>Vowels have open airflow. <b>Front to back</b> tracks where your tongue bunches; " +
      "<b>close to open</b> tracks tongue height and jaw opening.</p>" +
      '<div class="ipa-plain-definitions">' +
      '<span><b style="font-weight:700">Close</b> Tongue near the roof, as in <i>see</i>.</span>' +
      '<span><b style="font-weight:700">Open</b> Jaw and tongue lower, as in <i>father</i>.</span>' +
      '<span><b style="font-weight:700">Front</b> Tongue forward, as in <i>bed</i>.</span>' +
      '<span><b style="font-weight:700">Back</b> Tongue pulled back, as in <i>goose</i>.</span>' +
      "</div></div>" +
      '<div class="ipa-vowel-chart" role="group" aria-label="English vowel chart">' +
      '<span class="ipa-axis ipa-axis-front">Front</span><span class="ipa-axis ipa-axis-back">Back</span>' +
      '<span class="ipa-axis ipa-axis-close">Close</span><span class="ipa-axis ipa-axis-open">Open</span>' +
      vowelButtons +
      "</div>" +
      "</div>";
    bindPhonemeButtons(stage);
    setStatus("Choose an IPA symbol to hear it.");
  }

  function renderDialectLesson() {
    stage.innerHTML =
      '<div class="spectrogram-lesson-heading">' +
      '<span class="spectrogram-lesson-number">02</span>' +
      "<div><h3>IPA vs English</h3>" +
      "<p>IPA records what a speaker actually says. This means it can communicate phenomena like dialects. Orthographic spellings like written English don't capture these nuances.</p></div>" +
      "</div>" +
      '<div class="ipa-word-examples">' +
      "<article><span>rough</span><strong>/ɹʌf/</strong><p>Five letters, three sounds.</p></article>" +
      "<article><span>through</span><strong>/θɹu/</strong><p>Same <i>ough</i>, different sound.</p></article>" +
      "<article><span>colonel</span><strong>/ˈkɜ˞nəl/</strong><p>Silent letters disappear.</p></article>" +
      "<article><span>photo</span><strong>/ˈfoʊ̯.toʊ̯/</strong><p>'Ph' is just an /f/.</p></article>" +
      "</div>" +
      '<div class="dialect-table" role="table" aria-label="Pronunciation differences across English dialects">' +
      '<div class="dialect-row dialect-head" role="row"><span>Word</span><span>General American</span><span>Southern British</span><span>What moved?</span></div>' +
      '<div class="dialect-row" role="row"><b>tomato</b><span>/təˈmeɪtoʊ/</span><span>/təˈmɑːtəʊ/</span><small>The stressed vowel.</small></div>' +
      '<div class="dialect-row" role="row"><b>schedule</b><span>/ˈskedʒuːl/</span><span>/ˈʃedjuːl/</span><small>The opening consonants.</small></div>' +
      '<div class="dialect-row" role="row"><b>water</b><span>/ˈwɔɾɚ/</span><span>/ˈwɔːtə/</span><small>A tap and an R-colored ending.</small></div>' +
      '<div class="dialect-row" role="row"><b>cot / caught</b><span>Often both /kɑt/</span><span>/kɒt/ vs. /kɔːt/</span><small>Some dialects merge the vowels.</small></div>' +
      "</div>" +
      '<div class="spectrogram-callout"><b>Reading rule:</b> slashes mark a broad phoneme sequence, while square brackets can record finer details such as the American tap [ɾ].</div>';
    setStatus("IPA describes pronunciation without ranking dialects.");
  }

  function renderPressureLesson() {
    stage.innerHTML =
      '<div class="spectrogram-lesson-heading">' +
      '<span class="spectrogram-lesson-number">03</span>' +
      "<div><h3>Sound begins as crowded and uncrowded air</h3>" +
      "<p>A vibrating source alternately compresses and rarefies nearby air. The pattern travels; the individual molecules mostly wobble in place.</p></div>" +
      "</div>" +
      '<div class="pressure-lab">' +
      '<canvas data-pressure-canvas aria-label="Animated air-pressure wave traveling from a speaker to a microphone"></canvas>' +
      '<div class="pressure-controls">' +
      '<button type="button" data-pressure-toggle>Pause animation</button>' +
      '<span><i class="pressure-key is-compression"></i> Compression: above-average pressure</span>' +
      '<span><i class="pressure-key is-rarefaction"></i> Rarefaction: below-average pressure</span>' +
      "</div>" +
      "</div>" +
      '<div class="pressure-takeaway-grid">' +
      "<article><b>The air does not fly across the room.</b><p>Each molecule nudges its neighbor, like a crowd passing a pulse.</p></article>" +
      "<article><b>A microphone samples pressure.</b><p>It records tiny pressure deviations as numbers over time.</p></article>" +
      "<article><b>Your ear reverses the process.</b><p>The eardrum moves with the pressure and the brain interprets the pattern.</p></article>" +
      "</div>";

    const canvas = stage.querySelector("[data-pressure-canvas]");
    const toggle = stage.querySelector("[data-pressure-toggle]");
    let running = !reduceMotion;
    let startTime = performance.now();
    let frame = 0;
    let width = 0;
    let height = 0;
    let context = null;

    function resize() {
      const bounds = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(bounds.width * ratio);
      canvas.height = Math.round(bounds.height * ratio);
      canvas.getContext("2d").setTransform(ratio, 0, 0, ratio, 0, 0);
      width = bounds.width;
      height = bounds.height;
      context = canvas.getContext("2d");
      draw(performance.now());
    }

    function draw(now) {
      if (!context) return;
      const time = running ? (now - startTime) / 1000 : 0;
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#fffaf4";
      context.fillRect(0, 0, width, height);

      context.fillStyle = "#23313b";
      roundedRect(context, 20, height / 2 - 38, 34, 76, 6);
      context.fill();
      context.beginPath();
      context.moveTo(54, height / 2 - 27);
      context.lineTo(87, height / 2 - 53);
      context.lineTo(87, height / 2 + 53);
      context.lineTo(54, height / 2 + 27);
      context.closePath();
      context.fill();

      const columns = Math.max(20, Math.floor((width - 155) / 23));
      const rows = 6;
      for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < columns; column += 1) {
          const baseX =
            110 + column * ((width - 220) / Math.max(columns - 1, 1));
          const phase = column * 0.72 - time * 5.2;
          const displacement = Math.sin(phase) * 7;
          const compression = Math.cos(phase);
          context.beginPath();
          context.arc(
            baseX + displacement,
            38 + row * ((height - 76) / (rows - 1)),
            3.4,
            0,
            Math.PI * 2,
          );
          context.fillStyle =
            compression > 0.35
              ? "#ef7b61"
              : compression < -0.35
                ? "#69b9df"
                : "#6b7c84";
          context.fill();
        }
      }

      context.strokeStyle = "#147c75";
      context.lineWidth = 5;
      roundedRect(context, width - 87, height / 2 - 52, 48, 104, 22);
      context.stroke();
      context.beginPath();
      context.moveTo(width - 63, height / 2 + 52);
      context.lineTo(width - 63, height / 2 + 72);
      context.moveTo(width - 80, height / 2 + 72);
      context.lineTo(width - 46, height / 2 + 72);
      context.stroke();
      axisLabel(context, "source", 53, height - 10);
      axisLabel(context, "microphone", width - 63, height - 10);

      if (running) frame = requestAnimationFrame(draw);
    }

    listen(toggle, "click", function () {
      running = !running;
      toggle.textContent = running ? "Pause animation" : "Resume animation";
      if (running) {
        startTime = performance.now();
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(draw);
      } else {
        cancelAnimationFrame(frame);
      }
    });
    resize();
    listen(window, "resize", resize);
    if (running) frame = requestAnimationFrame(draw);
    cleanup.push(function () {
      cancelAnimationFrame(frame);
    });
    setStatus("The colored bands move; each air molecule only wobbles.");
  }

  function renderWaveLesson() {
    stage.innerHTML =
      '<div class="spectrogram-lesson-heading">' +
      '<span class="spectrogram-lesson-number">04</span>' +
      "<div><h3>A waveform graphs the pressure at a point over time</h3>" +
      "<p>Consider the pressure at the receiving microphone. Time runs left to right. Height is instantaneous pressure relative to the quiet background. Bigger swings are louder; faster swings have higher pitch.</p></div>" +
      "</div>" +
      '<div class="wave-lab">' +
      '<div class="wave-controls">' +
      '<label>Amplitude <input type="range" min="0.15" max="1" step="0.05" value="0.65" data-wave-amplitude><output data-wave-amplitude-output>65%</output></label>' +
      '<label>Frequency <input type="range" min="1" max="8" step="0.25" value="3" data-wave-frequency><output data-wave-frequency-output>300 Hz</output></label>' +
      '<button type="button" data-wave-play>Hear it</button>' +
      "</div>" +
      '<div class="wave-canvas-wrap"><span class="wave-y-label">pressure / amplitude</span><canvas data-wave-canvas></canvas><span class="wave-x-label">time</span></div>' +
      "</div>" +
      '<div class="spectrogram-equation"><span>slow repetition</span><b>→ lower frequency → lower pitch</b><span>fast repetition</span><b>→ higher frequency → higher pitch</b></div>';

    const canvas = stage.querySelector("[data-wave-canvas]");
    const amplitude = stage.querySelector("[data-wave-amplitude]");
    const frequency = stage.querySelector("[data-wave-frequency]");
    const amplitudeOutput = stage.querySelector("[data-wave-amplitude-output]");
    const frequencyOutput = stage.querySelector("[data-wave-frequency-output]");
    const play = stage.querySelector("[data-wave-play]");

    const redraw = canvasRenderer(canvas, function (context, width, height) {
      const center = height / 2;
      const amp = Number(amplitude.value) * (height * 0.38);
      const cycles = Number(frequency.value);
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#fffaf4";
      context.fillRect(0, 0, width, height);
      context.strokeStyle = "#ccd5d7";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(0, center);
      context.lineTo(width, center);
      context.stroke();
      context.strokeStyle = "#14877f";
      context.lineWidth = 4;
      context.beginPath();
      for (let x = 0; x <= width; x += 2) {
        const y = center - Math.sin((x / width) * Math.PI * 2 * cycles) * amp;
        if (x === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
      context.fillStyle = "#ef7b61";
      context.fillRect(0, center - amp, 8, amp);
      context.fillRect(0, center, 8, amp);
    });

    function update() {
      amplitudeOutput.textContent =
        Math.round(Number(amplitude.value) * 100) + "%";
      frequencyOutput.textContent =
        Math.round(Number(frequency.value) * 100) + " Hz";
      redraw();
    }
    listen(amplitude, "input", update);
    listen(frequency, "input", update);
    listen(play, "click", function () {
      const AudioContextClass =
        window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) return;
      const context = new AudioContextClass();
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.frequency.value = Number(frequency.value) * 100;
      gain.gain.setValueAtTime(0.0001, context.currentTime);
      gain.gain.exponentialRampToValueAtTime(
        Number(amplitude.value) * 0.12,
        context.currentTime + 0.03,
      );
      gain.gain.exponentialRampToValueAtTime(
        0.0001,
        context.currentTime + 0.75,
      );
      oscillator.connect(gain).connect(context.destination);
      oscillator.start();
      oscillator.stop(context.currentTime + 0.78);
      play.disabled = true;
      setStatus("Playing a " + frequencyOutput.textContent + " sine wave.");
      oscillator.onended = function () {
        play.disabled = false;
        context.close();
      };
    });
    setStatus("Move either slider and watch the waveform change.");
  }

  function renderSpectrumLesson() {
    stage.innerHTML =
      '<div class="spectrogram-lesson-heading">' +
      '<span class="spectrogram-lesson-number">05</span>' +
      "<div><h3>A complex sound is many sine waves added together</h3>" +
      "<p>Fourier analysis reverses that addition. It asks: which frequencies are present right now, and how strong is each one? When we plot the frequency distribution at a specific point in time, we get the spectrum.</p></div>" +
      "</div>" +
      '<div class="spectrum-lab">' +
      '<div class="spectrum-picker" role="group" aria-label="Choose component frequencies">' +
      '<label><input type="checkbox" value="2" data-component checked><span style="--tone:#17877f">220 Hz</span></label>' +
      '<label><input type="checkbox" value="4" data-component checked><span style="--tone:#ef7b61">440 Hz</span></label>' +
      '<label><input type="checkbox" value="7" data-component checked><span style="--tone:#7659c7">770 Hz</span></label>' +
      "</div>" +
      '<div class="spectrum-grid">' +
      '<figure><figcaption>Waveform</figcaption><canvas data-combined-wave></canvas><span class="wave-x-label">time</span></figure>' +
      '<figure><figcaption>Spectrum</figcaption><canvas data-spectrum-bars></canvas><span class="wave-x-label">frequency</span></figure>' +
      "</div>" +
      "</div>" +
      '<div class="spectrogram-callout"><b>The tradeoff:</b> the waveform preserves exact pressure at every instant, while a spectrum makes the ingredients legible but describes only a short slice of time. Luckily as we see next, we can plot the spectra over time into a spectrogram.</div>';

    const inputs = Array.from(stage.querySelectorAll("[data-component]"));
    const waveCanvas = stage.querySelector("[data-combined-wave]");
    const spectrumCanvas = stage.querySelector("[data-spectrum-bars]");
    const colors = ["#17877f", "#ef7b61", "#7659c7"];
    let drawWave;
    let drawSpectrum;

    drawWave = canvasRenderer(waveCanvas, function (context, width, height) {
      const active = inputs
        .map(function (input, index) {
          return input.checked
            ? { frequency: Number(input.value), color: colors[index] }
            : null;
        })
        .filter(Boolean);
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#fffaf4";
      context.fillRect(0, 0, width, height);
      context.strokeStyle = "#d5ddde";
      context.beginPath();
      context.moveTo(0, height / 2);
      context.lineTo(width, height / 2);
      context.stroke();
      context.strokeStyle = "#23313b";
      context.lineWidth = 3;
      context.beginPath();
      for (let x = 0; x <= width; x += 2) {
        const sum = active.reduce(function (value, tone, index) {
          return (
            value +
            Math.sin((x / width) * Math.PI * 2 * tone.frequency) *
              (1 / (index + 1))
          );
        }, 0);
        const divisor = Math.max(active.length * 0.65, 1);
        const y = height / 2 - (sum / divisor) * height * 0.35;
        if (x === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
    });
    drawSpectrum = canvasRenderer(
      spectrumCanvas,
      function (context, width, height) {
        context.clearRect(0, 0, width, height);
        context.fillStyle = "#fffaf4";
        context.fillRect(0, 0, width, height);
        context.strokeStyle = "#d5ddde";
        context.beginPath();
        context.moveTo(28, height - 24);
        context.lineTo(width - 12, height - 24);
        context.stroke();
        inputs.forEach(function (input, index) {
          if (!input.checked) return;
          const x = 45 + (Number(input.value) / 8) * (width - 80);
          const barHeight = (height - 58) * (1 - index * 0.18);
          context.fillStyle = colors[index];
          roundedRect(
            context,
            x - 8,
            height - 24 - barHeight,
            16,
            barHeight,
            5,
          );
          context.fill();
          axisLabel(
            context,
            input.parentElement.textContent.trim(),
            x,
            height - 7,
          );
        });
      },
    );
    inputs.forEach(function (input) {
      listen(input, "change", function () {
        drawWave();
        drawSpectrum();
      });
    });
    setStatus(
      "Toggle frequencies to see how the recipe and final wave stay linked.",
    );
  }

  function renderStackLesson() {
    stage.innerHTML =
      '<div class="spectrogram-lesson-heading">' +
      '<span class="spectrogram-lesson-number">06</span>' +
      "<div><h3>A spectrogram stacks spectra through time</h3>" +
      "<p>Each curve is a spectrum from one tiny moment. Stack the curves along a new time dimension and then encode the amplitude as darkness.</p></div>" +
      "</div>" +
      '<div class="stack-lab">' +
      '<canvas data-stack-canvas aria-label="Three-dimensional spectra rotated into a spectrogram"></canvas>' +
      '<label class="stack-rotation-control"><span>See amplitude</span><input type="range" min="0" max="1" step="0.01" value="0" data-stack-rotation><span>Amplitude faces you</span></label>' +
      "</div>" +
      '<div class="spectrogram-axis-summary">' +
      "<span><b>x</b> time</span><span><b>y</b> frequency</span><span><b>darkness</b> amplitude</span>" +
      "</div>";

    const canvas = stage.querySelector("[data-stack-canvas]");
    const rotation = stage.querySelector("[data-stack-rotation]");

    function energyAt(time, frequency) {
      const formantOne = Math.exp(
        -Math.pow((frequency - (0.2 + time * 0.05)) / 0.055, 2),
      );
      const formantTwo = Math.exp(
        -Math.pow((frequency - (0.62 - time * 0.12)) / 0.07, 2),
      );
      const harmonic = 0.2 * (Math.sin(frequency * 118 + time * 8) + 1);
      return Math.min(
        1,
        0.08 + formantOne * 0.72 + formantTwo * 0.62 + harmonic,
      );
    }

    const redraw = canvasRenderer(canvas, function (context, width, height) {
      const progress = Number(rotation.value);
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#fffaf4";
      context.fillRect(0, 0, width, height);
      const slices = 18;
      const bins = 38;
      const centerX = width * 0.5;
      const bottom = height - 42;
      const scaleX = Math.min(width * 0.34, 310);
      const scaleY = height * 0.72;
      const amplitudeScale = Math.min(width * 0.17, 120);
      const angle = ((1 - progress) * Math.PI) / 2;
      const meshOpacity = 1 - Math.max(0, (progress - 0.72) / 0.28);
      const imageOpacity = Math.max(0, (progress - 0.66) / 0.34);

      function project(time, frequency, amplitude) {
        const worldX = (time - 0.5) * 2;
        const worldZ = amplitude;
        const screenPlane =
          worldX * Math.cos(angle) -
          worldZ * Math.sin(angle) * (amplitudeScale / scaleX);
        const depth = worldX * Math.sin(angle) + worldZ * Math.cos(angle);
        return {
          x: centerX + screenPlane * scaleX,
          y: bottom - frequency * scaleY - depth * 20,
        };
      }

      function drawForeshortenedAxisLabel(text, x, y, projection) {
        context.save();
        context.translate(x, y);
        context.scale(Math.max(0.001, projection), 1);
        axisLabel(context, text, 0, 0);
        context.restore();
      }

      if (imageOpacity > 0) {
        const imageLeft = centerX - scaleX;
        const imageTop = bottom - scaleY;
        const cellWidth = (scaleX * 2) / slices;
        const cellHeight = scaleY / bins;
        for (let slice = 0; slice < slices; slice += 1) {
          for (let bin = 0; bin < bins; bin += 1) {
            const energy = energyAt(slice / (slices - 1), bin / (bins - 1));
            context.fillStyle =
              "rgba(0,0,0," + (energy * imageOpacity * 0.92).toFixed(3) + ")";
            context.fillRect(
              imageLeft + slice * cellWidth,
              imageTop + (bins - 1 - bin) * cellHeight,
              cellWidth + 1,
              cellHeight + 1,
            );
          }
        }
      }

      if (meshOpacity > 0) {
        context.strokeStyle =
          "rgba(19,127,122," + (0.76 * meshOpacity).toFixed(3) + ")";
        context.lineWidth = 1.7;
        for (let slice = 0; slice < slices; slice += 1) {
          const time = slice / (slices - 1);
          context.beginPath();
          for (let bin = 0; bin < bins; bin += 1) {
            const frequency = bin / (bins - 1);
            const point = project(time, frequency, energyAt(time, frequency));
            if (bin === 0) context.moveTo(point.x, point.y);
            else context.lineTo(point.x, point.y);
          }
          context.stroke();
        }
        context.strokeStyle =
          "rgba(35,49,61," + (0.22 * meshOpacity).toFixed(3) + ")";
        for (let bin = 0; bin < bins; bin += 5) {
          const frequency = bin / (bins - 1);
          context.beginPath();
          for (let slice = 0; slice < slices; slice += 1) {
            const time = slice / (slices - 1);
            const point = project(time, frequency, energyAt(time, frequency));
            if (slice === 0) context.moveTo(point.x, point.y);
            else context.lineTo(point.x, point.y);
          }
          context.stroke();
        }
      }

      const timeProjection = Math.cos(angle);
      const amplitudeProjection = Math.sin(angle);
      const axisY = bottom + 29;
      drawForeshortenedAxisLabel("time →", centerX, axisY, timeProjection);
      drawForeshortenedAxisLabel(
        "amplitude →",
        centerX + scaleX * timeProjection,
        axisY,
        amplitudeProjection,
      );
      context.save();
      context.translate(18, height / 2);
      context.rotate(-Math.PI / 2);
      axisLabel(context, "frequency →", 0, 0);
      context.restore();
      axisLabel(
        context,
        progress < 0.98 ? "" : "Amplitude is now encoded as blackness",
        width - 12,
        18,
        "right",
      );
    });

    listen(rotation, "input", redraw);
  }

  function renderMelLesson() {
    stage.innerHTML =
      '<div class="spectrogram-lesson-heading">' +
      '<span class="spectrogram-lesson-number">07</span>' +
      "<div><h3>Adjusting the spectrogram for reading speech</h3>" +
      "<p>A spectrogram is a grid of amplitudes with a frequency resolution and a time resolution. When we measure over a short time window, we have high time resolution but cannot accurately count the cycles so we have wide frequency bands. When we measure over a long time window, we have narrow/precise frequency bands but looser time resolution." +
      " For reading speech, we prefer the high time resolution with wide frequency bands. It may hide individual harmonic stripes but it makes it easier to follow broader patterns and quick changing sounds. We additionally map frequencies to the human-perceptual Mel scale and apply a logarithmic decibel scaling to amplitudes.</p></div>" +
      "</div>" +
      '<div class="mel-comparison">' +
      "<figure><figcaption><b>Narrowband</b><span>The same recording, analyzed with a long window</span></figcaption>" +
      '<div class="real-spectrogram-shell"><span class="real-spectrogram-y"><i>8 kHz</i><b>frequency</b><i>0 Hz</i></span><img src="/images/demo/spectrogram-tutorial/comparison/pronunciation-narrowband.webp" alt="Narrowband spectrogram of the phrase Can we check the pronunciation"><span class="real-spectrogram-x">time →</span></div>' +
      "</figure>" +
      "<figure><figcaption><b>Wideband</b><span>The same recording, analyzed with a short window</span></figcaption>" +
      '<div class="real-spectrogram-shell"><span class="real-spectrogram-y"><i>8 kHz</i><b>frequency</b><i>0 Hz</i></span><img src="/images/demo/spectrogram-tutorial/comparison/pronunciation-wideband.webp" alt="Wideband spectrogram of the phrase Can we check the pronunciation"><span class="real-spectrogram-x">time →</span></div>' +
      "</figure>" +
      "</div>" +
      '<div class="mel-explainer-grid">' +
      "<article><b>Window length</b><p>Narrowband reveals individual horizontal harmonics. Wideband sharpens timing and merges harmonics into broad formant bands.</p></article>" +
      "<article><b>Log amplitude</b><p>A whisper and a shout span a huge physical range. Decibels compress it into a scale closer to perception.</p></article>" +
      "<article><b>Mel frequency</b><p>Hearing resolves low-frequency differences more finely than equally sized high-frequency differences, so mel bins are non-linear.</p></article>" +
      "<article><b>Reading rule</b><p>Read timing, dark horizontal bands, bursts, gaps, and noisy regions. Black means more energy; white means less.</p></article>" +
      "</div>";
  }

  function renderFormantLesson() {
    const pointButtons = vowels
      .map(function (vowel) {
        const plotX = 100 - (((vowel.f2 - 700) / 1700) * 86 + 7);
        const plotY = ((vowel.f1 - 250) / 650) * 82 + 8;
        return (
          '<button type="button" data-formant-vowel="' +
          vowel.symbol +
          '" style="--x:' +
          Math.max(5, Math.min(95, plotX)) +
          "%;--y:" +
          Math.max(5, Math.min(95, plotY)) +
          '%" aria-label="Show formants for /' +
          vowel.symbol +
          "/ as in " +
          vowel.example +
          '">' +
          vowel.symbol +
          "</button>"
        );
      })
      .join("");
    stage.innerHTML =
      '<div class="spectrogram-lesson-heading">' +
      '<span class="spectrogram-lesson-number">08</span>' +
      "<div><h3>Formants are the vocal tract&apos;s favorite frequencies</h3>" +
      "<p>Your vocal folds provide harmonics; the throat and mouth amplify some frequency neighborhoods. Those resonance peaks are F1, F2, F3, and so on. The exact pitches will vary with the speaker, but the frequency ratios of the formants determine the vowel sound.</p></div>" +
      "</div>" +
      '<div class="formant-lab">' +
      '<figure class="formant-observation"><div class="formant-view-bar"><div role="group" aria-label="Formant view"><button type="button" data-formant-view="spectrum" aria-pressed="true">Spectrum</button><button type="button" data-formant-view="spectrogram" aria-pressed="false">Spectrogram</button></div><figcaption data-formant-view-caption>Peaks reveal resonances</figcaption></div>' +
      "<canvas data-formant-spectrum></canvas>" +
      '<div class="formant-spectrogram" data-formant-spectrogram hidden><span class="real-spectrogram-y"><i>8 kHz</i><b>frequency</b><i>0 Hz</i></span><div class="formant-image-wrap"><img data-formant-spectrogram-image src="' +
      soundImage(vowels[0].symbol) +
      '" alt="Spectrogram of /i/ as in fleece"><span class="formant-band is-f1" data-formant-band="f1"><b>F1</b></span><span class="formant-band is-f2" data-formant-band="f2"><b>F2</b></span></div><span class="real-spectrogram-x">time →</span></div>' +
      "</figure>" +
      '<div class="formant-vowel-plot">' +
      '<span class="formant-axis is-front">front / high F2</span><span class="formant-axis is-back">back / low F2</span>' +
      '<span class="formant-axis is-close">close / low F1</span><span class="formant-axis is-open">open / high F1</span>' +
      pointButtons +
      "</div>" +
      "</div>" +
      '<div class="formant-reading">' +
      "<div><span>Selected vowel</span><strong data-formant-symbol>/i/</strong><small data-formant-example>as in fleece</small></div>" +
      "<div><span>F1</span><strong data-formant-f1>280 Hz</strong><small>Higher F1 usually means a more open vowel.</small></div>" +
      "<div><span>F2</span><strong data-formant-f2>2230 Hz</strong><small>Higher F2 usually means a more front vowel.</small></div>" +
      "</div>" +
      '<div class="spectrogram-callout"><b>Memory trick:</b> plotting the first two formants yields the familiar vowel chart. A low first formant means a closed vowel and a low second formant means a back vowel. If you can remember the vowel chart, you can read vowels from a spectrogram.</div>';

    const canvas = stage.querySelector("[data-formant-spectrum]");
    const symbolOutput = stage.querySelector("[data-formant-symbol]");
    const exampleOutput = stage.querySelector("[data-formant-example]");
    const f1Output = stage.querySelector("[data-formant-f1]");
    const f2Output = stage.querySelector("[data-formant-f2]");
    const spectrogram = stage.querySelector("[data-formant-spectrogram]");
    const spectrogramImage = stage.querySelector(
      "[data-formant-spectrogram-image]",
    );
    const spectrumCaption = stage.querySelector("[data-formant-view-caption]");
    const f1Band = stage.querySelector('[data-formant-band="f1"]');
    const f2Band = stage.querySelector('[data-formant-band="f2"]');
    let selected = vowels[0];
    const redraw = canvasRenderer(canvas, function (context, width, height) {
      context.fillStyle = "#fffaf4";
      context.fillRect(0, 0, width, height);
      context.strokeStyle = "#d5ddde";
      context.beginPath();
      context.moveTo(30, 12);
      context.lineTo(30, height - 28);
      context.lineTo(width - 12, height - 28);
      context.stroke();
      context.strokeStyle = "#17877f";
      context.lineWidth = 4;
      context.beginPath();
      for (let x = 30; x < width - 10; x += 2) {
        const hz = ((x - 30) / (width - 42)) * 3000;
        const first = Math.exp(-Math.pow((hz - selected.f1) / 170, 2)) * 0.82;
        const second = Math.exp(-Math.pow((hz - selected.f2) / 250, 2)) * 0.72;
        const base = 0.08 + Math.sin(hz * 0.045) * 0.025;
        const y = height - 28 - (base + first + second) * (height - 50);
        if (x === 30) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
      [
        [selected.f1, "F1", "#ef7b61"],
        [selected.f2, "F2", "#7659c7"],
      ].forEach(function (item) {
        const x = 30 + (item[0] / 3000) * (width - 42);
        context.strokeStyle = item[2];
        context.setLineDash([5, 5]);
        context.beginPath();
        context.moveTo(x, 18);
        context.lineTo(x, height - 28);
        context.stroke();
        context.setLineDash([]);
        context.fillStyle = item[2];
        context.font = "800 13px system-ui, sans-serif";
        context.fillText(item[1], x + 5, 22);
      });
      axisLabel(context, "frequency →", width / 2, height - 7);
    });

    function updateSelectedVowel(button) {
      symbolOutput.textContent = "/" + selected.symbol + "/";
      exampleOutput.textContent = "as in " + selected.example;
      f1Output.textContent = selected.f1 + " Hz";
      f2Output.textContent = selected.f2 + " Hz";
      spectrogramImage.src = soundImage(selected.symbol);
      spectrogramImage.alt =
        "Spectrogram of /" + selected.symbol + "/ as in " + selected.example;
      f1Band.style.setProperty("--y", 100 - (selected.f1 / 8000) * 100 + "%");
      f2Band.style.setProperty("--y", 100 - (selected.f2 / 8000) * 100 + "%");
      redraw();
      if (button) playPhoneme(selected.symbol, button);
    }

    stage.querySelectorAll("[data-formant-view]").forEach(function (button) {
      listen(button, "click", function () {
        const showSpectrum = button.dataset.formantView === "spectrum";
        canvas.hidden = !showSpectrum;
        spectrogram.hidden = showSpectrum;
        spectrumCaption.textContent = showSpectrum
          ? "Peaks reveal resonances"
          : "Formants persist through time";
        stage.querySelectorAll("[data-formant-view]").forEach(function (item) {
          item.setAttribute(
            "aria-pressed",
            String(item.dataset.formantView === button.dataset.formantView),
          );
        });
      });
    });

    stage.querySelectorAll("[data-formant-vowel]").forEach(function (button) {
      listen(button, "click", function () {
        selected =
          vowels.find(function (vowel) {
            return vowel.symbol === button.dataset.formantVowel;
          }) || vowels[0];
        stage.querySelectorAll("[data-formant-vowel]").forEach(function (item) {
          item.classList.toggle("is-active", item === button);
        });
        updateSelectedVowel(button);
      });
    });
    stage.querySelector("[data-formant-vowel]").classList.add("is-active");
    updateSelectedVowel();
  }

  const formantQuestions = [
    {
      symbol: "i",
      vowel: "/i/ as in fleece",
      f1: 300,
      f2: 2250,
      choices: [
        "F1 ≈ 300, F2 ≈ 2250",
        "F1 ≈ 750, F2 ≈ 1100",
        "F1 ≈ 550, F2 ≈ 900",
      ],
    },
    {
      symbol: "æ",
      vowel: "/æ/ as in trap",
      f1: 820,
      f2: 1650,
      choices: [
        "F1 ≈ 350, F2 ≈ 900",
        "F1 ≈ 820, F2 ≈ 1650",
        "F1 ≈ 450, F2 ≈ 2200",
      ],
    },
    {
      symbol: "u",
      vowel: "/u/ as in goose",
      f1: 330,
      f2: 950,
      choices: [
        "F1 ≈ 700, F2 ≈ 1900",
        "F1 ≈ 330, F2 ≈ 950",
        "F1 ≈ 850, F2 ≈ 1400",
      ],
    },
  ];

  function renderFormantQuiz() {
    stage.classList.add("is-formant-quiz");
    stage.innerHTML =
      '<div class="spectrogram-lesson-heading">' +
      '<span class="spectrogram-lesson-number">09</span>' +
      "<div><h3>Reading vowels using formants</h3>" +
      "<p>Ignore the fine harmonic stripes. F1 is the lowest broad band; F2 is the next broad band above it. Estimate their frequencies from the vertical axis.</p></div>" +
      "</div>" +
      '<div class="formant-quiz">' +
      '<div class="formant-quiz-visual"><span class="real-spectrogram-y"><i>8 kHz</i><b>frequency</b><i>0 Hz</i></span><img data-formant-quiz-image alt=""><span class="real-spectrogram-x">time →</span></div>' +
      '<div class="formant-quiz-copy"><span data-formant-question-number>Vowel 1 of 3</span><h4 data-formant-question-vowel></h4><p>Which pair best matches the centers of F1 and F2?</p><div data-formant-options></div><p data-formant-feedback aria-live="polite"></p></div>' +
      "</div>";
    let questionIndex = 0;
    const image = stage.querySelector("[data-formant-quiz-image]");
    const number = stage.querySelector("[data-formant-question-number]");
    const vowel = stage.querySelector("[data-formant-question-vowel]");
    const options = stage.querySelector("[data-formant-options]");
    const feedback = stage.querySelector("[data-formant-feedback]");

    function showQuestion() {
      const question = formantQuestions[questionIndex];
      number.textContent =
        "Vowel " + (questionIndex + 1) + " of " + formantQuestions.length;
      vowel.textContent = question.vowel;
      image.src = soundImage(question.symbol);
      image.alt = "Real wideband spectrogram of " + question.vowel;
      feedback.textContent = "";
      options.innerHTML = question.choices
        .map(function (choice, index) {
          return (
            '<button type="button" data-formant-choice="' +
            index +
            '">' +
            choice +
            "</button>"
          );
        })
        .join("");
      options.querySelectorAll("button").forEach(function (button) {
        listen(button, "click", function () {
          const correct =
            button.textContent.includes("F1 ≈ " + question.f1) &&
            button.textContent.includes("F2 ≈ " + question.f2);
          options.querySelectorAll("button").forEach(function (item) {
            item.disabled = true;
            item.classList.toggle(
              "is-correct",
              item.textContent.includes("F1 ≈ " + question.f1) &&
                item.textContent.includes("F2 ≈ " + question.f2),
            );
          });
          button.classList.toggle("is-wrong", !correct);
          feedback.textContent = correct
            ? "Exactly. Read low-to-high: F1 first, then F2."
            : "Close. Follow the centers of the two broadest low-frequency bands.";
          if (questionIndex < formantQuestions.length - 1) {
            const advance = document.createElement("button");
            advance.type = "button";
            advance.className = "formant-next-question";
            advance.textContent = "Next vowel";
            feedback.append(" ", advance);
            advance.addEventListener(
              "click",
              function () {
                questionIndex += 1;
                showQuestion();
              },
              { once: true },
            );
          }
        });
      });
    }
    showQuestion();
  }

  const soundPatterns = {
    plosive: {
      title: "Plosives",
      text: "A complete closure briefly removes most energy; release makes a sharp burst. The following transition carries useful place and voicing cues.",
      details: [
        "Voiceless /p t k/: closure, burst, then a stretch of aspiration before voicing begins.",
        "Voiced /b d ɡ/: a low voice bar or periodic striations may continue through part of the closure.",
        "Place is read from the burst spectrum together with the formant transitions into the next vowel.",
      ],
      examples: [
        ["p", "/p/ voiceless"],
        ["b", "/b/ voiced"],
        ["t", "/t/ voiceless"],
        ["d", "/d/ voiced"],
        ["k", "/k/ voiceless"],
        ["ɡ", "/ɡ/ voiced"],
      ],
    },
    fricative: {
      title: "Fricatives",
      text: "A narrow constriction turns airflow into sustained aperiodic noise. Its vertical position and darkness help identify where the constriction occurs.",
      details: [
        "Voiced /v ð z ʒ/: turbulence is combined with a low voice bar or periodic vertical pulses.",
        "Voiceless /f θ s ʃ/: the noisy interval lacks regular vocal-fold pulses.",
        "/s/ has the highest average noise frequency; /ʃ/ is lower; /f/ and /θ/ are weaker and more diffuse.",
        "/h/ is diffuse glottal noise shaped by the following vowel, rather than a stable oral-frication band.",
      ],
      examples: [
        ["f", "/f/"],
        ["v", "/v/"],
        ["θ", "/θ/"],
        ["ð", "/ð/"],
        ["s", "/s/"],
        ["z", "/z/"],
        ["ʃ", "/ʃ/"],
        ["ʒ", "/ʒ/"],
        ["h", "/h/"],
      ],
    },
    nasal: {
      title: "Nasals",
      text: "Air through the nose adds a strong low nasal murmur while resonances and anti-resonances weaken patches higher in the spectrum.",
      details: [
        "Regular voicing remains visible, but upper formants are usually dimmer than in neighboring vowels.",
        "Transitions into and out of the nasal help distinguish lip /m/, tongue-tip /n/, and velar /ŋ/ closures.",
      ],
      examples: [
        ["m", "/m/ bilabial"],
        ["n", "/n/ alveolar"],
        ["ŋ", "/ŋ/ velar"],
      ],
    },
    tap: {
      title: "Tap / flap",
      text: "The tongue makes one extremely quick alveolar contact, often replacing /t/ or /d/ between vowels in North American English.",
      details: [
        "Look for a hairline closure or abrupt weakening between two vowel-like regions.",
        "Because the gesture is so short, its neighboring formant transitions often carry more information than the closure itself.",
      ],
      examples: [["ɾ", "/ɾ/ in water"]],
    },
    vowel: {
      title: "Vowels",
      text: "Regular vocal-fold vibration produces harmonics, while the vocal tract emphasizes broad horizontal bands called formants.",
      details: [
        "F1 tracks tongue height inversely: a close vowel has a lower F1.",
        "F2 broadly tracks frontness: a front vowel has a higher F2.",
        "A steady vowel has nearly horizontal formants; context bends them near consonant boundaries.",
      ],
      examples: [
        ["i", "/i/ fleece"],
        ["ɪ", "/ɪ/ kit"],
        ["ɛ", "/ɛ/ dress"],
        ["æ", "/æ/ trap"],
        ["ɑ", "/ɑ/ palm"],
        ["ɔ", "/ɔ/ thought"],
        ["ʊ", "/ʊ/ foot"],
        ["u", "/u/ goose"],
        ["ə", "/ə/ about"],
      ],
    },
    diphthong: {
      title: "Diphthongs",
      text: "A diphthong moves from one vowel target toward another, so F1 and F2 curve rather than staying level.",
      details: [
        "Read the starting and ending formant positions as one continuous gesture.",
        "The direction of F1 and F2 movement distinguishes otherwise similar-looking vowel nuclei.",
      ],
      examples: [
        ["eɪ", "/eɪ/ face"],
        ["aɪ", "/aɪ/ price"],
        ["aʊ", "/aʊ/ mouth"],
        ["oʊ", "/oʊ/ goat"],
        ["ɔɪ", "/ɔɪ/ choice"],
      ],
    },
    approximant: {
      title: "Approximants",
      text: "The vocal tract narrows without turbulent friction. The result remains voiced and vowel-like, but usually weaker and more transitional.",
      details: [
        "/w/ and /j/ make rapid formant movements from vowel-like positions.",
        "/ɹ/ lowers F3 toward F2; /l/ often has a strong low band and weakened higher energy.",
      ],
      examples: [
        ["w", "/w/"],
        ["j", "/j/"],
        ["ɹ", "/ɹ/"],
        ["l", "/l/"],
      ],
    },
    affricate: {
      title: "Affricates",
      text: "An affricate begins as a stop closure and releases into a sustained fricative, so both signatures appear in sequence.",
      details: [
        "/tʃ/ has a closure, burst, and voiceless postalveolar noise.",
        "/dʒ/ has the same basic sequence with voicing cues and often a shorter noisy interval.",
      ],
      examples: [
        ["tʃ", "/tʃ/ voiceless"],
        ["dʒ", "/dʒ/ voiced"],
      ],
    },
  };
  const patternOrder = [
    "vowel",
    "diphthong",
    "plosive",
    "fricative",
    "affricate",
    "nasal",
    "approximant",
    "tap",
  ];

  function renderPatternLesson() {
    stage.innerHTML =
      '<div class="spectrogram-lesson-heading">' +
      '<span class="spectrogram-lesson-number">10</span>' +
      "<div><h3>Recognize speech sounds by their spectrogram patterns</h3>" +
      "<p>Choose a sound category, then select an IPA example to hear it and compare its real spectrogram. Read timing, frequency, and neighboring cues together.</p></div>" +
      "</div>" +
      '<div class="pattern-tabs" role="tablist" aria-label="Speech sound categories">' +
      patternOrder
        .map(function (key, index) {
          return (
            '<button type="button" role="tab" data-pattern="' +
            key +
            '"' +
            (index === 0 ? ' aria-selected="true"' : ' aria-selected="false"') +
            ">" +
            soundPatterns[key].title +
            "</button>"
          );
        })
        .join("") +
      "</div>" +
      '<div class="pattern-unit">' +
      '<div class="real-spectrogram-shell pattern-spectrogram">' +
      '<div class="real-spectrogram-y"><span>high</span><b>frequency</b><span>low</span></div>' +
      '<img data-pattern-image alt="">' +
      '<span class="real-spectrogram-x">time &#8594;</span>' +
      "</div>" +
      '<div class="pattern-copy"><span data-pattern-kicker>Real audio example</span><h4 data-pattern-title></h4>' +
      "<p data-pattern-text></p><ul data-pattern-details></ul>" +
      '<div class="pattern-example-buttons" data-pattern-examples aria-label="Choose and play a sound example"></div></div>' +
      "</div>" +
      '<div class="pattern-check">' +
      '<div class="pattern-check-heading"><span>Quick check</span><h4>Which fricative is shown?</h4></div>' +
      '<div class="pattern-check-body"><div class="real-spectrogram-shell pattern-quiz-spectrogram">' +
      '<div class="real-spectrogram-y"><span>high</span><b>frequency</b><span>low</span></div><img src="' +
      soundImage("ʃ") +
      '" alt="Real wideband spectrogram of an unidentified fricative">' +
      '<span class="real-spectrogram-x">time &#8594;</span>' +
      '</div><div class="pattern-check-answers"><div data-pattern-quiz-options></div><p data-pattern-quiz-feedback aria-live="polite"></p></div></div>' +
      "</div>";
    const image = stage.querySelector("[data-pattern-image]");
    const title = stage.querySelector("[data-pattern-title]");
    const text = stage.querySelector("[data-pattern-text]");
    const details = stage.querySelector("[data-pattern-details]");
    const examples = stage.querySelector("[data-pattern-examples]");
    let active = patternOrder[0];

    function showPattern(key) {
      active = key;
      const pattern = soundPatterns[key];
      title.textContent = pattern.title;
      text.textContent = pattern.text;
      details.innerHTML = pattern.details
        .map(function (detail) {
          return "<li>" + detail + "</li>";
        })
        .join("");
      examples.innerHTML = pattern.examples
        .map(function (example, index) {
          return (
            '<button type="button" data-pattern-symbol="' +
            example[0] +
            '"' +
            (index === 0 ? ' aria-pressed="true"' : ' aria-pressed="false"') +
            ">" +
            example[1] +
            "</button>"
          );
        })
        .join("");
      image.src = soundImage(pattern.examples[0][0]);
      image.alt = "Real wideband spectrogram of " + pattern.examples[0][1];
      examples.querySelectorAll("button").forEach(function (button) {
        listen(button, "click", function () {
          const symbol = button.dataset.patternSymbol;
          image.src = soundImage(symbol);
          image.alt = "Real wideband spectrogram of " + button.textContent;
          examples.querySelectorAll("button").forEach(function (item) {
            item.setAttribute("aria-pressed", String(item === button));
          });
          playPhoneme(symbol, button);
        });
      });
      stage.querySelectorAll("[data-pattern]").forEach(function (button) {
        const selected = button.dataset.pattern === key;
        button.setAttribute("aria-selected", String(selected));
      });
    }
    stage.querySelectorAll("[data-pattern]").forEach(function (button) {
      listen(button, "click", function () {
        showPattern(button.dataset.pattern);
      });
    });
    const quizOptions = stage.querySelector("[data-pattern-quiz-options]");
    const quizFeedback = stage.querySelector("[data-pattern-quiz-feedback]");
    quizOptions.innerHTML = [
      ["s", "/s/"],
      ["ʃ", "/ʃ/"],
      ["f", "/f/ or /θ/"],
      ["h", "/h/"],
    ]
      .map(function (option) {
        return (
          '<button type="button" data-pattern-answer="' +
          option[0] +
          '">' +
          option[1] +
          "</button>"
        );
      })
      .join("");
    quizOptions.querySelectorAll("button").forEach(function (button) {
      listen(button, "click", function () {
        const correct = button.dataset.patternAnswer === "ʃ";
        quizOptions.querySelectorAll("button").forEach(function (item) {
          item.disabled = true;
          item.classList.toggle(
            "is-correct",
            item.dataset.patternAnswer === "ʃ",
          );
        });
        button.classList.toggle("is-correct", correct);
        button.classList.toggle("is-wrong", !correct);
        quizFeedback.textContent = correct
          ? "Yes. Its concentrated noise is lower than /s/, but stronger than diffuse /f/ or /θ/."
          : "Compare the center of the noise: lower than /s/, but more concentrated than /f/, /θ/, or /h/.";
      });
    });
    showPattern(active);
  }

  function renderEndlessLesson() {
    stage.classList.add("is-practice");
    if (!exercises.length) {
      stage.innerHTML =
        '<p class="spectrogram-empty">Practice assets are unavailable.</p>';
      return;
    }
    const exercise = exercises[endlessIndex % exercises.length];
    const candidates = exercise.candidates || [];
    stage.innerHTML =
      '<div class="spectrogram-lesson-heading">' +
      '<span class="spectrogram-lesson-number">11</span>' +
      "<div><h3>Find “" +
      escapeHtml(exercise.answer) +
      "” in the spectrogram</h3>" +
      "<p>Several word-sized regions are marked. Use their boundaries and sound signatures to choose the region that contains the target word.</p></div>" +
      "</div>" +
      '<div class="endless-score"><span><b data-endless-correct>' +
      endlessCorrect +
      "</b> correct</span><span><b data-endless-attempts>" +
      endlessAttempts +
      "</b> attempts</span><span>Example " +
      ((endlessIndex % exercises.length) + 1) +
      " of " +
      exercises.length +
      "</span></div>" +
      '<div class="endless-spectrogram">' +
      '<div class="spectrogram-frequency-axis"><span>high</span><span>frequency</span><span>low</span></div>' +
      '<div class="endless-image-wrap"><img src="' +
      exercise.image +
      '" alt="Grayscale log-mel spectrogram with several selectable word regions">' +
      candidates
        .map(function (candidate, index) {
          return (
            '<button type="button" class="practice-region" data-practice-region="' +
            index +
            '" style="--start:' +
            candidate.start * 100 +
            "%;--width:" +
            (candidate.end - candidate.start) * 100 +
            '%" aria-label="Choose region ' +
            String.fromCharCode(65 + index) +
            '"><span>' +
            String.fromCharCode(65 + index) +
            "</span></button>"
          );
        })
        .join("") +
      "</div>" +
      '<span class="spectrogram-time-axis">time →</span>' +
      "</div>" +
      '<div class="endless-actions">' +
      '<button type="button" data-endless-hint>Reveal sentence</button>' +
      '<button type="button" data-endless-next>Another example</button>' +
      "</div>" +
      '<p class="endless-hint" data-endless-hint-copy hidden><b>Sentence:</b> ' +
      escapeHtml(exercise.sentence) +
      '</p><p class="endless-feedback" data-endless-feedback aria-live="polite"></p>';

    const feedback = stage.querySelector("[data-endless-feedback]");
    const hintButton = stage.querySelector("[data-endless-hint]");
    const hint = stage.querySelector("[data-endless-hint-copy]");
    const another = stage.querySelector("[data-endless-next]");

    listen(hintButton, "click", function () {
      hint.hidden = false;
      hintButton.disabled = true;
    });
    stage.querySelectorAll("[data-practice-region]").forEach(function (button) {
      listen(button, "click", function () {
        const candidate = candidates[Number(button.dataset.practiceRegion)];
        const correct = Boolean(candidate && candidate.correct);
        endlessAttempts += 1;
        if (correct) endlessCorrect += 1;
        stage.querySelector("[data-endless-correct]").textContent =
          endlessCorrect;
        stage.querySelector("[data-endless-attempts]").textContent =
          endlessAttempts;
        button.classList.add(correct ? "is-correct" : "is-wrong");
        feedback.className =
          "endless-feedback " + (correct ? "is-correct" : "is-wrong");
        feedback.innerHTML = correct
          ? "<b>Exactly.</b> That region contains “" +
            escapeHtml(exercise.answer) +
            ".”"
          : "<b>Not that region.</b> Compare its boundaries and internal cues with the target word.";
        if (correct) {
          stage
            .querySelectorAll("[data-practice-region]")
            .forEach(function (item, index) {
              item.disabled = true;
              item.classList.toggle(
                "is-correct",
                Boolean(candidates[index].correct),
              );
            });
          another.focus({ preventScroll: true });
        }
      });
    });
    listen(another, "click", function () {
      stopLesson();
      endlessIndex = (endlessIndex + 1) % exercises.length;
      renderEndlessLesson();
      const firstRegion = stage.querySelector("[data-practice-region]");
      if (firstRegion) firstRegion.focus({ preventScroll: true });
    });
  }

  const renderers = [
    renderIpaLesson,
    renderDialectLesson,
    renderPressureLesson,
    renderWaveLesson,
    renderSpectrumLesson,
    renderStackLesson,
    renderMelLesson,
    renderFormantLesson,
    renderFormantQuiz,
    renderPatternLesson,
    renderEndlessLesson,
  ];

  function renderNavigation() {
    nav.innerHTML = lessonMeta
      .map(function (lesson, index) {
        return (
          '<button type="button" data-lesson-index="' +
          index +
          '" aria-label="Lesson ' +
          (index + 1) +
          ": " +
          lesson[0] +
          '"><span>' +
          (index + 1) +
          "</span><small>" +
          lesson[1] +
          "</small></button>"
        );
      })
      .join("");
    nav.querySelectorAll("button").forEach(function (button) {
      button.addEventListener("click", function () {
        showLesson(Number(button.dataset.lessonIndex), true);
      });
    });
  }

  function showLesson(index, focusStage) {
    stopLesson();
    currentLesson = Math.max(0, Math.min(renderers.length - 1, index));
    renderers[currentLesson]();
    previousButton.disabled = currentLesson === 0;
    nextButton.innerHTML =
      currentLesson === renderers.length - 1
        ? 'New exercise <span aria-hidden="true">&#10217;</span>'
        : 'Next <span aria-hidden="true">&#10217;</span>';
    nav.querySelectorAll("button").forEach(function (button, buttonIndex) {
      const active = buttonIndex === currentLesson;
      button.classList.toggle("is-active", active);
      if (active) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
    if (focusStage) stage.focus({ preventScroll: true });
  }

  previousButton.addEventListener("click", function () {
    showLesson(currentLesson - 1, true);
  });
  nextButton.addEventListener("click", function () {
    if (currentLesson === renderers.length - 1) {
      stopLesson();
      endlessIndex = (endlessIndex + 1) % Math.max(exercises.length, 1);
      renderEndlessLesson();
      return;
    }
    showLesson(currentLesson + 1, true);
  });

  renderNavigation();
  showLesson(0, false);
})();

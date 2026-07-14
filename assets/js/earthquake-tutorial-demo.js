(() => {
  const root = document.querySelector("[data-earthquake-demo]");
  if (!root) return;

  const mapCanvas = root.querySelector("[data-earthquake-map]");
  const moveoutCanvas = root.querySelector("[data-earthquake-moveout]");
  const feedback = root.querySelector("[data-earthquake-feedback]");
  const round = root.querySelector("[data-earthquake-round]");
  const next = root.querySelector("[data-earthquake-next]");
  const choices = [...root.querySelectorAll("[data-earthquake-choice]")];

  const stations = [
    { id: "ALP", x: 0.48, y: 0.3, distance: 18 },
    { id: "BIR", x: 0.66, y: 0.27, distance: 31 },
    { id: "CED", x: 0.82, y: 0.44, distance: 46 },
    { id: "DUN", x: 0.76, y: 0.68, distance: 58 },
    { id: "ELK", x: 0.54, y: 0.82, distance: 72 },
    { id: "FER", x: 0.19, y: 0.7, distance: 86 },
  ];
  const duration = 23;
  let activeScenario = 0;
  let answer = null;

  const scenarios = [
    {
      kind: "earthquake",
      title: "Earthquake",
      feedback:
        "This is a likely earthquake. At progressively farther stations, a small P arrival appears later, followed by a stronger S arrival that also moves later. That paired, distance-ordered moveout is the tell.",
      wrong:
        "This is a likely earthquake. The small P arrival and stronger S arrival both move later with station distance, a coherent wavefront pattern rather than isolated local noise.",
    },
    {
      kind: "noise",
      title: "Non-earthquake noise",
      feedback:
        "This is unlikely to be an earthquake. The large peaks do not arrive in distance order, and there is no consistent small-P then large-S pair shared across the network.",
      wrong:
        "This is unlikely to be an earthquake. The stations have unrelated bursts: no coherent moveout and no consistent P-then-S pairing.",
    },
  ];

  function noise(t, index, scenario) {
    return (
      0.035 * Math.sin(t * 7.2 + index * 1.7) +
      0.024 * Math.sin(t * 12.8 + index * 3.1) +
      0.018 * Math.sin(t * 20.1 + scenario * 2.4 + index)
    );
  }

  function burst(t, arrival, amplitude, frequency) {
    const delta = t - arrival;
    if (delta < 0 || delta > 2.7) return 0;
    return amplitude * Math.exp(-delta * 1.18) * Math.sin(delta * frequency);
  }

  function arrivals(station, index, scenario) {
    if (scenario.kind === "earthquake") {
      return {
        p: 2.15 + station.distance / 8.2,
        s: 2.75 + station.distance / 4.55,
      };
    }
    const falseP = [5.0, 8.7, 5.8, 12.1, 7.2, 10.0][index];
    const falseS = [6.1, 10.0, 6.9, 13.2, 8.8, 11.0][index];
    return { p: falseP, s: falseS };
  }

  function waveform(t, station, index, scenario) {
    const arrival = arrivals(station, index, scenario);
    if (scenario.kind === "earthquake") {
      return (
        noise(t, index, activeScenario) +
        burst(t, arrival.p, 0.22, 17 + index) +
        burst(t, arrival.s, 0.63, 11 + index * 0.45)
      );
    }
    const extra = [15.3, 4.1, 17.5, 9.3, 14.4, 6.4][index];
    return (
      noise(t, index, activeScenario) +
      burst(t, arrival.p, 0.56, 10 + index * 0.5) +
      burst(t, arrival.s, 0.2, 17 + index) +
      burst(t, extra, 0.38, 14)
    );
  }

  function sizeCanvas(canvas) {
    const ratio = window.devicePixelRatio || 1;
    const bounds = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(bounds.width * ratio));
    const height = Math.max(1, Math.round(bounds.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width: bounds.width, height: bounds.height };
  }

  function drawMap() {
    const { context, width, height } = sizeCanvas(mapCanvas);
    const padding = { top: 18, right: 18, bottom: 28, left: 32 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const px = (x) => padding.left + x * plotWidth;
    const py = (y) => padding.top + y * plotHeight;

    context.clearRect(0, 0, width, height);
    context.fillStyle = "#edf6f4";
    context.fillRect(padding.left, padding.top, plotWidth, plotHeight);
    context.strokeStyle = "rgba(35, 49, 61, 0.16)";
    context.lineWidth = 1;
    for (let tick = 0; tick <= 4; tick += 1) {
      const x = padding.left + (plotWidth * tick) / 4;
      const y = padding.top + (plotHeight * tick) / 4;
      context.beginPath();
      context.moveTo(x, padding.top);
      context.lineTo(x, padding.top + plotHeight);
      context.moveTo(padding.left, y);
      context.lineTo(padding.left + plotWidth, y);
      context.stroke();
    }
    const source = { x: 0.42, y: 0.5 };
    stations.forEach((station) => {
      context.strokeStyle = "rgba(14, 116, 109, 0.23)";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(px(source.x), py(source.y));
      context.lineTo(px(station.x), py(station.y));
      context.stroke();
    });

    context.fillStyle = "#ef7954";
    context.beginPath();
    context.arc(px(source.x), py(source.y), 7, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "#23313d";
    context.font = "700 11px Nunito, system-ui, sans-serif";
    context.fillText("candidate source", px(source.x) + 10, py(source.y) - 9);

    stations.forEach((station) => {
      const x = px(station.x);
      const y = py(station.y);
      context.fillStyle = "#0e746d";
      context.beginPath();
      context.arc(x, y, 5, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = "#23313d";
      context.font = "700 10px Nunito, system-ui, sans-serif";
      const label = `${station.id} ${station.distance} km`;
      const labelX = x > padding.left + plotWidth * 0.67 ? x - context.measureText(label).width - 8 : x + 8;
      context.fillText(label, labelX, y - 7);
    });
    context.fillStyle = "#5b6d78";
    context.font = "10px Nunito, system-ui, sans-serif";
    context.fillText("longitude", padding.left + plotWidth / 2 - 22, height - 7);
    context.save();
    context.translate(10, padding.top + plotHeight / 2 + 20);
    context.rotate(-Math.PI / 2);
    context.fillText("latitude", 0, 0);
    context.restore();
  }

  function drawMoveout() {
    const { context, width, height } = sizeCanvas(moveoutCanvas);
    const scenario = scenarios[activeScenario];
    const padding = { top: 18, right: 14, bottom: 28, left: 56 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const x = (time) => padding.left + (time / duration) * plotWidth;
    const row = plotHeight / stations.length;

    context.clearRect(0, 0, width, height);
    context.fillStyle = "#fffdf8";
    context.fillRect(padding.left, padding.top, plotWidth, plotHeight);
    context.strokeStyle = "rgba(35, 49, 61, 0.14)";
    context.lineWidth = 1;
    for (let tick = 0; tick <= duration; tick += 5) {
      context.beginPath();
      context.moveTo(x(tick), padding.top);
      context.lineTo(x(tick), padding.top + plotHeight);
      context.stroke();
      context.fillStyle = "#5b6d78";
      context.font = "10px Nunito, system-ui, sans-serif";
      context.fillText(`${tick}s`, x(tick) - 7, height - 7);
    }

    if (answer) {
      ["p", "s"].forEach((phase) => {
        context.strokeStyle = phase === "p" ? "#2f73d9" : "#ef7954";
        context.lineWidth = 1.5;
        context.setLineDash([4, 4]);
        context.beginPath();
        stations.forEach((station, index) => {
          const arrival = arrivals(station, index, scenario)[phase];
          const y = padding.top + row * (index + 0.5);
          if (index === 0) context.moveTo(x(arrival), y);
          else context.lineTo(x(arrival), y);
        });
        context.stroke();
        context.setLineDash([]);
      });
    }

    stations.forEach((station, index) => {
      const baseline = padding.top + row * (index + 0.5);
      context.strokeStyle = "rgba(35, 49, 61, 0.18)";
      context.beginPath();
      context.moveTo(padding.left, baseline);
      context.lineTo(padding.left + plotWidth, baseline);
      context.stroke();
      context.strokeStyle = "#0e746d";
      context.lineWidth = 1.45;
      context.beginPath();
      for (let point = 0; point <= 360; point += 1) {
        const time = (point / 360) * duration;
        const y = baseline - waveform(time, station, index, scenario) * row * 0.62;
        if (point === 0) context.moveTo(x(time), y);
        else context.lineTo(x(time), y);
      }
      context.stroke();
      context.fillStyle = "#23313d";
      context.font = "700 10px Nunito, system-ui, sans-serif";
      context.fillText(`${station.id}`, 8, baseline + 3);
    });

    if (answer) {
      context.font = "700 10px Nunito, system-ui, sans-serif";
      context.fillStyle = "#2f73d9";
      context.fillText("P: first, smaller", padding.left + 4, padding.top - 5);
      context.fillStyle = "#ef7954";
      context.fillText("S: later, larger", padding.left + 111, padding.top - 5);
    }
  }

  function render() {
    const scenario = scenarios[activeScenario];
    root.classList.remove("is-correct", "is-incorrect");
    round.textContent = `Trace set ${activeScenario + 1} of ${scenarios.length}`;
    choices.forEach((button) => {
      button.dataset.selected = answer === button.dataset.earthquakeChoice ? "true" : "false";
    });
    if (!answer) {
      feedback.textContent = "Look for a small P arrival followed by a stronger S arrival, both moving later as distance increases.";
    } else {
      const correct = answer === scenario.kind;
      feedback.textContent = correct ? scenario.feedback : scenario.wrong;
      root.classList.add(correct ? "is-correct" : "is-incorrect");
    }
    drawMap();
    drawMoveout();
  }

  choices.forEach((button) => {
    button.addEventListener("click", () => {
      answer = button.dataset.earthquakeChoice;
      render();
    });
  });

  next.addEventListener("click", () => {
    activeScenario = (activeScenario + 1) % scenarios.length;
    answer = null;
    render();
  });

  window.addEventListener("resize", render);
  render();
})();

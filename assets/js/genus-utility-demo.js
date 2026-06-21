(function () {
  const root = document.querySelector("[data-genus-demo]");
  if (!root) return;

  const board = root.querySelector("[data-genus-board]");
  const canvas = root.querySelector("[data-genus-canvas]");
  const ctx = canvas.getContext("2d");
  const markerCanvas = document.createElement("canvas");
  markerCanvas.className = "genus-marker-canvas";
  markerCanvas.setAttribute("aria-hidden", "true");
  board.append(markerCanvas);
  const markerCtx = markerCanvas.getContext("2d");
  const validateButton = root.querySelector("[data-genus-validate]");
  const solutionButton = root.querySelector("[data-genus-solution]");
  const resetButton = root.querySelector("[data-genus-reset]");
  const status = root.querySelector("[data-genus-status]");
  const nodes = Array.from(root.querySelectorAll("[data-node]"));
  const utilities = ["water", "gas", "power"];
  const houses = ["house-a", "house-b", "house-c"];
  const required = utilities.flatMap((utility) =>
    houses.map((house) => edgeKey(utility, house)),
  );

  let dpr = 1;
  let active = null;
  let edges = [];
  let missingEdges = [];
  let crossings = [];
  let solutionVisible = false;

  function edgeKey(a, b) {
    const utility = utilities.includes(a) ? a : b;
    const house = houses.includes(a) ? a : b;
    return `${utility}:${house}`;
  }

  function nodeKind(id) {
    if (utilities.includes(id)) return "utility";
    if (houses.includes(id)) return "house";
    return "";
  }

  function resizeCanvas() {
    const rect = board.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    markerCanvas.width = canvas.width;
    markerCanvas.height = canvas.height;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    markerCanvas.style.width = canvas.style.width;
    markerCanvas.style.height = canvas.style.height;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    markerCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (solutionVisible) edges = solutionEdges();
    draw();
  }

  function boardSize() {
    const rect = board.getBoundingClientRect();
    return { width: rect.width, height: rect.height };
  }

  function mod(value, size) {
    return ((value % size) + size) % size;
  }

  function pointFromEvent(event) {
    const rect = board.getBoundingClientRect();
    const size = boardSize();
    const rawX = event.clientX - rect.left;
    const rawY = event.clientY - rect.top;
    return {
      breakBefore:
        rawX < 0 || rawX > size.width || rawY < 0 || rawY > size.height,
      outside: rawX < 0 || rawX > size.width || rawY < 0 || rawY > size.height,
      rawX,
      rawY,
      x: mod(rawX, size.width),
      y: mod(rawY, size.height),
    };
  }

  function nodeCenter(id) {
    const node = root.querySelector(`[data-node="${id}"]`);
    const boardRect = board.getBoundingClientRect();
    const nodeRect = node.getBoundingClientRect();
    return {
      outside: false,
      rawX: nodeRect.left + nodeRect.width / 2 - boardRect.left,
      rawY: nodeRect.top + nodeRect.height / 2 - boardRect.top,
      x: nodeRect.left + nodeRect.width / 2 - boardRect.left,
      y: nodeRect.top + nodeRect.height / 2 - boardRect.top,
    };
  }

  function nodeAtPoint(point) {
    for (const node of nodes) {
      const id = node.dataset.node;
      const center = nodeCenter(id);
      const rect = node.getBoundingClientRect();
      const radius = Math.max(rect.width, rect.height) * 0.62;
      if (Math.hypot(point.x - center.x, point.y - center.y) <= radius)
        return id;
    }
    return "";
  }

  function splitSegment(a, b) {
    if (b.gapBefore) return [];
    const size = boardSize();
    const start = withRaw(a);
    const end = withRaw(b);
    const crossed = firstBoundaryCrossing(start, end, size);
    if (crossed) {
      return [
        [
          { x: start.x, y: start.y },
          { x: crossed.exitX, y: crossed.exitY },
        ],
        [
          { x: crossed.entryX, y: crossed.entryY },
          { x: end.x, y: end.y },
        ],
      ];
    }
    return [[a, b]];
  }

  function withRaw(point) {
    return {
      ...point,
      rawX: point.rawX ?? point.x,
      rawY: point.rawY ?? point.y,
      outside: point.outside || false,
    };
  }

  function firstBoundaryCrossing(a, b, size) {
    const dx = b.rawX - a.rawX;
    const dy = b.rawY - a.rawY;
    const candidates = [];

    if (dx > 0 && a.rawX <= size.width && b.rawX > size.width) {
      candidates.push({
        axis: "x",
        side: "right",
        t: (size.width - a.rawX) / dx,
      });
    }
    if (dx < 0 && a.rawX >= 0 && b.rawX < 0) {
      candidates.push({ axis: "x", side: "left", t: -a.rawX / dx });
    }
    if (dy > 0 && a.rawY <= size.height && b.rawY > size.height) {
      candidates.push({
        axis: "y",
        side: "bottom",
        t: (size.height - a.rawY) / dy,
      });
    }
    if (dy < 0 && a.rawY >= 0 && b.rawY < 0) {
      candidates.push({ axis: "y", side: "top", t: -a.rawY / dy });
    }

    const crossing = candidates
      .filter((candidate) => candidate.t >= 0 && candidate.t <= 1)
      .sort((first, second) => first.t - second.t)[0];
    if (!crossing) return null;

    const boundaryX = a.rawX + dx * crossing.t;
    const boundaryY = a.rawY + dy * crossing.t;
    if (crossing.axis === "x") {
      return {
        exitX: crossing.side === "right" ? size.width : 0,
        exitY: mod(boundaryY, size.height),
        entryX: crossing.side === "right" ? 0 : size.width,
        entryY: mod(boundaryY, size.height),
      };
    }
    return {
      exitX: mod(boundaryX, size.width),
      exitY: crossing.side === "bottom" ? size.height : 0,
      entryX: mod(boundaryX, size.width),
      entryY: crossing.side === "bottom" ? 0 : size.height,
    };
  }

  function visibleSegments(points) {
    const segments = [];
    for (let index = 1; index < points.length; index += 1) {
      segments.push(...splitSegment(points[index - 1], points[index]));
    }
    return segments;
  }

  function visibleChunks(points) {
    const chunks = [];
    let chunk = [points[0]];
    for (let index = 1; index < points.length; index += 1) {
      const previous = points[index - 1];
      const current = points[index];
      if (current.gapBefore) {
        if (chunk.length > 1) chunks.push(chunk);
        chunk = [current];
        continue;
      }

      const crossed = firstBoundaryCrossing(
        withRaw(previous),
        withRaw(current),
        boardSize(),
      );
      if (crossed) {
        chunk.push({ x: crossed.exitX, y: crossed.exitY });
        if (chunk.length > 1) chunks.push(chunk);
        chunk = [{ x: crossed.entryX, y: crossed.entryY }, current];
      } else {
        chunk.push(current);
      }
    }
    if (chunk.length > 1) chunks.push(chunk);
    return chunks;
  }

  function drawPath(points, options) {
    const chunks = visibleChunks(points);
    ctx.save();
    ctx.strokeStyle = options.color;
    ctx.lineWidth = options.width;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (options.dash) ctx.setLineDash(options.dash);
    for (const chunk of chunks) {
      ctx.beginPath();
      ctx.moveTo(chunk[0].x, chunk[0].y);
      if (chunk.length === 2 || !options.smooth) {
        for (const point of chunk.slice(1)) ctx.lineTo(point.x, point.y);
      } else {
        for (let index = 1; index < chunk.length - 1; index += 1) {
          const current = chunk[index];
          const next = chunk[index + 1];
          const midpoint = {
            x: (current.x + next.x) / 2,
            y: (current.y + next.y) / 2,
          };
          ctx.quadraticCurveTo(current.x, current.y, midpoint.x, midpoint.y);
        }
        const last = chunk[chunk.length - 1];
        ctx.lineTo(last.x, last.y);
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawMissing(edge) {
    const [utility, house] = edge.split(":");
    drawPath([nodeCenter(utility), nodeCenter(house)], {
      color: "#d13d3d",
      dash: [8, 7],
      width: 2.5,
    });
  }

  function draw() {
    const size = boardSize();
    ctx.clearRect(0, 0, size.width, size.height);
    markerCtx.clearRect(0, 0, size.width, size.height);

    for (const edge of edges) {
      drawPath(edge.points, {
        color: edge.valid ? "#12847e" : "#9aa8ad",
        smooth: edge.solution && edge.smooth !== false,
        width: edge.valid ? 4 : 3,
      });
    }

    if (active) {
      drawPath(active.points, {
        color: "#ff8a65",
        width: 4,
      });
    }

    for (const edge of missingEdges) drawMissing(edge);

    markerCtx.save();
    markerCtx.strokeStyle = "#d13d3d";
    markerCtx.fillStyle = "rgba(209, 61, 61, 0.14)";
    markerCtx.lineWidth = 3;
    for (const point of crossings) {
      markerCtx.beginPath();
      markerCtx.arc(point.x, point.y, 13, 0, Math.PI * 2);
      markerCtx.fill();
      markerCtx.stroke();
    }
    markerCtx.restore();
  }

  function setActiveNode(id, on) {
    const node = root.querySelector(`[data-node="${id}"]`);
    if (node) node.classList.toggle("active", on);
  }

  function startDrawing(event) {
    const point = pointFromEvent(event);
    const start = nodeAtPoint(point);
    if (!start) return;

    event.preventDefault();
    board.setPointerCapture(event.pointerId);
    missingEdges = [];
    crossings = [];
    solutionVisible = false;
    active = {
      pointerId: event.pointerId,
      start,
      points: [nodeCenter(start), point],
    };
    setActiveNode(start, true);
    status.textContent = "Release on the opposite row to keep the connection.";
    draw();
  }

  function continueDrawing(event) {
    if (!active || active.pointerId !== event.pointerId) return;
    event.preventDefault();
    const point = pointFromEvent(event);
    const last = active.points[active.points.length - 1];
    if (Math.hypot(point.x - last.x, point.y - last.y) > 1.5) {
      active.points.push(point);
      draw();
    }
  }

  function finishDrawing(event) {
    if (!active || active.pointerId !== event.pointerId) return;
    event.preventDefault();
    const endPoint = pointFromEvent(event);
    const end = nodeAtPoint(endPoint);
    const start = active.start;
    setActiveNode(start, false);

    if (end && nodeKind(start) !== nodeKind(end)) {
      const key = edgeKey(start, end);
      edges = edges.filter((edge) => edge.key !== key);
      active.points.push(nodeCenter(end));
      edges.push({
        key,
        valid: true,
        solution: false,
        points: simplifyPoints(active.points),
      });
      status.textContent = `${edges.length} of 9 connections drawn.`;
    } else if (end && start !== end) {
      status.textContent =
        "Utilities connect to houses; same-row lines do not count.";
    } else {
      status.textContent = "Release on another node to keep a connection.";
    }

    active = null;
    draw();
  }

  function simplifyPoints(points) {
    const simplified = [points[0]];
    for (let index = 1; index < points.length - 1; index += 1) {
      const previous = simplified[simplified.length - 1];
      const current = points[index];
      if (Math.hypot(current.x - previous.x, current.y - previous.y) > 4) {
        simplified.push(current);
      }
    }
    simplified.push(points[points.length - 1]);
    return simplified;
  }

  function segmentIntersection(a, b, c, d) {
    const denominator = (d.y - c.y) * (b.x - a.x) - (d.x - c.x) * (b.y - a.y);
    if (Math.abs(denominator) < 0.0001) return null;

    const ua =
      ((d.x - c.x) * (a.y - c.y) - (d.y - c.y) * (a.x - c.x)) / denominator;
    const ub =
      ((b.x - a.x) * (a.y - c.y) - (b.y - a.y) * (a.x - c.x)) / denominator;
    if (ua < -0.001 || ua > 1.001 || ub < -0.001 || ub > 1.001) return null;

    return {
      x: a.x + ua * (b.x - a.x),
      y: a.y + ua * (b.y - a.y),
    };
  }

  function findCrossings() {
    const found = [];
    for (let first = 0; first < edges.length; first += 1) {
      for (let second = first + 1; second < edges.length; second += 1) {
        if (edges[first].solution && edges[second].solution) continue;
        const sharedEndpoints = sharedEndpointIds(
          edges[first].key,
          edges[second].key,
        );
        const firstSegments = visibleSegments(edges[first].points);
        const secondSegments = visibleSegments(edges[second].points);
        for (const [a, b] of firstSegments) {
          for (const [c, d] of secondSegments) {
            const crossing = segmentIntersection(a, b, c, d);
            if (
              crossing &&
              !nearSharedEndpoint(crossing, sharedEndpoints) &&
              !nearExisting(found, crossing)
            ) {
              found.push(crossing);
            }
          }
        }
      }
    }
    return found;
  }

  function sharedEndpointIds(first, second) {
    const firstParts = first.split(":");
    const secondParts = second.split(":");
    return firstParts.filter((part) => secondParts.includes(part));
  }

  function nearSharedEndpoint(point, endpointIds) {
    return endpointIds.some(
      (id) =>
        Math.hypot(point.x - nodeCenter(id).x, point.y - nodeCenter(id).y) < 22,
    );
  }

  function nearExisting(points, point) {
    return points.some(
      (existing) => Math.hypot(existing.x - point.x, existing.y - point.y) < 12,
    );
  }

  function validate() {
    const drawn = new Set(edges.map((edge) => edge.key));
    missingEdges = required.filter((edge) => !drawn.has(edge));
    crossings = findCrossings();

    if (!missingEdges.length && !crossings.length) {
      status.textContent =
        "Solved: all nine connections and no visible crossings.";
    } else if (missingEdges.length && crossings.length) {
      status.textContent = `${missingEdges.length} missing connection${missingEdges.length === 1 ? "" : "s"} and ${crossings.length} crossing${crossings.length === 1 ? "" : "s"} found.`;
    } else if (missingEdges.length) {
      status.textContent = `${missingEdges.length} missing connection${missingEdges.length === 1 ? "" : "s"} shown in red.`;
    } else {
      status.textContent = `${crossings.length} crossing${crossings.length === 1 ? "" : "s"} circled in red.`;
    }
    draw();
  }

  function solutionEdges() {
    return required.map((key) => {
      const [utility, house] = key.split(":");
      const points = solutionPath(utility, house);
      return {
        key,
        valid: true,
        solution: true,
        smooth: key !== "water:house-b" && key !== "gas:house-c",
        wraps: points.some((point) => point.breakBefore),
        points,
      };
    });
  }

  function solutionPath(utility, house) {
    const start = nodeCenter(utility);
    const end = nodeCenter(house);
    const size = boardSize();
    const point = (x, y) => ({
      outside: false,
      rawX: x * size.width,
      rawY: y * size.height,
      x: x * size.width,
      y: y * size.height,
    });
    const absolutePoint = (x, y) => ({
      outside: false,
      rawX: x,
      rawY: y,
      x,
      y,
    });
    const gapPoint = (x, y) => ({ ...point(x, y), gapBefore: true });
    const absoluteGapPoint = (x, y) => ({
      ...absolutePoint(x, y),
      gapBefore: true,
    });
    const key = edgeKey(utility, house);
    const powerCenter = nodeCenter("power");
    const topLaneY = Math.max(
      10,
      Math.min(size.height * 0.16, powerCenter.y - 42),
    );
    const powerLeftX = Math.max(
      size.width * 0.58,
      powerCenter.x - Math.max(52, size.width * 0.08),
    );

    const paths = {
      "water:house-a": [start, end],
      "gas:house-b": [start, end],
      "power:house-c": [start, end],

      "water:house-b": [start, end],

      "gas:house-c": [start, end],

      "power:house-b": [
        start,
        point(0.86, 0.32),
        point(0.93, 0.52),
        point(0.87, 0.88),
        point(0.66, 0.91),
        end,
      ],

      "water:house-c": [
        start,
        absolutePoint(Math.max(0, start.x - size.width * 0.08), topLaneY),
        absolutePoint(0, topLaneY),
        absoluteGapPoint(size.width, topLaneY),
        absolutePoint(
          Math.min(size.width, powerCenter.x + size.width * 0.2),
          topLaneY,
        ),
        absolutePoint(powerLeftX, topLaneY + size.height * 0.02),
        absolutePoint(powerLeftX - size.width * 0.02, size.height * 0.36),
        absolutePoint(
          powerCenter.x - Math.max(18, size.width * 0.035),
          size.height * 0.61,
        ),
        end,
      ],

      "gas:house-a": [
        start,
        point(0.47, 0.15),
        point(0.39, 0.0),
        gapPoint(0.39, 1.0),
        point(0.33, 0.94),
        point(0.24, 0.9),
        end,
      ],

      "power:house-a": [
        start,
        point(0.93, 0.35),
        point(0.98, 0.68),
        point(0.84, 0.94),
        point(0.56, 0.96),
        point(0.34, 0.88),
        end,
      ],
    };

    return paths[key] || [start, end];
  }

  function showSolution() {
    active = null;
    missingEdges = [];
    crossings = [];
    solutionVisible = true;
    nodes.forEach((node) => node.classList.remove("active"));
    edges = solutionEdges();
    status.textContent = "Solution shown: some lines wrap around the edge.";
    draw();
  }

  function reset() {
    edges = [];
    missingEdges = [];
    crossings = [];
    active = null;
    solutionVisible = false;
    nodes.forEach((node) => node.classList.remove("active"));
    status.textContent = "Draw nine utility-to-house lines.";
    draw();
  }

  board.addEventListener("pointerdown", startDrawing);
  board.addEventListener("pointermove", continueDrawing);
  board.addEventListener("pointerup", finishDrawing);
  board.addEventListener("pointercancel", finishDrawing);
  validateButton.addEventListener("click", validate);
  solutionButton.addEventListener("click", showSolution);
  resetButton.addEventListener("click", reset);
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();
})();

(function () {
  const root = document.querySelector("[data-embedding-demo]");
  if (!root) return;

  const API_SOCKET = "wss://genus.fly.dev/stream_calc_genus";
  const canvas = root.querySelector("[data-embedding-canvas]");
  const ctx = canvas.getContext("2d");
  const runButton = root.querySelector("[data-embedding-run]");
  const moveButton = root.querySelector("[data-embedding-move]");
  const sampleButton = root.querySelector("[data-embedding-sample]");
  const clearButton = root.querySelector("[data-embedding-clear]");
  const status = root.querySelector("[data-embedding-status]");
  const output = root.querySelector("[data-embedding-output]");
  const genusTarget = root.querySelector("[data-embedding-genus]");
  const rotationTarget = root.querySelector("[data-embedding-rotation]");
  const imageOverlay = document.createElement("div");
  imageOverlay.className = "embedding-image-overlay";
  imageOverlay.hidden = true;
  canvas.parentElement.append(imageOverlay);

  const sampleAdjacency = [
    [15, 16, 17],
    [18, 19, 20],
    [21, 22, 23],
    [15, 18, 21],
    [15, 24, 25],
    [18, 26, 27],
    [21, 28, 29],
    [16, 19, 22],
    [16, 26, 28],
    [19, 24, 29],
    [22, 25, 27],
    [17, 20, 23],
    [17, 27, 29],
    [20, 25, 28],
    [23, 24, 26],
    [0, 3, 4],
    [0, 7, 8],
    [0, 11, 12],
    [1, 3, 5],
    [1, 7, 9],
    [1, 11, 13],
    [2, 3, 6],
    [2, 7, 10],
    [2, 11, 14],
    [4, 9, 14],
    [4, 10, 13],
    [5, 8, 14],
    [5, 10, 12],
    [6, 8, 13],
    [6, 9, 12],
  ];

  let dpr = 1;
  let nodes = [];
  let edges = [];
  let nextId = 0;
  let moveMode = false;
  let interaction = null;
  let embedded = false;
  const secondaryActionButtons = [runButton, moveButton, sampleButton];

  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(Math.max(rect.height, 420) * dpr);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${Math.max(rect.height, 420)}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function canvasPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  }

  function nodeAt(point) {
    for (let index = nodes.length - 1; index >= 0; index -= 1) {
      const node = nodes[index];
      if (Math.hypot(point.x - node.x, point.y - node.y) <= 16) return node;
    }
    return null;
  }

  function draw() {
    const width = canvas.width / dpr;
    const height = canvas.height / dpr;
    ctx.clearRect(0, 0, width, height);

    ctx.save();
    ctx.strokeStyle = "rgba(35, 49, 61, 0.12)";
    ctx.lineWidth = 1;
    for (let x = 32; x < width; x += 32) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 32; y < height; y += 32) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = "#12847e";
    ctx.lineCap = "round";
    ctx.lineWidth = 3;
    for (const edge of edges) {
      const a = nodes.find((node) => node.id === edge.a);
      const b = nodes.find((node) => node.id === edge.b);
      if (!a || !b) continue;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    if (interaction?.type === "edge") {
      ctx.strokeStyle = "#ff8a65";
      ctx.beginPath();
      ctx.moveTo(interaction.from.x, interaction.from.y);
      ctx.lineTo(interaction.to.x, interaction.to.y);
      ctx.stroke();
    }
    ctx.restore();

    for (const node of nodes) {
      ctx.save();
      ctx.fillStyle = "#fffdf8";
      ctx.strokeStyle = "#23313d";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(node.x, node.y, 16, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#12847e";
      ctx.font = "900 12px Nunito, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(node.id), node.x, node.y);
      ctx.restore();
    }
  }

  function addNode(point) {
    nodes.push({ id: nextId, x: point.x, y: point.y });
    nextId += 1;
    setStatus(`Added vertex ${nextId - 1}.`);
    draw();
  }

  function addEdge(a, b) {
    if (a.id === b.id) return;
    const low = Math.min(a.id, b.id);
    const high = Math.max(a.id, b.id);
    if (edges.some((edge) => edge.a === low && edge.b === high)) return;
    edges.push({ a: low, b: high });
    setStatus(`Added edge ${low}-${high}.`);
  }

  function removeNode(node) {
    nodes = nodes.filter((candidate) => candidate.id !== node.id);
    edges = edges.filter((edge) => edge.a !== node.id && edge.b !== node.id);
    interaction = null;
    setStatus(`Deleted vertex ${node.id}.`);
    draw();
  }

  function clearEmbeddingImage() {
    embedded = false;
    imageOverlay.hidden = true;
    imageOverlay.innerHTML = "";
  }

  function setResultActionsHidden(hidden) {
    for (const button of secondaryActionButtons) button.hidden = hidden;
  }

  function setStatus(message, busy) {
    status.textContent = message;
    runButton.disabled = !!busy;
    sampleButton.disabled = !!busy;
    clearButton.disabled = !!busy;
    moveButton.disabled = !!busy;
  }

  function onPointerDown(event) {
    if (embedded) return;
    const point = canvasPoint(event);
    const hit = nodeAt(point);
    canvas.setPointerCapture(event.pointerId);
    if (moveMode) {
      if (hit)
        interaction = {
          type: "move",
          node: hit,
          offsetX: point.x - hit.x,
          offsetY: point.y - hit.y,
        };
      return;
    }
    if (hit) {
      interaction = { type: "edge", from: hit, to: point };
    } else {
      addNode(point);
    }
    draw();
  }

  function onPointerMove(event) {
    if (embedded) return;
    if (!interaction) return;
    const point = canvasPoint(event);
    if (interaction.type === "move") {
      interaction.node.x = point.x - interaction.offsetX;
      interaction.node.y = point.y - interaction.offsetY;
    } else if (interaction.type === "edge") {
      interaction.to = point;
    }
    draw();
  }

  function onPointerUp(event) {
    if (embedded) return;
    if (!interaction) return;
    const point = canvasPoint(event);
    if (interaction.type === "edge") {
      const hit = nodeAt(point);
      if (hit) addEdge(interaction.from, hit);
    }
    interaction = null;
    draw();
  }

  function adjacencyList() {
    const ids = nodes.map((node) => node.id).sort((a, b) => a - b);
    const index = new Map(ids.map((id, i) => [id, i]));
    const adj = ids.map(() => []);
    for (const edge of edges) {
      const a = index.get(edge.a);
      const b = index.get(edge.b);
      adj[a].push(b);
      adj[b].push(a);
    }
    return adj.map((neighbors) => neighbors.sort((a, b) => a - b));
  }

  function connected(adj) {
    if (!adj.length) return false;
    const seen = new Set([0]);
    const stack = [0];
    while (stack.length) {
      const node = stack.pop();
      for (const neighbor of adj[node]) {
        if (!seen.has(neighbor)) {
          seen.add(neighbor);
          stack.push(neighbor);
        }
      }
    }
    return seen.size === adj.length;
  }

  function formatRotation(rotation) {
    return rotation
      .map((neighbors, index) => `${index}: (${neighbors.join(", ")})`)
      .join("\n");
  }

  function runApi(adj, alg, outputFormat) {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(API_SOCKET);
      const result = {
        stderr: [],
        stdout: [],
        runtime: "",
        json: null,
        image: "",
      };
      const timeout = window.setTimeout(() => {
        socket.close();
        reject(
          new Error(
            "The genus service is still working. Try a smaller graph or wait and try again.",
          ),
        );
      }, 120000);

      socket.onopen = () =>
        socket.send(JSON.stringify({ adj, alg, outputFormat }));
      socket.onerror = () => {
        window.clearTimeout(timeout);
        reject(new Error("Could not reach the genus service."));
      };
      socket.onmessage = (event) => {
        const text = String(event.data);
        const separator = text.indexOf(":");
        const type = separator === -1 ? text : text.slice(0, separator);
        const data = separator === -1 ? "" : text.slice(separator + 1);
        if (type === "STDERR") {
          result.stderr.push(data);
          const compact = data.trim().replace(/\s+/g, " ");
          if (compact) setStatus(compact, true);
        } else if (type === "STDOUT") {
          result.stdout.push(data);
        } else if (type === "TIME") {
          result.runtime = data;
        } else if (type === "JSON") {
          try {
            result.json = JSON.parse(data);
          } catch {
            result.json = { error: data };
          }
        } else if (type === "IMAGE") {
          result.image = data;
        }
      };
      socket.onclose = () => {
        window.clearTimeout(timeout);
        resolve(result);
      };
    });
  }

  async function embed() {
    const adj = adjacencyList();
    if (nodes.length < 2 || edges.length < 1) {
      setStatus("Add at least two vertices and one edge first.");
      return;
    }
    if (!connected(adj)) {
      setStatus(
        "This demo currently embeds one connected component at a time. Connect the graph first.",
      );
      return;
    }

    output.hidden = true;
    clearEmbeddingImage();
    rotationTarget.textContent = "";
    genusTarget.textContent = "";
    setResultActionsHidden(true);
    setStatus("Running PAGE on genus.fly.dev...", true);

    try {
      let response = await runApi(adj, "page", "json");
      let data = response.json;
      if (!data || data.error || !data.rotation_system) {
        setStatus(
          "PAGE did not return a rotation system; trying MULTI_GENUS...",
          true,
        );
        response = await runApi(adj, "multi_genus", "json");
        data = response.json;
      }
      if (!data || data.error || !data.rotation_system) {
        throw new Error(data?.error || "No rotation system returned.");
      }

      setStatus("Rendering the polygon embedding...", true);
      const drawing = await runApi(data.rotation_system, "none", "drawing");
      if (!drawing.image) {
        throw new Error(
          drawing.json?.error ||
            "The drawing renderer did not return an image.",
        );
      }

      const img = document.createElement("img");
      img.src = drawing.image;
      img.alt =
        "Crossing-free graph embedding drawn on a polygon with glued sides";
      imageOverlay.replaceChildren(img);
      imageOverlay.hidden = false;
      embedded = true;
      genusTarget.textContent = String(data.genus);
      rotationTarget.textContent = formatRotation(data.rotation_system);
      output.hidden = false;
      const statusMessage =
        Number(data.genus) === 0
          ? `Embedded ${nodes.length} vertices and ${edges.length} edges on a plane. This graph is 'planar'.`
          : `Embedded ${nodes.length} vertices and ${edges.length} edges. To form the 3D surface, glue matching colors.`;
      setStatus(
        statusMessage,
      );
    } catch (error) {
      setResultActionsHidden(false);
      setStatus(error.message || "Embedding failed.");
    } finally {
      runButton.disabled = false;
      sampleButton.disabled = false;
      clearButton.disabled = false;
      moveButton.disabled = false;
    }
  }

  function loadSample() {
    moveMode = false;
    moveButton.setAttribute("aria-pressed", "false");
    const rect = canvas.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const radius = Math.min(rect.width, rect.height) * 0.42;
    nodes = sampleAdjacency.map((_, index) => {
      const angle =
        -Math.PI / 2 + (index / sampleAdjacency.length) * Math.PI * 2;
      return {
        id: index,
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
      };
    });
    nextId = nodes.length;
    edges = [];
    sampleAdjacency.forEach((neighbors, a) => {
      neighbors.forEach((b) => {
        if (a < b) edges.push({ a, b });
      });
    });
    output.hidden = true;
    clearEmbeddingImage();
    setResultActionsHidden(false);
    setStatus("Loaded the 3,8-cage test graph. It may take a bit to embed.");
    draw();
  }

  function clear() {
    moveMode = false;
    moveButton.setAttribute("aria-pressed", "false");
    nodes = [];
    edges = [];
    nextId = 0;
    interaction = null;
    output.hidden = true;
    clearEmbeddingImage();
    setResultActionsHidden(false);
    setStatus("Cleared. Click to place vertices.");
    draw();
  }

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  canvas.addEventListener("dblclick", (event) => {
    if (embedded) return;
    const hit = nodeAt(canvasPoint(event));
    if (hit) removeNode(hit);
  });
  runButton.addEventListener("click", embed);
  sampleButton.addEventListener("click", loadSample);
  clearButton.addEventListener("click", clear);
  moveButton.addEventListener("click", () => {
    moveMode = !moveMode;
    moveButton.setAttribute("aria-pressed", String(moveMode));
    setStatus(
      moveMode
        ? "Move mode is on. Drag vertices to reposition them."
        : "Move mode is off. Drag between vertices to add edges.",
    );
  });
  window.addEventListener("resize", resize);
  resize();
})();

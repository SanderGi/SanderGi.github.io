(function () {
  const root = document.querySelector("[data-living-demo]");
  if (!root) return;

  const products = [
    {
      id: "slay-speech",
      name: "Slay Speech",
      seller: "Koel Labs",
      price: "$5/mo",
      badge: "Lowest impact",
      url: "https://slayspeech.com",
      category: "Software",
      initials: "SS",
      image: "/images/demo/slay-speech.webp",
      co2e: 0,
      water: "0 L",
      repair: "Digital",
      risk: "None",
      score: 100,
      note: "A pronunciation and language-learning app with no shipping, no rare materials, no product disposal, and a seller you can actually email.",
      stages: [0, 0, 0, 0],
      sources: [
        { label: "Seller page", url: "https://slayspeech.com" },
        "No physical freight",
        "No product safety flags",
      ],
    },
    {
      id: "elder-wand",
      name: "The Elder Wand",
      seller: "Dumbledore Estate Sales",
      price: "$999",
      badge: "Durable",
      category: "Artifacts",
      initials: "EW",
      image: "/images/demo/living-products/elder-wand.png",
      icon: "wand",
      color: "#6d4c41",
      co2e: 11,
      water: "18 L",
      repair: "Excellent",
      risk: "Ethically complicated",
      score: 82,
      note: "Long service life helps, but provenance is doing a lot of work here.",
      stages: [8, 9, 12, 71],
      sources: [
        "Antique wood proxy",
        "No battery",
        "Lifetime durability claim",
      ],
    },
    {
      id: "excalibur",
      name: "Excalibur",
      seller: "Lake Logistics",
      price: "One destiny",
      badge: "Reusable",
      category: "Artifacts",
      initials: "EX",
      image: "/images/demo/living-products/excalibur.png",
      icon: "sword",
      color: "#5c6f82",
      co2e: 86,
      water: "44 L",
      repair: "Legendary",
      risk: "Royal succession externalities",
      score: 74,
      note: "Forged metal is impact-heavy, but centuries of reuse make the per-adventure footprint surprisingly reasonable.",
      stages: [52, 6, 4, 38],
      sources: [
        "Steel forging proxy",
        "Artifact lifespan assumption",
        "No retail packaging",
      ],
    },
    {
      id: "pet-elephant",
      name: "A Pet Elephant",
      seller: "Questionable Menagerie",
      price: "$48,000",
      badge: "Please reconsider",
      category: "Living Goods",
      initials: "PE",
      image: "/images/demo/living-products/pet-elephant.png",
      icon: "elephant",
      color: "#8b9aa3",
      co2e: 18000,
      water: "Huge",
      repair: "Not applicable",
      risk: "Very high",
      score: 4,
      note: "The interface is gently, firmly asking you not to buy a charismatic megafauna roommate.",
      stages: [2, 0, 93, 5],
      sources: [
        "Feed and care estimate",
        "Animal welfare red flag",
        "Freight excluded",
      ],
    },
    {
      id: "flown-turkey",
      name: "A Flown Away Turkey",
      seller: "WKRP Fulfillment",
      price: "$0 if you can catch it",
      badge: "Unavailable",
      category: "Food",
      initials: "FT",
      image: "/images/demo/living-products/flown-turkey.png",
      icon: "turkey",
      color: "#b76e45",
      co2e: 32,
      water: "520 L",
      repair: "Escaped",
      risk: "Supply chain airborne",
      score: 38,
      note: "Main recommendation: switch to a local plant-based main dish, and do not use helicopters for grocery logistics.",
      stages: [18, 44, 30, 8],
      sources: ["Poultry proxy", "Aviation warning", "Availability: negative"],
    },
    {
      id: "one-ring",
      name: "The One Ring",
      seller: "Mordor Direct",
      price: "$1,999",
      badge: "High risk",
      category: "Jewelry",
      initials: "OR",
      image: "/images/demo/living-products/one-ring.png",
      icon: "ring",
      color: "#c8911c",
      co2e: 420,
      water: "1,900 L",
      repair: "Destroy only",
      risk: "Extremely high",
      score: 9,
      note: "Tiny mass, enormous governance risk. The model flags the sourcing story before the cart can pretend this is normal.",
      stages: [62, 5, 12, 21],
      sources: [
        "Precious metal proxy",
        "Volcanic disposal note",
        "Supplier transparency: none",
      ],
    },
    {
      id: "infinity-gauntlet",
      name: "The Infinity Gauntlet",
      seller: "Titan Estate Liquidators",
      price: "$3,000,000",
      badge: "Material intensive",
      category: "Wearables",
      initials: "IG",
      image: "/images/demo/living-products/infinity-gauntlet.png",
      icon: "gauntlet",
      color: "#c9a227",
      co2e: 9800,
      water: "High",
      repair: "Catastrophic warranty",
      risk: "Universal",
      score: 2,
      note: "Six stones, one glove, infinite externalities. Sustainability score: snapped.",
      stages: [88, 3, 2, 7],
      sources: [
        "Gem mining proxy",
        "Gold alloy proxy",
        "Safety assessment failed",
      ],
    },
    {
      id: "triforce",
      name: "The Triforce",
      seller: "Hyrule Relics",
      price: "3 easy virtues",
      badge: "Unknown",
      category: "Artifacts",
      initials: "TF",
      image: "/images/demo/living-products/triforce.png",
      icon: "triforce",
      color: "#e0a921",
      co2e: 120,
      water: "Unknown",
      repair: "Quest based",
      risk: "Wisdom required",
      score: 61,
      note: "The data source is mostly prophecy. Living Sustainability would surface that uncertainty instead of hiding it.",
      stages: [34, 4, 2, 60],
      sources: [
        "Gold artifact proxy",
        "Unverified magic factor",
        "Uncertain functional unit",
      ],
    },
    {
      id: "mjolnir",
      name: "Mjölnir",
      seller: "Asgard Hardware",
      price: "$799",
      badge: "Heavy freight",
      category: "Tools",
      initials: "MJ",
      image: "/images/demo/living-products/mjolnir.png",
      icon: "hammer",
      color: "#7a8790",
      co2e: 510,
      water: "230 L",
      repair: "Worthiness locked",
      risk: "Delivery exception",
      score: 52,
      note: "Durable, repairable, and nearly impossible to ship. Pickup recommended if worthy.",
      stages: [55, 29, 1, 15],
      sources: [
        "Tool steel proxy",
        "Weight-sensitive freight",
        "Repairability inferred",
      ],
    },
    {
      id: "lightsaber",
      name: "A Lightsaber",
      seller: "Outer Rim Outfitters",
      price: "$1,299",
      badge: "Energy efficient",
      category: "Tools",
      initials: "LS",
      image: "/images/demo/living-products/lightsaber.png",
      icon: "lightsaber",
      color: "#34b6ff",
      co2e: 175,
      water: "86 L",
      repair: "Modular",
      risk: "Kyber sourcing",
      score: 66,
      note: "Compact and repairable, but crystal sourcing needs much better documentation.",
      stages: [47, 8, 18, 27],
      sources: [
        "Electronics proxy",
        "Rare crystal flag",
        "Modular repair note",
      ],
    },
    {
      id: "flux-capacitor",
      name: "A Flux Capacitor",
      seller: "Hill Valley Auto Parts",
      price: "$88.88",
      badge: "Use-phase heavy",
      category: "Electronics",
      initials: "FC",
      image: "/images/demo/living-products/flux-capacitor.png",
      icon: "flux",
      color: "#33a6a6",
      co2e: 1210,
      water: "140 L",
      repair: "Doc only",
      risk: "Temporal leakage",
      score: 29,
      note: "The build is modest; the 1.21 gigawatt use phase is where the panel starts sweating.",
      stages: [12, 3, 82, 3],
      sources: [
        "Power draw estimate",
        "Automotive electronics proxy",
        "Temporal safety unknown",
      ],
    },
    {
      id: "proton-pack",
      name: "A Proton Pack",
      seller: "Firehouse Refurbished",
      price: "$4,500",
      badge: "Refurbished",
      category: "Appliances",
      initials: "PP",
      image: "/images/demo/living-products/proton-pack.png",
      icon: "proton",
      color: "#d24f3f",
      co2e: 740,
      water: "310 L",
      repair: "Good",
      risk: "Do not cross streams",
      score: 57,
      note: "Refurbishment helps, but the energy system and containment risks still dominate the impact story.",
      stages: [40, 10, 42, 8],
      sources: [
        "Refurbished equipment proxy",
        "Battery pack proxy",
        "Use safety note",
      ],
    },
    {
      id: "kryptonite-necklace",
      name: "A Kryptonite Necklace",
      seller: "Lex Luxury Goods",
      price: "$299",
      badge: "Toxicity flag",
      category: "Jewelry",
      initials: "KN",
      image: "/images/demo/living-products/kryptonite-necklace.png",
      icon: "kryptonite",
      color: "#42b853",
      co2e: 155,
      water: "77 L",
      repair: "Poor",
      risk: "High toxicity",
      score: 21,
      note: "Small object, big toxicity warning. The panel treats human and ecological hazard as part of the buying context.",
      stages: [41, 8, 5, 46],
      sources: [
        "Gemstone jewelry proxy",
        "Hazard flag",
        "Low transparency seller",
      ],
    },
    {
      id: "energon-runoff",
      name: "Energon Runoff",
      seller: "Cybertron Surplus",
      price: "$19/gal",
      badge: "Do not spill",
      category: "Fuel",
      initials: "ER",
      image: "/images/demo/living-products/energon-runoff.png",
      icon: "energon",
      color: "#9c4dcc",
      co2e: 6500,
      water: "Contaminates it",
      repair: "Containment only",
      risk: "Very high",
      score: 3,
      note: "A product whose name includes runoff is doing the warning label's job for us.",
      stages: [18, 6, 31, 45],
      sources: [
        "Synthetic fuel proxy",
        "Aquatic toxicity warning",
        "Spill risk",
      ],
    },
    {
      id: "hexxus-leveller",
      name: "Hexxus / Leveller",
      seller: "SmogCo Industrial",
      price: "$666",
      badge: "Reject",
      category: "Industrial",
      initials: "HL",
      image: "/images/demo/living-products/hexxus-leveller.png",
      icon: "smog",
      color: "#4d5b3f",
      co2e: 9999,
      water: "Severe",
      repair: "Please dismantle",
      risk: "Maximum",
      score: 1,
      note: "This is less a product than a cautionary tale with a checkout button.",
      stages: [25, 12, 18, 45],
      sources: [
        "Industrial combustion proxy",
        "Deforestation risk",
        "Villainy disclosure",
      ],
    },
  ];

  const productGrid = root.querySelector("[data-living-products]");
  const score = root.querySelector("[data-living-score]");
  const scoreBar = root.querySelector("[data-living-score-bar]");
  const co2e = root.querySelector("[data-living-co2e]");
  const water = root.querySelector("[data-living-water]");
  const repair = root.querySelector("[data-living-repair]");
  const risk = root.querySelector("[data-living-risk]");
  const note = root.querySelector("[data-living-note]");
  const stages = root.querySelector("[data-living-stages]");
  const sources = root.querySelector("[data-living-sources]");
  const equivalency = root.querySelector("[data-living-equivalency]");

  let selectedId = "slay-speech";

  function productImage(product) {
    if (product.image) return product.image;
    const color = product.color || "#52b788";
    const icon = product.icon || "box";
    const svg = `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 160">
        <defs>
          <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
            <stop stop-color="#e6f2ed" offset="0"/>
            <stop stop-color="#fff3df" offset="1"/>
          </linearGradient>
          <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="6" stdDeviation="4" flood-color="#23313d" flood-opacity=".22"/>
          </filter>
        </defs>
        <rect width="220" height="160" rx="18" fill="url(#bg)"/>
        <g fill="none" stroke="${color}" stroke-linecap="round" stroke-linejoin="round" stroke-width="9" filter="url(#shadow)">
          ${iconPath(icon)}
        </g>
        <text x="18" y="138" fill="#033425" font-family="system-ui, sans-serif" font-size="18" font-weight="900">${product.initials}</text>
      </svg>
    `;
    return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
  }

  function iconPath(icon) {
    const paths = {
      wand: '<path d="M62 115 153 24"/><path d="M139 26h28M153 12v28M80 42l12 12M48 74l12 12"/>',
      sword:
        '<path d="M111 20v86"/><path d="M86 101h50"/><path d="m111 20 18 24-18 17-18-17 18-24Z"/><path d="M101 106h20v29h-20z"/>',
      elephant:
        '<path d="M58 93c0-30 23-48 55-48 28 0 49 17 49 44v31H72V96"/><path d="M71 62c-22 2-34 17-29 36 4 14 16 23 30 24"/><path d="M160 88c18 0 26 10 24 23-2 11-11 17-22 18"/><path d="M129 46c6 12 7 26 1 42"/><path d="M75 121v20M151 121v20"/>',
      turkey:
        '<path d="M107 91c0-19 14-33 33-33s33 14 33 33-14 34-33 34-33-15-33-34Z"/><path d="M111 78 55 50M111 94 42 92M113 108 60 130"/><path d="M148 61c0-16 13-27 28-24"/><path d="M132 121v18M150 121v18"/>',
      ring: '<circle cx="110" cy="82" r="44"/><circle cx="110" cy="82" r="25"/><path d="M78 36c23-14 44-14 65 0"/>',
      gauntlet:
        '<path d="M73 75V38c0-8 12-8 12 0v33"/><path d="M93 70V28c0-8 13-8 13 0v42"/><path d="M115 70V31c0-8 13-8 13 0v42"/><path d="M136 74V44c0-8 12-8 12 0v51c0 32-18 48-42 48-23 0-38-14-38-39V75"/><circle cx="89" cy="98" r="5"/><circle cx="111" cy="96" r="5"/><circle cx="132" cy="99" r="5"/>',
      triforce:
        '<path d="m110 26 29 50H81l29-50Z"/><path d="m81 76 29 50H52l29-50Z"/><path d="m139 76 29 50h-58l29-50Z"/>',
      hammer:
        '<path d="M79 42h74v38H79z"/><path d="M101 80 70 132"/><path d="M133 80l-31 52"/><path d="M63 132h48"/>',
      lightsaber:
        '<path d="M58 123 154 27"/><path d="M142 39 161 20"/><path d="M51 130l25-25"/><path d="M74 107l16 16"/>',
      flux: '<path d="M110 30v46"/><path d="M70 124 106 82"/><path d="M150 124 114 82"/><circle cx="110" cy="82" r="10"/><path d="M88 40h44M55 126h30M135 126h30"/>',
      proton:
        '<path d="M83 48h54v74H83z"/><path d="M94 61h32M94 78h24M137 84c30-6 43 24 17 39"/><path d="M137 96c14-2 20 9 8 17"/><path d="M99 122v18M121 122v18"/>',
      kryptonite:
        '<path d="M110 24 74 82l36 54 36-54-36-58Z"/><path d="M74 82h72M110 24v112"/><path d="M84 36c14-14 38-14 52 0"/>',
      energon:
        '<path d="M80 35h60l18 37-48 64-48-64 18-37Z"/><path d="M82 72h76M110 35v101"/><path d="M64 129c24 12 68 12 92 0"/>',
      smog: '<path d="M61 101c-16 0-26-11-26-25s11-25 26-25c6-18 23-29 43-26 18 3 31 17 34 35 19-2 36 12 36 32 0 18-14 31-32 31H61Z"/><path d="M72 133h82M84 145h58"/>',
    };
    return (
      paths[icon] ||
      '<path d="M65 55h90v70H65z"/><path d="m65 55 45-25 45 25"/>'
    );
  }

  function formatCO2e(value) {
    return value >= 1000
      ? `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)} t CO2e`
      : `${value} kg CO2e`;
  }

  function getProduct(id) {
    return products.find((product) => product.id === id) || products[0];
  }

  function sortedProducts() {
    const copy = products.slice();
    return copy.sort((a, b) => {
      if (a.id === "slay-speech") return -1;
      if (b.id === "slay-speech") return 1;
      return b.score - a.score;
    });
  }

  function renderProducts() {
    productGrid.innerHTML = "";
    for (const product of sortedProducts()) {
      const card = document.createElement("button");
      card.className = "living-product-card";
      card.type = "button";
      card.dataset.productId = product.id;
      card.setAttribute(
        "aria-pressed",
        product.id === selectedId ? "true" : "false",
      );
      card.innerHTML = `
        <span class="living-product-art">
          <img src="${productImage(product)}" alt="${product.name}" loading="lazy" />
        </span>
        <span class="living-product-info">
          <span class="living-product-name">${product.name}</span>
          <span class="living-product-seller">${product.seller}</span>
          <span class="living-product-meta">
            <span>${product.price}</span>
            <span>${formatCO2e(product.co2e)}</span>
          </span>
        </span>
        <span class="living-product-badge">
          <img src="/images/demo/lca-128.png" alt="" aria-hidden="true" />
          <span>${product.badge}</span>
        </span>
      `;
      card.addEventListener("click", () => selectProduct(product.id));
      productGrid.append(card);
    }
  }

  function stageLabel(index) {
    return ["Materials", "Shipping", "Use", "Risk"][index];
  }

  function renderPanel(product) {
    score.textContent = product.score;
    scoreBar.style.width = `${product.score}%`;
    co2e.textContent = formatCO2e(product.co2e);
    water.textContent = product.water;
    repair.textContent = product.repair;
    risk.textContent = product.risk;
    note.textContent = product.note;
    equivalency.textContent = `Environmental impact is equivalent to about ${(product.co2e * 2.5).toLocaleString(undefined, { maximumFractionDigits: 1 })} miles driven.`;

    stages.innerHTML = product.stages
      .map(
        (value, index) => `
      <div class="living-stage-row">
        <span>${stageLabel(index)}</span>
        <span class="living-stage-track"><span style="width: ${value}%"></span></span>
        <b>${value}%</b>
      </div>
    `,
      )
      .join("");

    sources.innerHTML = product.sources
      .map((source) => {
        if (typeof source === "string") return `<li>${source}</li>`;
        return `<li><a href="${source.url}" target="_blank" rel="noopener">${source.label}</a></li>`;
      })
      .join("");
  }

  function selectProduct(id) {
    selectedId = id;
    renderProducts();
    renderPanel(getProduct(id));
  }

  renderProducts();
  renderPanel(getProduct(selectedId));
})();

(() => {
  const root = document.querySelector("[data-tennis-demo]");
  if (!root) return;

  const video = root.querySelector("[data-tennis-video]");
  const toggle = root.querySelector("[data-tennis-toggle]");
  const statusEl = root.querySelector("[data-tennis-status]");
  const timeline = root.querySelector("[data-tennis-timeline]");
  const kicker = root.querySelector("[data-tennis-stage-kicker]");
  const title = root.querySelector("[data-tennis-stage-title]");
  const summary = root.querySelector("[data-tennis-stage-summary]");
  const rulesEl = root.querySelector("[data-tennis-rules]");
  const imagePreview = root.querySelector("[data-tennis-image-preview]");
  const previewImage = root.querySelector("[data-tennis-preview-image]");

  let activeIndex = 0;
  let activePreviewButton = null;

  const stages = [
    {
      name: "Start",
      timelineName: "Start",
      range: [0, 0],
      clip: "start.mp4",
      color: "#0e746d",
      summary: "Checks the stance before motion begins: base width and shoulder line.",
      rules: [
        ["pass", "Foot Width", "Ankle width 2.06x shoulder width, meeting the >=0.80 stance threshold.", "1_1_foot_width_frame_0000.webp"],
        ["pass", "Shoulder Level", "Shoulder tilt is 38.5°, clearing the >15° setup criterion.", "1_2_shoulder_level_frame_0000.webp"],
      ],
    },
    {
      name: "Release",
      timelineName: "Release",
      range: [1, 12],
      clip: "release.mp4",
      color: "#1e88e5",
      summary: "Looks for the beginning of wind-up and whether the toss arm rises cleanly.",
      rules: [
        ["pass", "Elbow Rising", "Serving elbow is 88px above the hip as the wind-up starts.", "2_1_elbow_rising_frame_0006.webp"],
        ["warning", "Toss Arm Rising", "Tossing wrist is still 134px below the tossing shoulder during release.", "2_2_toss_arm_rising_frame_0006.webp"],
      ],
    },
    {
      name: "Loading",
      timelineName: "Loading",
      range: [13, 20],
      clip: "loading.mp4",
      color: "#7b61c8",
      summary: "Measures whether the body reaches a strong loaded position before the racket drop.",
      rules: [
        ["risk", "Elbow Height", "Serving elbow is 44px below the shoulder in loading position.", "3_1_elbow_height_frame_0016.webp"],
        ["pass", "Elbow Angle", "Serving elbow angle is 83.1°, inside the target loading range.", "3_2_elbow_angle_frame_0016.webp"],
        ["pass", "Toss Arm Extension", "Toss arm is extended at 173.5°.", "3_3_toss_arm_extension_frame_0016.webp"],
        ["warning", "Knee Bend", "Serving knee is too straight at loading: 171.2°, with <=160° desired.", "3_4_knee_bend_frame_0016.webp"],
      ],
    },
    {
      name: "Cocking",
      timelineName: "Cocking",
      range: [21, 34],
      clip: "cocking.mp4",
      color: "#f4a940",
      summary: "Checks whether the racket drops below the elbow before acceleration.",
      rules: [
        ["warning", "Racket Drop", "Serving wrist is not below elbow, suggesting the racket may not be fully loaded.", "4_1_racket_drop_frame_0028.webp"],
      ],
    },
    {
      name: "Acceleration",
      timelineName: "Accel.",
      range: [35, 41],
      clip: "acceleration.mp4",
      color: "#f47c54",
      summary: "Looks for upward elbow lead, wrist acceleration, and leg drive into the swing.",
      rules: [
        ["pass", "Elbow Leading", "Serving elbow is above the shoulder during the upward swing.", "5_1_elbow_leading_frame_0036.webp"],
        ["pass", "Wrist Acceleration", "Serving wrist moves 52px above the elbow during swing.", "5_2_wrist_acceleration_frame_0041.webp"],
        ["warning", "Leg Drive", "Serving knee remains flexed at 137.8°; >=155° is desired for leg drive.", "5_3_leg_drive_frame_0041.webp"],
      ],
    },
    {
      name: "Contact",
      timelineName: "Contact",
      range: [42, 42],
      clip: "contact.mp4",
      color: "#01bf67",
      summary: "Analyzes arm extension, contact height, and shoulder rotation at the strike.",
      rules: [
        ["pass", "Arm Extension", "Serving arm is extended at contact: 162.2°.", "6_1_arm_extension_frame_0042.webp"],
        ["pass", "Contact Height", "Contact proxy is 263px above shoulder level.", "6_2_contact_height_frame_0042.webp"],
        ["pass", "Shoulder Rotation", "Shoulder line is rotated 116.6° relative to the hips.", "6_3_shoulder_rotation_frame_0042.webp"],
      ],
    },
    {
      name: "Deceleration",
      timelineName: "Decel.",
      range: [43, 55],
      clip: "deceleration.mp4",
      color: "#d9534f",
      summary: "Checks follow-through mechanics that can indicate injury risk after contact.",
      rules: [
        ["risk", "Cross-Body Follow-Through", "Racket-arm wrist does not cross the body midline during follow-through.", "7_1_cross_body_follow_through_frame_0054.webp"],
        ["warning", "Elbow Deceleration", "Serving elbow is still extended in follow-through: 166.5°, with <=130° desired.", "7_2_elbow_deceleration_frame_0054.webp"],
      ],
    },
    {
      name: "Finish",
      timelineName: "Finish",
      range: [56, 61],
      clip: "finish.mp4",
      color: "#23313d",
      summary: "Looks at landing balance and knee flexion on both sides.",
      rules: [
        ["warning", "Landing Knee Flexion (Serving Side)", "Serving knee is too straight on landing: 168.2°, with <=160° desired.", "8_1_landing_knee_flexion_serving_side_frame_0061.webp"],
        ["pass", "Landing Knee Flexion (Non-Dominant Side)", "Non-dominant knee is 160.2°, indicating a balanced landing.", "8_2_landing_knee_flexion_non_dominant_side_frame_0061.webp"],
      ],
    },
  ];

  function labelFor(rule) {
    if (rule === "pass") return "Pass";
    if (rule === "risk") return "Risk";
    return "Warning";
  }

  function renderStage(index) {
    activeIndex = index;
    const stage = stages[index];
    kicker.textContent = `Stage ${index + 1} of ${stages.length}`;
    title.textContent = stage.name;
    summary.textContent = stage.summary;
    statusEl.textContent = `Frames ${stage.range[0]}-${stage.range[1]}`;
    rulesEl.innerHTML = stage.rules
      .map(
        ([severity, name, text, image]) => `
          <article class="tennis-rule-card" data-severity="${severity}">
            <img class="tennis-rule-thumbnail" src="/images/demo/tennis-serve/frames/${image}" alt="Annotated serve frame for ${name}" loading="lazy" decoding="async" />
            <div>
              <span class="tennis-rule-badge ${severity}">${labelFor(severity)}</span>
              <strong>${name}</strong>
              <p>${text}</p>
            </div>
            <button
              class="tennis-rule-view"
              type="button"
              data-tennis-rule-image="${image}"
              data-tennis-rule-name="${name}"
              aria-pressed="false"
            >View</button>
          </article>
        `
      )
      .join("");
    timeline.querySelectorAll("button").forEach((button, buttonIndex) => {
      button.classList.toggle("active", buttonIndex === index);
      button.setAttribute("aria-pressed", buttonIndex === index ? "true" : "false");
    });
  }

  async function replayStage() {
    hidePreview();
    video.currentTime = 0;
    try {
      await video.play();
    } catch {
      // Browsers may require a second explicit user gesture after a source change.
    }
  }

  async function selectStage(index) {
    const stage = stages[index];
    renderStage(index);
    hidePreview();
    video.pause();
    root.classList.add("tennis-loading-stage");
    video.src = `/images/demo/tennis-serve/clips/${stage.clip}`;
    video.load();
    try {
      await video.play();
    } catch {
      // The selected clip is still ready for the visible replay control.
    }
  }

  function hidePreview() {
    imagePreview.hidden = true;
    if (!activePreviewButton) return;
    activePreviewButton.textContent = "View";
    activePreviewButton.setAttribute("aria-pressed", "false");
    activePreviewButton.closest(".tennis-rule-card")?.classList.remove("is-viewing");
    activePreviewButton = null;
  }

  function showPreview(button) {
    hidePreview();
    video.pause();
    previewImage.src = `/images/demo/tennis-serve/frames/${button.dataset.tennisRuleImage}`;
    previewImage.alt = `Annotated serve frame for ${button.dataset.tennisRuleName}`;
    imagePreview.hidden = false;
    activePreviewButton = button;
    activePreviewButton.textContent = "Viewing";
    activePreviewButton.setAttribute("aria-pressed", "true");
    activePreviewButton.closest(".tennis-rule-card")?.classList.add("is-viewing");
  }

  stages.forEach((stage, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "tennis-stage-button";
    button.style.setProperty("--stage-color", stage.color);
    button.textContent = stage.timelineName;
    button.setAttribute("aria-label", `${stage.name}, frames ${stage.range[0]} through ${stage.range[1]}`);
    button.addEventListener("click", () => selectStage(index));
    timeline.append(button);
  });

  toggle.addEventListener("click", replayStage);
  video.addEventListener("loadeddata", () => root.classList.remove("tennis-loading-stage"));
  rulesEl.addEventListener("click", (event) => {
    const previewButton = event.target.closest("[data-tennis-rule-image]");
    if (!previewButton) return;
    if (previewButton === activePreviewButton) {
      hidePreview();
      return;
    }
    showPreview(previewButton);
  });

  renderStage(0);
})();

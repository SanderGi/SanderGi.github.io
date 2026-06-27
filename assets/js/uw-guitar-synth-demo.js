(function () {
  const root = document.querySelector("[data-uw-guitar-demo]");
  if (!root) return;

  const SAMPLE_FALLBACK = 44100;
  const DECAY = 0.996;
  const KEYBOARD = "q2we4r5ty7u8i9op-[=zxdcfvgbnjmk,.;/' ";
  const PITCH_OFFSET = 24;
  const VISIBLE_MIN_PITCH = -PITCH_OFFSET;
  const VISIBLE_MAX_PITCH = KEYBOARD.length - PITCH_OFFSET - 1;
  const SONG_MIN_PITCH = -57;
  const SONG_MAX_PITCH = 39;
  const CONCERT_A = 440;
  const SONGS = [
    ["ACDCThunderstruck.txt", "AC/DC Thunderstruck"],
    ["AWholeNewWorld.txt", "A Whole New World"],
    ["Avengers.txt", "Avengers"],
    ["FurElise.txt", "Fur Elise"],
    ["MoonlightSonata.txt", "Moonlight Sonata"],
    ["Queen-BohemianRhapsody.txt", "Bohemian Rhapsody"],
    ["bumblebee.txt", "Flight of the Bumblebee"],
    ["countryroads.txt", "Country Roads"],
    ["ifhappyandyouknowit.txt", "If You're Happy and You Know It"],
    ["mozartpiano9.txt", "Mozart Piano Sonata No. 9"],
    ["pirates.txt", "Pirates"],
    ["rachmaninocConcerto2-3.txt", "Rachmaninoff Concerto 2, III"],
    ["speechless.txt", "Speechless"],
    ["tetris.txt", "Tetris"],
    ["underthesea.txt", "Under the Sea"],
    ["youreOnlyLonely.txt", "You're Only Lonely"],
  ];

  const piano = root.querySelector("[data-guitar-piano]");
  const songSelect = root.querySelector("[data-guitar-song]");
  const playSongButton = root.querySelector("[data-guitar-play-song]");
  const stopButton = root.querySelector("[data-guitar-stop]");
  const status = root.querySelector("[data-guitar-status]");

  let audioContext = null;
  let processor = null;
  let guitar = null;
  let songTimers = [];
  let songEndTimer = null;
  let activeSongToken = 0;
  let raf = 0;

  class GuitarString {
    constructor(frequency, sampleRate) {
      const capacity = Math.max(2, Math.round(sampleRate / frequency));
      this.buffer = new Float32Array(capacity);
      this.index = 0;
    }

    pluck() {
      for (let i = 0; i < this.buffer.length; i += 1) {
        this.buffer[i] = Math.random() - 0.5;
      }
    }

    sample() {
      return this.buffer[this.index];
    }

    tic() {
      const next = (this.index + 1) % this.buffer.length;
      this.buffer[this.index] = (DECAY / 2) * (this.buffer[this.index] + this.buffer[next]);
      this.index = next;
    }
  }

  class Guitar37 {
    constructor(sampleRate) {
      this.minPitch = Math.min(SONG_MIN_PITCH, VISIBLE_MIN_PITCH);
      this.maxPitch = Math.max(SONG_MAX_PITCH, VISIBLE_MAX_PITCH);
      this.strings = Array.from({ length: this.maxPitch - this.minPitch + 1 }, (_, index) => {
        const pitch = this.minPitch + index;
        const frequency = CONCERT_A * Math.pow(2, pitch / 12);
        return new GuitarString(frequency, sampleRate);
      });
      this.lastLevels = new Float32Array(KEYBOARD.length);
    }

    stringForPitch(pitch) {
      if (pitch < this.minPitch || pitch > this.maxPitch) return null;
      return this.strings[pitch - this.minPitch];
    }

    pluckKey(key) {
      const index = KEYBOARD.indexOf(key);
      if (index === -1) return false;
      this.stringForPitch(index - PITCH_OFFSET).pluck();
      this.lastLevels[index] = 1;
      setKeyActive(index);
      return true;
    }

    playNote(pitch) {
      const string = this.stringForPitch(pitch);
      if (!string) return false;
      string.pluck();
      if (pitch >= VISIBLE_MIN_PITCH && pitch <= VISIBLE_MAX_PITCH) {
        const visibleIndex = pitch + PITCH_OFFSET;
        this.lastLevels[visibleIndex] = 1;
        setKeyActive(visibleIndex);
      }
      return true;
    }

    sample() {
      let sum = 0;
      for (const string of this.strings) sum += string.sample();
      return sum * 0.12;
    }

    tic() {
      for (let i = 0; i < this.strings.length; i += 1) {
        this.strings[i].tic();
      }
      for (let i = 0; i < this.lastLevels.length; i += 1) {
        const string = this.stringForPitch(i - PITCH_OFFSET);
        this.lastLevels[i] = Math.max(Math.abs(string.sample()) * 2.2, this.lastLevels[i] * 0.996);
      }
    }
  }

  function noteNameForPitch(pitch) {
    const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    const midi = 69 + pitch;
    return `${names[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
  }

  function keyLabel(key) {
    if (key === " ") return "Space";
    if (key === "'") return "'";
    return key;
  }

  function buildUi() {
    songSelect.innerHTML = SONGS.map(
      ([file, label]) => `<option value="${encodeURIComponent(file)}">${escapeHtml(label)}</option>`
    ).join("");

    piano.innerHTML = Array.from(KEYBOARD, (key, index) => {
      const pitch = index - PITCH_OFFSET;
      const note = noteNameForPitch(pitch);
      const isBlack = note.includes("#");
      return `
        <button class="guitar-key ${isBlack ? "black" : "white"}" type="button" data-guitar-key="${escapeHtml(
          key
        )}" data-string-index="${index}" aria-label="Play ${note} with keyboard key ${escapeHtml(keyLabel(key))}">
          <span class="guitar-string-line" aria-hidden="true"></span>
          <span class="guitar-note">${note}</span>
          <span class="guitar-keyboard">${escapeHtml(keyLabel(key))}</span>
        </button>
      `;
    }).join("");
  }

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  async function ensureAudio() {
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      guitar = new Guitar37(audioContext.sampleRate || SAMPLE_FALLBACK);
      processor = audioContext.createScriptProcessor(1024, 0, 1);
      processor.onaudioprocess = (event) => {
        const output = event.outputBuffer.getChannelData(0);
        for (let i = 0; i < output.length; i += 1) {
          output[i] = guitar.sample();
          guitar.tic();
        }
      };
      processor.connect(audioContext.destination);
      animateStrings();
    }
    if (audioContext.state === "suspended") await audioContext.resume();
  }

  async function pluckKey(key) {
    await ensureAudio();
    if (guitar.pluckKey(key)) setStatus("Plucked " + noteNameForPitch(KEYBOARD.indexOf(key) - PITCH_OFFSET));
  }

  function setKeyActive(index) {
    const key = piano.querySelector(`[data-string-index="${index}"]`);
    if (!key) return;
    key.classList.add("active");
    window.setTimeout(() => key.classList.remove("active"), 140);
  }

  function animateStrings() {
    const keys = Array.from(piano.querySelectorAll("[data-string-index]"));
    const tick = () => {
      if (guitar) {
        keys.forEach((key) => {
          const index = Number(key.dataset.stringIndex);
          const level = Math.min(1, guitar.lastLevels[index] || 0);
          const phase = performance.now() / 120 + index;
          key.style.setProperty("--wave", level.toFixed(3));
          key.style.setProperty("--string-x", `${Math.sin(phase) * level * 0.42}em`);
          key.style.setProperty("--string-glow-a", `${Math.sin(phase) * level * 0.65}em`);
          key.style.setProperty("--string-glow-b", `${Math.cos(phase) * level * -0.45}em`);
        });
      }
      raf = window.requestAnimationFrame(tick);
    };
    if (!raf) raf = window.requestAnimationFrame(tick);
  }

  function parseSong(text) {
    const values = text.trim().split(/\s+/);
    const events = [];
    for (let i = 0; i + 1 < values.length; i += 2) {
      const pitch = Number(values[i]);
      const duration = Number(values[i + 1]);
      if (Number.isFinite(pitch) && Number.isFinite(duration)) events.push({ pitch, duration });
    }
    return events;
  }

  async function playSong() {
    await ensureAudio();
    stopSong();
    const token = ++activeSongToken;
    const selected = songSelect.value;
    const label = songSelect.options[songSelect.selectedIndex]?.textContent || "song";
    setStatus(`Loading ${label}...`);
    playSongButton.disabled = true;

    try {
      const response = await fetch(`/assets/data/uw-guitar/alexmusic/${selected}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const events = parseSong(await response.text());
      let elapsed = 0;
      for (const event of events) {
        const timeout = window.setTimeout(() => {
          if (token !== activeSongToken) return;
          guitar.playNote(event.pitch);
        }, elapsed * 1000);
        songTimers.push(timeout);
        elapsed += Math.max(0, event.duration);
      }
      songEndTimer = window.setTimeout(() => {
        if (token === activeSongToken) {
          playSongButton.disabled = false;
          setStatus(`Finished ${label}.`);
        }
      }, elapsed * 1000 + 250);
      setStatus(`Playing ${label}.`);
    } catch {
      playSongButton.disabled = false;
      setStatus("Could not load that song.");
    }
  }

  function stopSong() {
    activeSongToken += 1;
    songTimers.forEach((timer) => window.clearTimeout(timer));
    songTimers = [];
    if (songEndTimer) window.clearTimeout(songEndTimer);
    songEndTimer = null;
    playSongButton.disabled = false;
  }

  function setStatus(message) {
    status.textContent = message;
  }

  buildUi();
  setStatus("Click a key or use your keyboard to pluck a synthesized string.");

  piano.addEventListener("click", (event) => {
    const key = event.target.closest("[data-guitar-key]");
    if (key) void pluckKey(key.dataset.guitarKey);
  });

  document.addEventListener("keydown", (event) => {
    if (event.repeat) return;
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (/^(a|button|input|select|textarea)$/i.test(event.target?.tagName || "")) return;
    const key = event.key === "Spacebar" ? " " : event.key;
    if (KEYBOARD.includes(key)) {
      event.preventDefault();
      void pluckKey(key);
    }
  });

  playSongButton.addEventListener("click", () => void playSong());
  stopButton.addEventListener("click", () => {
    stopSong();
    setStatus("Song stopped. The strings keep ringing until they decay.");
  });
})();

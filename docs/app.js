"use strict";

(() => {
  const PLAY_ICON = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 4.5v15l12-7.5z"/></svg>';
  const PAUSE_ICON = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M6 5h4v14H6zm8 0h4v14h-4z"/></svg>';
  const LOADING_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="8" opacity=".25"/><path d="M12 4a8 8 0 0 1 8 8"/></svg>';
  const players = [];
  let desiredPlayer = null;
  const stopButton = document.getElementById("stop-all");
  const announcement = document.getElementById("playback-announcement");

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function durationText(value) {
    const seconds = Math.max(0, Number.isFinite(value) ? value : 0);
    const minutes = Math.floor(seconds / 60);
    return `${minutes}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
  }

  function axisTime(value) {
    return `${Number(value.toFixed(2))} s`;
  }

  function sampleAnchor(id) {
    return id.startsWith("sample-") ? id : `sample-${id}`;
  }

  function sampleDisplayTitle(title) {
    const match = title.match(/^Sample\s+\d+\s*\((.+)\)\s*$/i);
    const label = (match ? match[1] : title).trim();
    return label.charAt(0).toUpperCase() + label.slice(1);
  }

  function stopPlayback() {
    desiredPlayer = null;
    for (const player of players) {
      player.pending = false;
      player.audio.pause();
      if (player.audio.readyState > 0) player.audio.currentTime = 0;
      player.render();
    }
    stopButton.disabled = true;
    announcement.textContent = "Playback stopped.";
  }

  stopButton.addEventListener("click", stopPlayback);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && desiredPlayer) stopPlayback();
  });

  function makePlayer(track, sample, group) {
    const card = element("article", "track");
    card.dataset.state = "ready";
    const label = `${sample.title}: ${track.label}`;
    const trackId = `${sample.id}-${group}-${track.nfe ?? track.kind}`;
    const labelRow = element("div", "track-label-row");
    const heading = element("h4", "", track.label);
    heading.id = `${trackId}-title`;
    card.setAttribute("aria-labelledby", heading.id);
    const download = element("a", "track-download", "WAV ↓");
    download.href = track.audio;
    download.download = "";
    download.setAttribute("aria-label", `Download ${label} as WAV`);
    labelRow.append(heading, download);

    const plot = element("div", "spectrogram-wrap");
    const frequency = element("div", "frequency-axis");
    frequency.setAttribute("aria-hidden", "true");
    ["8k", "4k", "0"].forEach((text) => frequency.append(element("span", "", text)));
    const toggle = element("button", "spectrogram-button");
    toggle.type = "button";
    toggle.setAttribute("aria-label", `Play ${label}`);
    toggle.setAttribute("aria-pressed", "false");
    const img = document.createElement("img");
    img.src = track.spectrogram;
    img.alt = `${track.label} log-magnitude spectrogram, time 0 to ${sample.duration_seconds} seconds, frequency 0 to 8 kilohertz`;
    img.width = 640;
    img.height = 240;
    img.loading = "lazy";
    img.decoding = "async";
    const imageFallback = element("span", "image-fallback", "Spectrogram unavailable");
    img.addEventListener("error", () => {
      img.hidden = true;
      if (!imageFallback.isConnected) toggle.append(imageFallback);
    });
    img.addEventListener("load", () => { img.hidden = false; imageFallback.remove(); });
    const icon = element("span", "play-icon");
    icon.innerHTML = PLAY_ICON;
    icon.setAttribute("aria-hidden", "true");
    const playhead = element("span", "playhead");
    playhead.setAttribute("aria-hidden", "true");
    toggle.append(img, playhead, icon);
    plot.append(frequency, toggle);
    const timeAxis = element("div", "time-axis");
    timeAxis.setAttribute("aria-hidden", "true");
    [0, sample.duration_seconds / 2, sample.duration_seconds].forEach((value) => timeAxis.append(element("span", "", axisTime(value))));

    const seek = element("input", "player-seek");
    seek.type = "range";
    seek.min = "0";
    seek.max = sample.duration_seconds.toString();
    seek.step = ".01";
    seek.value = "0";
    seek.setAttribute("aria-label", `Seek ${label}`);
    const meta = element("div", "player-meta");
    const state = element("span", "track-state", "Play");
    const time = element("span", "player-time", `0:00 / ${durationText(sample.duration_seconds)}`);
    time.setAttribute("aria-hidden", "true");
    meta.append(state, time);
    const error = element("p", "track-error");
    error.id = `${trackId}-error`;
    error.setAttribute("role", "status");
    error.hidden = true;
    const audio = document.createElement("audio");
    audio.src = track.audio;
    audio.preload = "none";
    audio.setAttribute("aria-label", label);
    card.append(labelRow, plot, timeAxis, seek, meta, error, audio);

    let frame = 0;
    let seeking = false;
    let pendingSeek = null;
    let seekingMetadata = false;
    let sourceRevision = 0;
    const player = { audio, pending: false, render, label, setMode };

    function setMode(mode) {
      const asset = mode === "matched" ? track.matched : track;
      sourceRevision += 1;
      player.pending = false;
      seeking = seekingMetadata = false;
      pendingSeek = null;
      cancelAnimationFrame(frame);
      audio.src = asset.audio;
      audio.load();
      img.src = asset.spectrogram;
      download.href = asset.audio;
      download.setAttribute("aria-label", `Download ${label}, ${mode === "matched" ? "RMS-matched" : "original relative levels"}, as WAV`);
      error.hidden = true;
      toggle.removeAttribute("aria-describedby");
      render();
    }

    function renderProgress() {
      const duration = Number.isFinite(audio.duration) ? audio.duration : sample.duration_seconds;
      const current = pendingSeek ?? (audio.currentTime || 0);
      if (!seeking) seek.value = current.toString();
      seek.max = duration.toString();
      seek.setAttribute("aria-valuetext", `${current.toFixed(1)} of ${duration.toFixed(1)} seconds`);
      time.textContent = `${durationText(current)} / ${durationText(duration)}`;
      const fraction = Math.min(1, Math.max(0, current / duration));
      playhead.style.left = `${fraction * 100}%`;
      playhead.style.opacity = current > 0 ? "1" : "0";
    }

    function render() {
      const playing = !audio.paused && !audio.ended;
      const loading = player.pending && desiredPlayer === player;
      card.dataset.state = loading ? "loading" : playing ? "playing" : "ready";
      icon.innerHTML = loading ? LOADING_ICON : playing ? PAUSE_ICON : PLAY_ICON;
      toggle.setAttribute("aria-label", `${loading ? "Cancel loading" : playing ? "Pause" : "Play"} ${label}`);
      toggle.setAttribute("aria-pressed", (playing || loading).toString());
      toggle.setAttribute("aria-busy", loading.toString());
      state.textContent = loading ? "Loading…" : playing ? "Playing" : audio.ended ? "Replay" : audio.currentTime > 0 ? "Paused" : "Play";
      stopButton.disabled = !desiredPlayer;
      renderProgress();
    }

    function animate() {
      renderProgress();
      if (!audio.paused && !audio.ended) frame = requestAnimationFrame(animate);
    }

    function showError(message) {
      player.pending = false;
      if (desiredPlayer === player) desiredPlayer = null;
      error.textContent = message;
      error.hidden = false;
      toggle.setAttribute("aria-describedby", error.id);
      render();
      state.textContent = "Try again";
    }

    async function playOrPause() {
      const revision = sourceRevision;
      if (desiredPlayer === player && (!audio.paused || player.pending)) {
        desiredPlayer = null;
        player.pending = false;
        audio.pause();
        render();
        return;
      }
      desiredPlayer = player;
      for (const other of players) {
        if (other !== player) {
          other.pending = false;
          other.audio.pause();
          other.render();
        }
      }
      error.hidden = true;
      toggle.removeAttribute("aria-describedby");
      if (audio.error) audio.load();
      if (audio.ended) audio.currentTime = 0;
      player.pending = true;
      render();
      try {
        await audio.play();
        if (revision !== sourceRevision) return;
        player.pending = false;
        if (desiredPlayer !== player) audio.pause();
        else announcement.textContent = `Playing ${label}.`;
        render();
      } catch (reason) {
        if (revision !== sourceRevision || desiredPlayer !== player || reason.name === "AbortError") return;
        showError("Audio could not play. Click to retry, or download the WAV file.");
      }
    }

    toggle.addEventListener("click", playOrPause);
    seek.addEventListener("input", () => {
      seeking = true;
      const value = Number(seek.value);
      // Browsers can defer fetching metadata until the first seek or play.
      if (audio.readyState === 0) {
        pendingSeek = value;
        if (!seekingMetadata) {
          seekingMetadata = true;
          audio.load();
        }
      } else {
        pendingSeek = null;
        audio.currentTime = value;
      }
      seeking = false;
      renderProgress();
    });
    audio.addEventListener("loadedmetadata", () => {
      seekingMetadata = false;
      if (pendingSeek !== null) {
        audio.currentTime = Math.min(pendingSeek, audio.duration);
        pendingSeek = null;
      }
      renderProgress();
    });
    audio.addEventListener("timeupdate", renderProgress);
    audio.addEventListener("playing", () => {
      player.pending = false;
      cancelAnimationFrame(frame);
      if (desiredPlayer !== player) audio.pause();
      else { render(); animate(); }
    });
    audio.addEventListener("waiting", () => {
      if (desiredPlayer === player) { player.pending = true; render(); }
    });
    audio.addEventListener("pause", () => { cancelAnimationFrame(frame); render(); });
    audio.addEventListener("ended", () => {
      cancelAnimationFrame(frame);
      if (desiredPlayer === player) desiredPlayer = null;
      player.pending = false;
      render();
    });
    audio.addEventListener("error", () => {
      seekingMetadata = false;
      showError("Audio is unavailable. Click to retry, or use the WAV download.");
    });
    players.push(player);
    renderProgress();
    return card;
  }

  function makeGroup(title, note, tracks, sample, group) {
    const section = element("section", "track-group");
    const groupHeading = element("div", "group-heading");
    const h4 = element("h3", "", title);
    h4.id = `${sample.id}-${group}-heading`;
    section.setAttribute("aria-labelledby", h4.id);
    groupHeading.append(h4);
    if (note) groupHeading.append(element("span", "", note));
    const grid = element("div", "tracks-grid");
    tracks.forEach((track) => grid.append(makePlayer(track, sample, group)));
    section.append(groupHeading, grid);
    return section;
  }

  function makeSample(sample, index) {
    const section = element("section", "sample");
    section.id = sampleAnchor(sample.id);
    section.setAttribute("aria-labelledby", `${sample.id}-heading`);
    const header = element("div", "sample-header");
    const number = element("span", "sample-number", (index + 1).toString().padStart(2, "0"));
    number.setAttribute("aria-hidden", "true");
    const heading = element("div", "sample-heading");
    const displayTitle = sampleDisplayTitle(sample.title);
    const h3 = element("h2", "", displayTitle);
    h3.id = `${sample.id}-heading`;
    h3.setAttribute("aria-label", sample.title);
    heading.append(h3);
    if (sample.description && !displayTitle.toLocaleLowerCase().includes(sample.description.trim().toLocaleLowerCase())) {
      heading.append(element("p", "sample-description", sample.description));
    }
    header.append(number, heading, element("span", "sample-duration", `${Number(sample.duration_seconds.toFixed(2))} sec / mono`));
    section.append(header);
    const cleanAvailable = sample.references.some((track) => track.kind === "clean");
    section.append(makeGroup("Source audio", cleanAvailable ? "Input and clean reference" : "No clean reference is available", sample.references, sample, "references"));
    section.append(makeGroup("Our restored outputs", "Same checkpoint · same starting noise", sample.outputs, sample, "outputs"));
    if (sample.source_note) {
      const note = element("p", "source-note", sample.source_note);
      if (sample.source_url) {
        const link = element("a", "", " Source ↗");
        link.href = sample.source_url;
        note.append(link);
      }
      section.append(note);
    }
    return section;
  }

  function validateManifest(data) {
    if (data.schema_version !== 1 || !Array.isArray(data.samples) || !data.samples.length) throw new Error("Unsupported results manifest.");
    if (!data.checkpoint || data.nfes.join(",") !== "1,20,50,100") throw new Error("Results are incomplete.");
    for (const sample of data.samples) {
      if (!Number.isFinite(sample.duration_seconds) || sample.duration_seconds <= 0) throw new Error("Invalid sample duration.");
      if (!Array.isArray(sample.references) || !Array.isArray(sample.outputs)) throw new Error("Missing audio tracks.");
      if (sample.outputs.map((track) => track.nfe).join(",") !== data.nfes.join(",")) throw new Error("Missing sampling result.");
      for (const track of [...sample.references, ...sample.outputs]) {
        if (!track.audio || !track.spectrogram || !track.label) throw new Error("Missing audio asset.");
      }
    }
  }

  async function loadResults() {
    const status = document.getElementById("page-status");
    status.hidden = false;
    status.classList.remove("error");
    status.textContent = "Loading audio examples…";
    try {
      const response = await fetch("results.json", { cache: "no-cache" });
      if (!response.ok) throw new Error(`Manifest returned ${response.status}`);
      const data = await response.json();
      validateManifest(data);
      const stepLabel = data.checkpoint.step.toLocaleString("en-US");
      document.querySelector(".section-helper").textContent = `${stepLabel}-update checkpoint · Click a spectrogram to play or pause. Use the slider to seek.`;
      const spectrum = data.spectrogram;
      const gainDb = data.playback?.common_gain_db;
      const gainLabel = Number.isFinite(gainDb)
        ? `${gainDb < 0 ? "−" : gainDb > 0 ? "+" : ""}${Math.abs(gainDb).toFixed(1)} dB`
        : "not recorded";
      document.getElementById("legend-note").textContent = `Shared playback gain ${gainLabel} · relative loudness preserved`;
      document.getElementById("scale-label").textContent = `${spectrum.min_db} to ${spectrum.max_db} dB`;
      document.querySelector(".color-scale").dataset.colormap = spectrum.colormap || "viridis";
      document.getElementById("spectrogram-details").textContent = `Audio: ${data.sample_rate.toLocaleString("en-US")} Hz, mono. Shared playback gain: ${gainLabel}, applied uniformly to every input and output; no per-clip normalization. Spectrograms: ${spectrum.n_fft}-sample STFT windows with a ${spectrum.hop_length}-sample hop; log magnitude displayed from ${spectrum.min_db} to ${spectrum.max_db} dB. ${spectrum.reference || "A shared reference magnitude of 1 is used for all window-normalized spectrograms."}`;
      const checkpointDetails = document.getElementById("checkpoint-details");
      const conditioning = data.checkpoint.conditioning_architecture === "scm-positional-adn-v1"
        ? "Positional time embeddings and adaptive double normalization"
        : `Gaussian Fourier frequency width ${data.checkpoint.frequency_width}`;
      checkpointDetails.replaceChildren(document.createTextNode(`${stepLabel} optimizer updates · ${data.checkpoint.weights} weights · ${conditioning}. Checkpoint SHA-256: `), element("code", "", data.checkpoint.sha256));
      const samples = document.getElementById("samples");
      const nav = document.getElementById("sample-navigation");
      stopPlayback();
      players.length = 0;
      samples.replaceChildren();
      nav.replaceChildren();
      data.samples.forEach((sample, index) => {
        samples.append(makeSample(sample, index));
        const link = element("a");
        link.href = `#${sampleAnchor(sample.id)}`;
        link.setAttribute("aria-label", `Jump to ${sample.title}`);
        link.append(element("span", "nav-number", (index + 1).toString().padStart(2, "0")), document.createTextNode(sampleDisplayTitle(sample.title)));
        nav.append(link);
      });
      window.RMFSRPlaybackMode.initialize(data, players, stopPlayback);
      status.hidden = true;
    } catch (error) {
      status.classList.add("error");
      status.textContent = "The audio examples could not load. Check your connection and try again.";
      const retry = element("button", "retry-button", "Retry");
      retry.type = "button";
      retry.addEventListener("click", loadResults);
      status.append(retry);
      console.error("Could not load listening results:", error);
    }
  }

  loadResults();
})();

"use strict";

window.RMFSRPlaybackMode = (() => {
  const controls = document.getElementById("playback-mode");
  const note = document.getElementById("playback-mode-note");
  const radios = [...controls.querySelectorAll("input")];
  let context = null;

  function hasMatched(data) {
    const tracks = data.samples.flatMap(sample => [...sample.references, ...sample.outputs]);
    return !!data.playback?.rms_matched && tracks.every(track => track.matched?.audio
      && track.matched?.spectrogram && Number.isFinite(track.matched.gain_db)
      && Number.isFinite(track.matched.rms_dbfs) && Number.isFinite(track.matched.peak));
  }

  function describe(mode) {
    const matched = mode === "matched";
    note.textContent = matched
      ? "Whole-clip RMS matched to the input, then one shared safety gain per example. This changes playback loudness only; it does not improve or rerun restoration."
      : "Original shared gain preserves relative loudness across inputs and outputs. Individual clips are not loudness-normalized.";
    document.getElementById("legend-note").textContent = matched
      ? "RMS matched to input · shared safety gain within each example" : context.legend;
    document.getElementById("spectrogram-details").textContent = matched
      ? `${context.spectrum} Spectrograms show the RMS-matched playback audio. Whole-clip RMS is matched to each input, followed by one shared safety gain within each example; recorded gains are in the results manifest.`
      : context.details;
  }

  function changeMode(mode) {
    if (!context || (mode === "matched" && !context.available)) return;
    context.stop();
    context.players.forEach(player => player.setMode(mode));
    describe(mode);
    document.getElementById("playback-announcement").textContent = mode === "matched"
      ? "Playback stopped. RMS-matched listening selected; playback gain only."
      : "Playback stopped. Original relative levels selected.";
  }

  function initialize(data, players, stop) {
    const s = data.spectrogram;
    context = { players, stop, available: hasMatched(data),
      legend: document.getElementById("legend-note").textContent,
      details: document.getElementById("spectrogram-details").textContent,
      spectrum: `Audio: ${data.sample_rate.toLocaleString("en-US")} Hz, mono. Spectrograms: ${s.n_fft}-sample STFT windows, ${s.hop_length}-sample hop, shared ${s.min_db} to ${s.max_db} dB scale.` };
    controls.disabled = false;
    radios.forEach(radio => { radio.checked = radio.value === "original";
      radio.disabled = radio.value === "matched" && !context.available; });
    describe("original");
    if (!context.available) note.textContent += " RMS-matched playback is not available for this snapshot.";
  }

  radios.forEach(radio => radio.addEventListener("change", () => {
    if (radio.checked) changeMode(radio.value);
  }));
  return { initialize };
})();

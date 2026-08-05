// The worker source is kept as a string and launched from a Blob URL so the app
// still works when index.html is opened directly from disk (file://), where
// loading a separate worker file is blocked by the browser's CORS rules.
const WORKER_SOURCE = `
import { pipeline, WhisperTextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js";

let transcriber = null;
let currentModel = null;
let currentDevice = null;
let jobId = null;

const post = (msg) => self.postMessage(Object.assign({ id: jobId }, msg));

async function detectDevice() {
  try {
    if (self.navigator && self.navigator.gpu) {
      const adapter = await self.navigator.gpu.requestAdapter();
      if (adapter) return "webgpu";
    }
  } catch (e) { /* fall through */ }
  return "wasm";
}

async function loadModel(model) {
  const progress_callback = (p) => {
    if (p && p.status === "progress") {
      post({ type: "model-progress", file: p.file, progress: p.progress || 0, loaded: p.loaded, total: p.total });
    } else if (p && p.status === "initiate") {
      post({ type: "model-file", file: p.file });
    }
  };
  let device = await detectDevice();
  const opts = (dev) => ({
    device: dev,
    dtype: dev === "webgpu" ? { encoder_model: "fp32", decoder_model_merged: "q4" } : "q8",
    progress_callback,
  });
  try {
    transcriber = await pipeline("automatic-speech-recognition", model, opts(device));
  } catch (err) {
    if (device === "webgpu") {
      device = "wasm";
      transcriber = await pipeline("automatic-speech-recognition", model, opts(device));
    } else {
      throw err;
    }
  }
  currentModel = model;
  currentDevice = device;
}

self.onmessage = async (e) => {
  const { audio, model, duration, language } = e.data;
  jobId = e.data.id;
  try {
    if (!transcriber || currentModel !== model) {
      transcriber = null;
      post({ type: "status", text: "Loading model (first time only)…" });
      await loadModel(model);
    }
    post({ type: "ready", device: currentDevice });

    const streamer = new WhisperTextStreamer(transcriber.tokenizer, {
      time_precision: transcriber.processor.feature_extractor.config.chunk_length /
                      transcriber.model.config.max_source_positions,
      on_chunk_start: (t) => post({ type: "chunk-start", time: t, duration }),
      callback_function: (text) => post({ type: "partial", text }),
    });

    const genOpts = {
      chunk_length_s: 30,
      stride_length_s: 5,
      return_timestamps: true,
      streamer,
    };
    if (language && language !== "auto") {
      genOpts.language = language;
      genOpts.task = "transcribe";
    }
    const output = await transcriber(audio, genOpts);

    post({ type: "done", text: output.text, chunks: output.chunks || [] });
  } catch (err) {
    let message = (err && err.message) ? err.message : String(err);
    if (/fetch|network|Failed to load/i.test(message)) {
      message = "Model download failed (" + message + "). Check your internet connection and reload. The model only needs to download once.";
    }
    post({ type: "error", message });
  }
};
`;

(() => {
  const $ = (id) => document.getElementById(id);
  const dropzone = $("dropzone"), fileInput = $("fileInput"),
        progressCard = $("progressCard"), resultCard = $("resultCard"),
        statusText = $("statusText"), mainBar = $("mainBar"),
        fileRows = $("fileRows"), errorBox = $("errorBox"),
        transcript = $("transcript"), spinner = $("spinner"),
        deviceBadge = $("deviceBadge");

  const MAX_FILE_BYTES = 750 * 1024 * 1024;
  const PREFS_KEY = "babelfish-prefs";

  let worker = null;
  let busy = false;
  let liveText = "";
  let audioDuration = 0;
  let doneChunks = null;
  let baseName = "transcript";
  // Incremented for every job and on reset; the worker echoes it back on each
  // message so output from a cancelled/stale job can't repopulate the UI.
  let currentJob = 0;
  const fileBars = new Map();

  // Remember model / language / timestamp choices across visits.
  // localStorage can throw (private mode, blocked storage): never fatal.
  function loadPrefs() {
    try { return JSON.parse(localStorage.getItem(PREFS_KEY)) || {}; } catch (e) { return {}; }
  }
  function savePrefs() {
    try {
      localStorage.setItem(PREFS_KEY, JSON.stringify({
        model: $("modelSelect").value,
        language: $("langSelect").value,
        timestamps: $("tsToggle").checked,
      }));
    } catch (e) { /* storage unavailable: ignore */ }
  }
  (() => {
    const p = loadPrefs();
    const setIfValid = (sel, v) => {
      if (v != null && [...sel.options].some((o) => o.value === v)) sel.value = v;
    };
    setIfValid($("modelSelect"), p.model);
    setIfValid($("langSelect"), p.language);
    if (typeof p.timestamps === "boolean") $("tsToggle").checked = p.timestamps;
  })();
  ["modelSelect", "langSelect"].forEach((id) => $(id).addEventListener("change", savePrefs));

  // WebGPU badge (informational; the worker does its own detection)
  (async () => {
    let label = "CPU (WASM)";
    try {
      if (navigator.gpu && await navigator.gpu.requestAdapter()) label = "GPU (WebGPU)";
    } catch (e) {}
    deviceBadge.textContent = label;
  })();

  function getWorker() {
    if (worker) return worker;
    const blob = new Blob([WORKER_SOURCE], { type: "text/javascript" });
    worker = new Worker(URL.createObjectURL(blob), { type: "module" });
    worker.onmessage = onWorkerMessage;
    worker.onerror = (e) => showError(
      "The transcription engine failed to start" + (e.message ? " (" + e.message + ")" : "") +
      ". It loads from a CDN on first use, so check your internet connection and reload the page."
    );
    return worker;
  }

  function stopWorker() {
    if (worker) {
      worker.terminate();
      worker = null;
    }
  }

  function fmtTime(s) {
    s = Math.max(0, Math.round(s));
    const m = Math.floor(s / 60), sec = s % 60;
    return String(m).padStart(2, "0") + ":" + String(sec).padStart(2, "0");
  }

  function fmtBytes(n) {
    if (!n) return "";
    if (n > 1e6) return (n / 1e6).toFixed(0) + " MB";
    return (n / 1e3).toFixed(0) + " KB";
  }

  function showError(msg) {
    progressCard.classList.remove("hidden");
    spinner.classList.add("hidden");
    errorBox.textContent = msg;
    errorBox.classList.remove("hidden");
    statusText.textContent = "Something went wrong";
    busy = false;
  }

  function resetUI() {
    progressCard.classList.add("hidden");
    resultCard.classList.add("hidden");
    errorBox.classList.add("hidden");
    spinner.classList.remove("hidden");
    fileRows.innerHTML = "";
    fileBars.clear();
    mainBar.style.width = "0%";
    transcript.value = "";
    liveText = "";
    doneChunks = null;
    busy = false;
  }

  function onWorkerMessage(e) {
    const m = e.data;
    if (m.id != null && m.id !== currentJob) return; // stale job: ignore
    switch (m.type) {
      case "status":
        statusText.textContent = m.text;
        break;
      case "model-file": {
        if (!fileBars.has(m.file)) {
          const row = document.createElement("div");
          row.className = "file-row";
          const name = document.createElement("span");
          name.textContent = m.file;
          const pct = document.createElement("span");
          pct.textContent = "0%";
          row.append(name, pct);
          fileRows.appendChild(row);
          fileBars.set(m.file, pct);
        }
        break;
      }
      case "model-progress": {
        const el = fileBars.get(m.file);
        if (el) el.textContent = Math.round(m.progress) + "%" + (m.total ? " of " + fmtBytes(m.total) : "");
        // Rough overall model bar: average of all files seen
        let sum = 0;
        fileBars.forEach((span) => { sum += parseFloat(span.textContent) || 0; });
        mainBar.style.width = (sum / Math.max(1, fileBars.size)) + "%";
        break;
      }
      case "ready":
        deviceBadge.textContent = m.device === "webgpu" ? "GPU (WebGPU)" : "CPU (WASM)";
        statusText.textContent = "Transcribing…";
        fileRows.innerHTML = "";
        fileBars.clear();
        mainBar.style.width = "0%";
        resultCard.classList.remove("hidden");
        break;
      case "chunk-start":
        if (audioDuration > 0) {
          const pct = Math.min(99, (m.time / audioDuration) * 100);
          mainBar.style.width = pct + "%";
          statusText.textContent = "Transcribing… " + fmtTime(m.time) + " / " + fmtTime(audioDuration);
        }
        break;
      case "partial":
        liveText += m.text;
        transcript.value = liveText;
        transcript.scrollTop = transcript.scrollHeight;
        break;
      case "done": {
        doneChunks = m.chunks;
        renderFinal(m);
        mainBar.style.width = "100%";
        spinner.classList.add("hidden");
        statusText.textContent = "Done";
        busy = false;
        break;
      }
      case "error":
        showError(m.message);
        break;
    }
  }

  function renderFinal(m) {
    const withTs = $("tsToggle").checked;
    if (withTs && m.chunks && m.chunks.length) {
      transcript.value = m.chunks.map((c) => {
        const t0 = c.timestamp && c.timestamp[0] != null ? fmtTime(c.timestamp[0]) : "??:??";
        const t1 = c.timestamp && c.timestamp[1] != null ? fmtTime(c.timestamp[1]) : "??:??";
        return "[" + t0 + " - " + t1 + "] " + c.text.trim();
      }).join("\n");
    } else {
      transcript.value = (m.text || "").trim();
    }
  }

  async function decodeToMono16k(file) {
    const arrayBuf = await file.arrayBuffer();
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx({ sampleRate: 16000 });
    try {
      const decoded = await ctx.decodeAudioData(arrayBuf);
      audioDuration = decoded.duration;
      let audio;
      if (decoded.numberOfChannels > 1) {
        audio = new Float32Array(decoded.length);
        for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
          const data = decoded.getChannelData(ch);
          for (let i = 0; i < data.length; i++) audio[i] += data[i] / decoded.numberOfChannels;
        }
      } else {
        audio = decoded.getChannelData(0);
      }
      return audio;
    } finally {
      ctx.close();
    }
  }

  async function handleFile(file) {
    if (!file || busy) return;
    resetUI();
    if (file.size === 0) {
      showError("\"" + file.name + "\" is empty (0 bytes). Pick a different file.");
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      showError("\"" + file.name + "\" is " + fmtBytes(file.size) +
        ", which is too large to decode in the browser. Trim it or convert it to a compressed format like mp3 first.");
      return;
    }
    busy = true;
    currentJob++;
    baseName = (file.name || "transcript").replace(/\.[^.]+$/, "");
    progressCard.classList.remove("hidden");
    statusText.textContent = "Decoding audio…";
    try {
      const audio = await decodeToMono16k(file);
      statusText.textContent = "Starting…";
      const model = $("modelSelect").value;
      const language = $("langSelect").value;
      getWorker().postMessage({ id: currentJob, audio, model, duration: audioDuration, language }, [audio.buffer]);
    } catch (err) {
      showError("Could not decode this file as audio (" + (err.message || err) + "). Try converting it to mp3 or wav.");
    }
  }

  // Drag & drop + click + keyboard
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));
  ["dragover", "dragenter"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
  dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files && e.dataTransfer.files[0]));

  // Timestamp toggle re-renders finished transcript
  $("tsToggle").addEventListener("change", () => {
    savePrefs();
    if (doneChunks) renderFinal({ text: liveText, chunks: doneChunks });
  });

  // Buttons
  let copyFlashTimer = null;
  function flashCopyLabel(text) {
    clearTimeout(copyFlashTimer);
    $("copyBtn").textContent = text;
    copyFlashTimer = setTimeout(() => { $("copyBtn").textContent = "Copy"; }, 1500);
  }
  $("copyBtn").addEventListener("click", async () => {
    if (!transcript.value) { flashCopyLabel("Nothing to copy"); return; }
    let ok = false;
    try {
      await navigator.clipboard.writeText(transcript.value);
      ok = true;
    } catch (err) {
      // Clipboard API can be unavailable (http, permissions): fall back to selection copy
      try {
        transcript.focus();
        transcript.select();
        ok = document.execCommand("copy");
        transcript.setSelectionRange(0, 0);
      } catch (err2) { /* both paths failed */ }
    }
    flashCopyLabel(ok ? "Copied!" : "Copy failed");
  });
  $("downloadBtn").addEventListener("click", () => {
    if (!transcript.value) return;
    const blob = new Blob([transcript.value], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = baseName + ".txt";
    a.click();
    URL.revokeObjectURL(a.href);
  });
  $("resetBtn").addEventListener("click", () => {
    // Mid-job reset is a cancel: kill the worker so it stops burning CPU.
    // (The model re-downloads from browser cache on the next run, so this is cheap.)
    if (busy) stopWorker();
    currentJob++;
    resetUI();
    fileInput.value = "";
  });
})();

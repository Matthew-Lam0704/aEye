let lastSpoken = "";
let latestText = "";
let lastSpokenAt = 0;
let selectedVoice = null;
let pauseInfer = false;
let ocrBusy = false;
let startupPromptPending = true;
const STARTUP_MESSAGE = "The app has started. Single tap to stop reading. Double tap to read what is in front of you. Triple tap to read text in front of you.";
let lastUrgentSpoken = "";
let lastUrgentAt = 0;
const URGENT_CLIENT_COOLDOWN_MS = 1800;
let suppressUrgentUntil = 0;
const SPEECH_FIX_RULES = [
  // Common OCR and pronunciation pain points.
  [/\binstro+ments?\b/gi, "instruments"],
  [/\binstrooments?\b/gi, "instruments"],
  [/\binstrument5\b/gi, "instruments"],
  [/\binstrurnents?\b/gi, "instruments"],
  [/\blnstruments?\b/gi, "instruments"],
  // Read list punctuation more naturally.
  [/\s*&\s*/g, " and "],
  [/\s*\/\s*/g, " slash "]
];

const video = document.getElementById("cam");
const canvas = document.getElementById("grab");
const readTextBtn = document.getElementById("readText");
const tapSurface = document.getElementById("tapSurface");
const cameraStatus = document.getElementById("cameraStatus");
const annotatedImg = document.getElementById("annotated");
let activeSpeechSequence = null;

function setCameraStatus(text) {
  if (cameraStatus) {
    cameraStatus.textContent = text;
  }
}

function pickVoice() {
  const voices = speechSynthesis.getVoices();
  const englishVoices = voices.filter((v) => /en/i.test(v.lang));
  const tier1 = englishVoices.find((v) => /enhanced|premium|natural|neural/i.test(v.name));
  if (tier1) {
    selectedVoice = tier1;
    return;
  }

  const preferred = englishVoices.find(
    (v) => /samantha|daniel|karen|moira|google uk english female|google us english/i.test(v.name)
  );
  if (preferred) {
    selectedVoice = preferred;
    return;
  }

  selectedVoice = englishVoices[0] || voices[0] || null;
}
speechSynthesis.onvoiceschanged = pickVoice;
pickVoice();

function normalizeSpeechText(text) {
  let out = (text || "").replace(/\s+/g, " ").trim();
  if (!out) return out;
  for (const [pattern, replacement] of SPEECH_FIX_RULES) {
    out = out.replace(pattern, replacement);
  }
  return out;
}

function buildUtterance(text, mode = "default") {
  const safeText = normalizeSpeechText(text);
  const u = new SpeechSynthesisUtterance(safeText);
  if (selectedVoice) {
    u.voice = selectedVoice;
    if (selectedVoice.lang) u.lang = selectedVoice.lang;
  } else {
    u.lang = "en-GB";
  }

  // Slightly slower + warmer cadence sounds less robotic.
  if (mode === "urgent") {
    u.rate = 0.94;
    u.pitch = 1.0;
  } else if (mode === "ocr") {
    u.rate = 0.87;
    u.pitch = 1.0;
  } else {
    u.rate = 0.9;
    u.pitch = 1.0;
  }
  u.volume = 1.0;
  return u;
}

function speak(text, opts = {}) {
  const { force = false, dedupe = true, mode = "default" } = opts;
  if (!text) return;

  const now = Date.now();
  if (!force && now - lastSpokenAt < 2500) return;
  if (dedupe && text === lastSpoken) return;

  lastSpoken = text;
  lastSpokenAt = now;

  const u = buildUtterance(text, mode);
  speechSynthesis.cancel();
  speechSynthesis.speak(u);

  const el = document.getElementById("lastSpeech");
  if (el) el.textContent = text;
}

function stopSpeaking({ updateUi = true } = {}) {
  speechSynthesis.cancel();
  if (activeSpeechSequence) {
    activeSpeechSequence.cancelled = true;
    activeSpeechSequence.resolveNow();
    activeSpeechSequence = null;
  }

  if (updateUi && readTextBtn) {
    readTextBtn.textContent = "Read Text";
    readTextBtn.disabled = false;
  }
}

function speakUrgent(text) {
  if (!text) return;
  const now = Date.now();
  if (text === lastUrgentSpoken && now - lastUrgentAt < URGENT_CLIENT_COOLDOWN_MS) return;
  lastUrgentSpoken = text;
  lastUrgentAt = now;
  speak(text, { force: true, dedupe: false, mode: "urgent" });
}

function speakWithStartDetection(text) {
  return new Promise((resolve) => {
    const u = buildUtterance(text, "default");

    let done = false;
    const finish = (started) => {
      if (done) return;
      done = true;
      resolve(started);
    };

    u.onstart = () => finish(true);
    u.onerror = () => finish(false);
    setTimeout(() => finish(false), 1200);

    speechSynthesis.cancel();
    speechSynthesis.speak(u);

    const el = document.getElementById("lastSpeech");
    if (el) el.textContent = text;
  });
}

async function announceStartup() {
  if (!startupPromptPending) return;
  const started = await speakWithStartDetection(STARTUP_MESSAGE);
  if (started) {
    startupPromptPending = false;
  }
}

async function startCamera() {
  if (video && video.srcObject) return true;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setCameraStatus("Camera is not supported on this browser.");
    return false;
  }

  try {
    setCameraStatus("Starting camera…");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false
    });

    if (!video) return false;
    video.srcObject = stream;
    await video.play();
    setCameraStatus("Camera running ✅");
    return true;
  } catch (e) {
    setCameraStatus("Camera blocked. Allow camera permission and refresh.");
    console.log(e);
    return false;
  }
}

async function sendFrame() {
  if (pauseInfer || ocrBusy) return;
  if (!video || !video.srcObject) return;

  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;

  const targetW = 640;
  const targetH = Math.round((h / w) * targetW);
  canvas.width = targetW;
  canvas.height = targetH;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, targetW, targetH);

  const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.7));
  if (!blob) return;

  const form = new FormData();
  form.append("frame", blob, "frame.jpg");

  try {
    const r = await fetch("/infer", { method: "POST", body: form });
    const data = await r.json();

    if (annotatedImg && data.annotated_jpg) {
      annotatedImg.style.display = "block";
      annotatedImg.src = "data:image/jpeg;base64," + data.annotated_jpg;
    }

    if (data.urgent_speech && Date.now() > suppressUrgentUntil) {
      // Safety alerts should be heard immediately.
      speakUrgent(data.urgent_speech);
    }

    const summary = (data.summary_speech || data.speech || "").trim();
    if (summary) {
      // Keep summary updated silently; user hears it on double tap.
      latestText = summary;
    }
  } catch (e) {
    console.log("Infer error:", e);
  }
}

async function runOCRCapture(targetW = 960, quality = 0.85, filename = "ocr.jpg") {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return "";

  const targetH = Math.round((h / w) * targetW);
  canvas.width = targetW;
  canvas.height = targetH;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, targetW, targetH);

  const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", quality));
  if (!blob) return "";

  const form = new FormData();
  form.append("frame", blob, filename);

  const r = await fetch("/ocr", { method: "POST", body: form });
  const data = await r.json();
  return (data.text || "").trim();
}

function formatOcrForSpeech(text) {
  let cleaned = normalizeSpeechText(text);
  if (!cleaned) return "";

  // Ensure pauses are respected.
  cleaned = cleaned
    .replace(/([,;:.!?])(?!\s|$)/g, "$1 ")
    .replace(/\s{2,}/g, " ");

  if (!/[.!?]$/.test(cleaned) && cleaned.length > 30) {
    cleaned += ".";
  }
  return cleaned;
}

function chunkSpeechText(text, maxLen = 180) {
  if (!text) return [];
  const sentenceParts = text.split(/(?<=[.!?])\s+/).filter(Boolean);
  if (!sentenceParts.length) return [text];

  const chunks = [];
  let current = "";
  for (const part of sentenceParts) {
    if ((current + " " + part).trim().length <= maxLen) {
      current = (current ? current + " " : "") + part;
    } else {
      if (current) chunks.push(current.trim());
      current = part;
    }
  }
  if (current) chunks.push(current.trim());
  return chunks;
}

function speakInSequence(chunks, mode = "default") {
  return new Promise((resolve) => {
    if (!chunks.length) {
      resolve();
      return;
    }

    // Ensure only one active sequence and make it stoppable.
    stopSpeaking({ updateUi: false });
    const sequence = {
      cancelled: false,
      done: false,
      resolveNow: () => {
        if (sequence.done) return;
        sequence.done = true;
        resolve();
      }
    };
    activeSpeechSequence = sequence;

    let idx = 0;
    const finish = () => {
      if (activeSpeechSequence === sequence) {
        activeSpeechSequence = null;
      }
      sequence.resolveNow();
    };

    const speakNext = () => {
      if (sequence.cancelled) {
        finish();
        return;
      }
      if (idx >= chunks.length) {
        finish();
        return;
      }
      const piece = chunks[idx++];
      const u = buildUtterance(piece, mode);
      u.onend = speakNext;
      u.onerror = speakNext;
      speechSynthesis.speak(u);
    };
    speakNext();

    const el = document.getElementById("lastSpeech");
    if (el) el.textContent = chunks.join(" ");
  });
}

async function readTextFromCamera() {
  if (ocrBusy) return;
  if (!video || !video.srcObject) {
    const started = await startCamera();
    if (!started) {
      speak("Camera is not available yet. Please allow camera permission.", { force: true, dedupe: false });
      return;
    }
  }

  ocrBusy = true;
  pauseInfer = true;
  suppressUrgentUntil = Date.now() + 4500;
  const oldText = readTextBtn.textContent;

  try {
    readTextBtn.textContent = "Reading…";
    readTextBtn.disabled = true;
    const text = await runOCRCapture(960, 0.85, "ocr.jpg");
    const formatted = formatOcrForSpeech(text);
    if (!formatted) {
      speak("I could not find readable text. Try moving closer.", { force: true, dedupe: false, mode: "ocr" });
    } else {
      const chunks = chunkSpeechText(formatted, 170);
      await speakInSequence(chunks, "ocr");
    }
  } catch (e) {
    speak("There was a problem reading text.", { force: true, dedupe: false, mode: "ocr" });
    console.log("Read Text error:", e);
  } finally {
    ocrBusy = false;
    pauseInfer = false;
    readTextBtn.textContent = oldText;
    readTextBtn.disabled = false;
  }
}

readTextBtn.onclick = async () => {
  await readTextFromCamera();
};

function speakLatestSummary() {
  const text = latestText || "I do not have a scene summary yet.";
  speak(text, { force: true, dedupe: false });
}

let tapCount = 0;
let tapTimer = null;
tapSurface.addEventListener("pointerup", async () => {
  if (startupPromptPending) {
    // iOS/Safari may block TTS on page load; first user gesture unlocks it.
    announceStartup();
    return;
  }

  if (!video.srcObject) {
    startCamera();
  }

  tapCount += 1;
  if (tapTimer) {
    clearTimeout(tapTimer);
  }

  tapTimer = setTimeout(async () => {
    const count = tapCount;
    tapCount = 0;
    tapTimer = null;

    if (count >= 3) {
      await readTextFromCamera();
      return;
    }
    if (count === 2) {
      speakLatestSummary();
      return;
    }
    if (count === 1) {
      if (speechSynthesis.speaking || ocrBusy || activeSpeechSequence) {
        stopSpeaking();
        const el = document.getElementById("lastSpeech");
        if (el) el.textContent = "Speech stopped.";
      }
    }
  }, 420);
});

window.addEventListener("load", async () => {
  await announceStartup();
  await startCamera();
});

setInterval(sendFrame, 500);

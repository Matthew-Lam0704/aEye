let speechEnabled = false;
let lastSpoken = "";
let latestText = "";
let lastSpokenAt = 0;
let selectedVoice = null;
let pauseInfer = false;
let ocrBusy = false;

const video = document.getElementById("cam");
const canvas = document.getElementById("grab");
const startBtn = document.getElementById("startCam");
const enableBtn = document.getElementById("enable");
const speakNowBtn = document.getElementById("speakNow");

function pickVoice() {
  const voices = speechSynthesis.getVoices();
  selectedVoice = voices.find(v => /en/i.test(v.lang)) || voices[0] || null;
}
speechSynthesis.onvoiceschanged = pickVoice;
pickVoice();

enableBtn.onclick = () => {
  speechEnabled = true;
  enableBtn.textContent = "Speech enabled ✅";
  enableBtn.style.background = "#28a745";

  // Unlock speech on iOS (must happen on a user tap)
  const u = new SpeechSynthesisUtterance("Speech enabled");
  if (selectedVoice) u.voice = selectedVoice;
  u.rate = 1.0;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
};

function speak(text) {
  if (!speechEnabled) return;
  if (!text) return;

  // throttle so iOS doesn't block / spam
  const now = Date.now();
  if (now - lastSpokenAt < 2500) return;
  if (text === lastSpoken) return;

  lastSpoken = text;
  lastSpokenAt = now;

  const u = new SpeechSynthesisUtterance(text);
  if (selectedVoice) u.voice = selectedVoice;
  u.rate = 1.0;

  speechSynthesis.speak(u);

  const el = document.getElementById("lastSpeech");
  if (el) el.textContent = text;
}

startBtn.onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    startBtn.textContent = "Camera running ✅";
    startBtn.style.background = "#28a745";
  } catch (e) {
    alert("Camera error: " + e.name + " — " + e.message);
    console.log(e);
  }
};

async function sendFrame() {
  async function sendFrame() {
  if (pauseInfer) return;
  // ... existing code
  if (!video.srcObject) return;

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

  const r = await fetch("/infer", { method: "POST", body: form });
  const data = await r.json();

  const img = document.getElementById("annotated");
  if (img && data.annotated_jpg) {
    img.style.display = "block";
    img.src = "data:image/jpeg;base64," + data.annotated_jpg;
  }

  const jsonEl = document.getElementById("json");
  if (jsonEl) jsonEl.textContent = JSON.stringify(data, null, 2);

  if (data.speech) {
    latestText = data.speech;
    speak(data.speech);
  }
  }
}

document.getElementById("readText").onclick = async () => {
  if (ocrBusy) return;
  if (!video.srcObject) return alert("Start camera first");

  ocrBusy = true;
  pauseInfer = true; // ✅ pause normal detection while OCR runs

  try {
    // (optional) show user feedback
    const btn = document.getElementById("readText");
    const oldText = btn.textContent;
    btn.textContent = "Reading…";
    btn.disabled = true;

    // --- your OCR capture + fetch code here ---
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;

    // IMPORTANT: keep OCR capture moderate so it’s not too slow
    const targetW = 960; // ✅ try 960 first (1280 can be heavy on iPhone)
    const targetH = Math.round((h / w) * targetW);

    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, targetW, targetH);

    const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.85));
    if (!blob) return;

    const form = new FormData();
    form.append("frame", blob, "ocr.jpg");

    const r = await fetch("/ocr", { method: "POST", body: form });
    const data = await r.json();

    const text = (data.text || "").trim();
    const say = text || "I could not find readable text. Try moving closer.";

    // speak immediately (user gesture)
    const u = new SpeechSynthesisUtterance(say);
    u.rate = 1.0;
    speechSynthesis.cancel();
    speechSynthesis.speak(u);

    // restore button
    btn.textContent = oldText;
    btn.disabled = false;

  } catch (e) {
    alert("Read Text error: " + e.message);
  } finally {
    ocrBusy = false;
    pauseInfer = false; // ✅ resume normal detection
  }
};

speakNowBtn.onclick = () => {
  if (!speechEnabled) return;
  if (!latestText) return;

  // This is a user gesture, so it's safe to cancel + speak immediately
  const u = new SpeechSynthesisUtterance(latestText);
  if (selectedVoice) u.voice = selectedVoice;
  u.rate = 1.0;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
};

// ~2 fps
let inferTimer = setInterval(sendFrame, 500);

let latestText = "";
let pauseInfer = false;
let ocrBusy = false;

const video = document.getElementById("cam");
const canvas = document.getElementById("grab");
const startBtn = document.getElementById("startCam");
const readTextBtn = document.getElementById("readText");
const tapSurface = document.getElementById("tapSurface");
const img = document.getElementById("annotated");

function speakNow(text) {
  if (!text) return;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.0;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

function speakLatestSummary() {
  speakNow(latestText || "No summary yet. Point the camera at something.");
}

// Double tap anywhere to speak
let lastTap = 0;
tapSurface.addEventListener("pointerup", () => {
  const now = Date.now();
  if (now - lastTap < 350) {
    speakLatestSummary();
    lastTap = 0;
  } else {
    lastTap = now;
  }
});

// Start camera
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

    // iPhone will allow speech here because user tapped Start Camera
    speakNow(
      "Camera running. Double tap anywhere to hear what is in front of you. Press Read Text to read printed text."
    );
  } catch (e) {
    alert("Camera error: " + e.name + " — " + e.message);
    console.log(e);
  }
};

// Send frames for detection (no auto speaking)
async function sendFrame() {
  if (pauseInfer) return;
  if (!video.srcObject) return;

  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;

  // small for speed
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

  if (img && data.annotated_jpg) {
    img.src = "data:image/jpeg;base64," + data.annotated_jpg;
  }

  // store latest summary for on-demand speech
  if (data.speech) latestText = data.speech;
}

// Read Text (OCR) - pauses detection while running
readTextBtn.onclick = async () => {
  if (ocrBusy) return;
  if (!video.srcObject) return alert("Start camera first");

  ocrBusy = true;
  pauseInfer = true;

  const oldText = readTextBtn.textContent;
  readTextBtn.textContent = "Reading…";
  readTextBtn.disabled = true;

  try {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;

    // moderate size for OCR
    const targetW = 960;
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
    speakNow(text || "I could not find readable text. Try moving closer.");
  } catch (e) {
    alert("Read Text error: " + e.message);
  } finally {
    readTextBtn.textContent = oldText;
    readTextBtn.disabled = false;
    ocrBusy = false;
    pauseInfer = false;
  }
};

// ~2 FPS inference
setInterval(sendFrame, 500);
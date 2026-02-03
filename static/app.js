let speechEnabled = false;
let lastSpoken = "";

document.getElementById("enable").onclick = () => {
  speechEnabled = true;
  document.getElementById("enable").textContent = "Speech enabled ✅";
};

function speak(text) {
  if (!speechEnabled) return;
  if (!text || text === lastSpoken) return;
  lastSpoken = text;

  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.0;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);

  document.getElementById("lastSpeech").textContent = text;
}

async function poll() {
  const res = await fetch("/detections");
  const data = await res.json();
  if (data.speech) {
  latestText = data.speech;
  speak(data.speech);
}

  document.getElementById("json").textContent = JSON.stringify(data, null, 2);

  if (data.speech) speak(data.speech);
}

let latestText = "";

document.getElementById("speakNow").onclick = () => {
  if (latestText) speak(latestText);
};

setInterval(poll, 800);
poll();

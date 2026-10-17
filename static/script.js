const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const result = document.getElementById("result");
const ctx = canvas.getContext("2d");

canvas.width = 640;
canvas.height = 480;

let lastBox = null;
let lastEmotion = "--";
let lastConfidence = 0;

let busy = false;
const POLL_INTERVAL = 1200; 

navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } })
    .then(stream => {
        video.srcObject = stream;
        return video.play();
    })
    .catch(err => {
        console.error("Camera error:", err);
        result.innerText = "Camera error: " + err.message;
    });

// --- Draw loop: paint video + bounding box every animation frame ---
function drawVideo() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    if (lastBox) {
        const x = lastBox[0] * canvas.width;
        const y = lastBox[1] * canvas.height;
        const w = lastBox[2] * canvas.width;
        const h = lastBox[3] * canvas.height;
        ctx.strokeStyle = "lime";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
    }
    requestAnimationFrame(drawVideo);
}
requestAnimationFrame(drawVideo);

function sendFrame() {
    if (busy) return;                 
    if (video.readyState < 2) {       
        scheduleNext();
        return;
    }
    busy = true;

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 320;
    tempCanvas.height = 240;
    const tctx = tempCanvas.getContext("2d");
    tctx.drawImage(video, 0, 0, 320, 240);
    const imageData = tempCanvas.toDataURL("image/jpeg", 0.7);

    // Show a subtle "processing..." hint so the user knows it's alive
    result.innerText = `Emotion: ${lastEmotion} (${lastConfidence}%)  ...`;

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData })
    })
    .then(res => {
        if (!res.ok) throw new Error("Server returned " + res.status);
        return res.json();
    })
    .then(data => {
        lastEmotion = data.emotion || "--";
        lastConfidence = data.confidence || 0;
        lastBox = data.box || null;
        result.innerText = `Emotion: ${lastEmotion} (${lastConfidence}%)`;
    })
    .catch(err => {
        console.error("Predict failed:", err);
        result.innerText = `Emotion: ${lastEmotion} (retrying...)`;
    })
    .finally(() => {
        busy = false;
        scheduleNext();
    });
}

function scheduleNext() {
    setTimeout(sendFrame, POLL_INTERVAL);
}


video.addEventListener("playing", scheduleNext, { once: true });
setTimeout(scheduleNext, 2000);

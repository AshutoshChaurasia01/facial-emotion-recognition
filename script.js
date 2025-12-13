const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const result = document.getElementById("result");
const ctx = canvas.getContext("2d");

canvas.width = 640;
canvas.height = 480;

let lastBox = null;
let lastEmotion = "--";
let lastConfidence = 0;


navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
})
.catch(err => alert("Camera error: " + err));


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


setInterval(() => {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 320;   
    tempCanvas.height = 240;
    const tctx = tempCanvas.getContext("2d");
    tctx.drawImage(video, 0, 0, 320, 240);

    const imageData = tempCanvas.toDataURL("image/jpeg", 0.7);

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData })
    })
    .then(res => res.json())
    .then(data => {
        lastEmotion = data.emotion;
        lastConfidence = data.confidence;
        lastBox = data.box;

        result.innerText =
          `Emotion: ${lastEmotion} (${lastConfidence}%)`;
    })
    .catch(err => console.error(err));

}, 2500); 


const byId = (id) => document.getElementById(id);
const confidence = byId('confidence');
confidence.addEventListener('input', () => byId('confidence-value').textContent = confidence.value);

async function loadStatus() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('Status unavailable');
    const status = await response.json();
    byId('model-status').textContent = status.model_ready ? 'Ready' : 'Weights missing';
    byId('model-weights').textContent = status.weights;
    byId('telegram-status').textContent = status.telegram_configured ? 'Connected' : 'Not configured';
  } catch {
    byId('model-status').textContent = 'Dashboard unavailable';
    byId('telegram-status').textContent = 'Unknown';
  }
}

async function loadAlerts() {
  const target = byId('alerts');
  target.innerHTML = '<p class="empty">Loading alerts…</p>';
  let events;
  try {
    const response = await fetch('/api/alerts');
    if (!response.ok) throw new Error('Alerts unavailable');
    events = await response.json();
  } catch {
    target.innerHTML = '<p class="empty">Could not load local alerts. Check that the dashboard is running.</p>';
    return;
  }
  if (!events.length) { target.innerHTML = '<p class="empty">No local alerts have been recorded yet.</p>'; return; }
  target.innerHTML = '';
  const template = byId('alert-template');
  events.forEach(event => {
    const node = template.content.cloneNode(true);
    const labels = (event.detections || []).map(d => `${d.label} ${Math.round(d.confidence * 100)}%`).join(', ') || 'Suspected weapon';
    const camera = event.camera || {};
    const img = node.querySelector('img'); img.src = event.evidence_url || ''; img.hidden = !event.evidence_url;
    node.querySelector('h3').textContent = labels;
    node.querySelector('.alert-time').textContent = new Date(event.detected_at_utc).toLocaleString();
    node.querySelector('.alert-location').textContent = camera.location || 'Location unavailable';
    const telegram = event.delivery?.telegram;
    const notification = telegram
      ? `Telegram: ${telegram.status}${telegram.attempts ? ` (${telegram.attempts} attempt${telegram.attempts === 1 ? '' : 's'})` : ''}`
      : 'Telegram: not configured';
    node.querySelector('.alert-meta').textContent = `Camera: ${camera.camera_id || '—'} · ${notification}`;
    target.appendChild(node);
  });
}

byId('refresh-alerts').addEventListener('click', loadAlerts);
byId('image-input').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) byId('upload-zone').querySelector('b').textContent = file.name;
});
byId('image-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = byId('test-message'); message.textContent = 'Analyzing image with the local model…';
  const button = event.currentTarget.querySelector('button'); button.disabled = true;
  try {
    const response = await fetch('/api/detect-image', { method: 'POST', body: new FormData(event.currentTarget) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Detection failed.');
    const result = byId('test-result'); result.classList.remove('hidden');
    byId('annotated-image').src = `${data.image_url}?v=${Date.now()}`;
    const list = byId('detection-list'); list.innerHTML = '';
    if (data.detections.length) {
      byId('result-title').textContent = `${data.detections.length} object(s) detected`;
      data.detections.forEach(item => { const row = document.createElement('li'); row.textContent = `${item.label} — ${Math.round(item.confidence * 100)}% confidence`; list.appendChild(row); });
    } else { byId('result-title').textContent = 'No weapon detected'; }
    message.textContent = 'Analysis complete. Review the image before taking any action.';
  } catch (error) { message.textContent = error.message; }
  finally { button.disabled = false; }
});

byId('video-input').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) byId('video-zone').querySelector('b').textContent = file.name;
});
byId('video-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = byId('video-message');
  const button = event.currentTarget.querySelector('button');
  message.textContent = 'Analyzing video locally. This can take a few minutes for longer videos…';
  button.disabled = true;
  try {
    const data = new FormData(event.currentTarget);
    data.append('confidence', confidence.value);
    const response = await fetch('/api/detect-video', { method: 'POST', body: data });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Video detection failed.');
    byId('annotated-video').src = `${result.video_url}?v=${Date.now()}`;
    byId('video-result').classList.remove('hidden');
    const summary = Object.entries(result.detections || {}).map(([label, count]) => `${label}: ${count}`).join(', ');
    byId('video-summary').textContent = `${result.frames} frame(s) processed.${summary ? ` Detections — ${summary}.` : ' No weapons detected.'}`;
    message.textContent = 'Video analysis complete. Review the annotated video before acting.';
  } catch (error) { message.textContent = error.message; }
  finally { button.disabled = false; }
});

let webcamStream;
let webcamTimer;
let webcamBusy = false;
const webcamVideo = byId('webcam-video');
const webcamCanvas = document.createElement('canvas');

function stopWebcam() {
  clearInterval(webcamTimer);
  webcamTimer = undefined;
  webcamStream?.getTracks().forEach(track => track.stop());
  webcamStream = undefined;
  webcamVideo.srcObject = null;
  byId('webcam-view').classList.add('hidden');
  byId('webcam-toggle').textContent = 'Start live webcam detection';
  byId('webcam-status').textContent = 'Camera is off';
  byId('webcam-message').textContent = '';
}

async function detectWebcamFrame() {
  if (webcamBusy || !webcamStream || !webcamVideo.videoWidth) return;
  webcamBusy = true;
  const width = Math.min(webcamVideo.videoWidth, 640);
  webcamCanvas.width = width;
  webcamCanvas.height = Math.round(webcamVideo.videoHeight * (width / webcamVideo.videoWidth));
  webcamCanvas.getContext('2d').drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
  try {
    const blob = await new Promise(resolve => webcamCanvas.toBlob(resolve, 'image/jpeg', 0.82));
    if (!blob) throw new Error('Could not capture a webcam frame.');
    const data = new FormData();
    data.append('frame', blob, 'webcam.jpg');
    data.append('confidence', confidence.value);
    const response = await fetch('/api/detect-frame', { method: 'POST', body: data });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Live detection failed.');
    byId('webcam-result').src = `data:image/jpeg;base64,${result.annotated_image}`;
    const count = result.detections.length;
    byId('webcam-message').textContent = result.alert_created
      ? 'High-confidence alert recorded and sent to Telegram. Review the evidence.'
      : count ? `${count} object(s) detected — verify the video and context.` : 'No weapon detected in the current frame.';
  } catch (error) {
    stopWebcam();
    byId('webcam-message').textContent = error.message;
  } finally { webcamBusy = false; }
}

byId('webcam-toggle').addEventListener('click', async () => {
  if (webcamStream) { stopWebcam(); return; }
  if (!navigator.mediaDevices?.getUserMedia) {
    byId('webcam-message').textContent = 'This browser does not support webcam access.';
    return;
  }
  try {
    byId('webcam-status').textContent = 'Requesting camera access…';
    const videoConstraints = {
      video: {
        facingMode: { ideal: 'user' },
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    };
    webcamStream = await navigator.mediaDevices.getUserMedia(videoConstraints).catch(async () => {
      return navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    });
    webcamVideo.srcObject = webcamStream;
    await webcamVideo.play();
    byId('webcam-view').classList.remove('hidden');
    byId('webcam-toggle').textContent = 'Stop live webcam detection';
    byId('webcam-status').textContent = 'Live detection active';
    byId('webcam-message').textContent = 'Camera connected. Waiting for the first analysis…';
    webcamTimer = setInterval(detectWebcamFrame, 100);
    detectWebcamFrame();
  } catch (error) {
    stopWebcam();
    const message = error?.name === 'NotAllowedError' || /permission|blocked/i.test(error?.message || '')
      ? 'Camera permission was blocked. Allow camera access in this browser and refresh the page.'
      : /in use|could not start|hardware|source/i.test(error?.message || '')
        ? 'Camera was locked by a background Python process. The process has been released — please click "Start live webcam detection" again.'
        : `Could not access the webcam: ${error.message || 'Unknown error'}`;
    byId('webcam-message').textContent = message;
  }
});
byId('mark-negative')?.addEventListener('click', async () => {
  if (!webcamStream || !webcamVideo || webcamVideo.paused || webcamVideo.ended) {
    byId('webcam-message').textContent = 'Start live webcam detection first before marking negative frames.';
    return;
  }
  const canvas = document.createElement('canvas');
  canvas.width = webcamVideo.videoWidth || 640;
  canvas.height = webcamVideo.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(async (blob) => {
    if (!blob) return;
    const formData = new FormData();
    formData.append('frame', blob, 'webcam_negative.jpg');
    try {
      const resp = await fetch('/api/add-negative-frame', { method: 'POST', body: formData });
      const data = await resp.json();
      if (resp.ok && data.success) {
        byId('webcam-message').textContent = `Saved hard negative frame (${data.filename}) to training dataset! Move water bottle slightly and click again.`;
      } else {
        byId('webcam-message').textContent = `Failed to save negative frame: ${data.error || 'Unknown error'}`;
      }
    } catch (err) {
      byId('webcam-message').textContent = `Error saving negative frame: ${err.message}`;
    }
  }, 'image/jpeg', 0.9);
});

window.addEventListener('beforeunload', stopWebcam);
loadStatus(); loadAlerts();

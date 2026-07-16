(() => {
  const video = document.getElementById('video');
  const overlay = document.getElementById('overlay');
  const ctx = overlay.getContext('2d');
  const stats = document.getElementById('stats');
  const videoInput = document.getElementById('videoInput');
  const overlayInput = document.getElementById('overlayInput');
  const playPause = document.getElementById('playPause');

  let frames = new Map();

  function colorForId(idStr) {
    let h = 0;
    for (let i = 0; i < idStr.length; i++) {
      h = (h * 31 + idStr.charCodeAt(i)) & 0xffffff;
    }
    const r = (h >> 16) & 0xff;
    const g = (h >> 8) & 0xff;
    const b = h & 0xff;
    return `rgb(${r},${g},${b})`;
  }

  async function loadOverlay(file) {
    const text = await file.text();
    frames.clear();
    for (const line of text.trim().split(/\r?\n/)) {
      if (!line) continue;
      const record = JSON.parse(line);
      frames.set(record.frame_index, record.faces || []);
    }
    stats.textContent = `Loaded overlay for ${frames.size} frames.`;
  }

  function resizeCanvas() {
    const rect = video.getBoundingClientRect();
    overlay.width = rect.width;
    overlay.height = rect.height;
  }

  function draw() {
    if (!video.videoWidth) {
      requestAnimationFrame(draw);
      return;
    }
    resizeCanvas();
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const scaleX = overlay.width / video.videoWidth;
    const scaleY = overlay.height / video.videoHeight;
    const frameIndex = Math.floor(video.currentTime * (video.videoFrameRate || 30));
    const faces = frames.get(frameIndex) || [];

    for (const face of faces) {
      const [x1, y1, x2, y2] = face.bbox_xyxy;
      const color = colorForId(face.canonical_track_id || face.raw_tracklet_id);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
      ctx.fillStyle = color;
      ctx.font = '12px sans-serif';
      const label = `${face.raw_tracklet_id} | ${face.canonical_track_id} | ${face.display_label || 'unresolved'}`;
      ctx.fillText(label, x1 * scaleX, Math.max(y1 * scaleY - 4, 12));
    }
    requestAnimationFrame(draw);
  }

  videoInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      video.src = URL.createObjectURL(file);
      video.load();
    }
  });

  overlayInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) loadOverlay(file);
  });

  playPause.addEventListener('click', () => {
    if (video.paused) video.play(); else video.pause();
  });

  video.addEventListener('loadedmetadata', resizeCanvas);
  video.addEventListener('play', draw);
  draw();
})();

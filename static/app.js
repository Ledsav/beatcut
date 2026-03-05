/* ── State ──────────────────────────────────────────── */
let currentJobId = null;
let eventSource  = null;

/* ── DOM refs ───────────────────────────────────────── */
const stageUpload     = document.getElementById('stage-upload');
const stageProcessing = document.getElementById('stage-processing');
const stageDone       = document.getElementById('stage-done');
const stageError      = document.getElementById('stage-error');

const inputPhotos  = document.getElementById('input-photos');
const inputMusic   = document.getElementById('input-music');
const photosStatus = document.getElementById('photos-status');
const musicStatus  = document.getElementById('music-status');
const btnProcess   = document.getElementById('btn-process');
const btnCancel    = document.getElementById('btn-cancel');
const btnAgain     = document.getElementById('btn-again');
const btnRetry     = document.getElementById('btn-retry');
const btnDownload  = document.getElementById('btn-download');

const progressBar   = document.getElementById('progress-bar');
const progressLabel = document.getElementById('progress-label');
const logBox        = document.getElementById('log-box');
const errorMsg      = document.getElementById('error-msg');

const rangeSensitivity = document.getElementById('range-sensitivity');
const valSensitivity   = document.getElementById('val-sensitivity');
const rangeZoom        = document.getElementById('range-zoom');
const valZoom          = document.getElementById('val-zoom');
const selTransition        = document.getElementById('sel-transition');
const polaroidEntryItem    = document.getElementById('polaroid-entry-item');
const selPolaroidEntry     = document.getElementById('sel-polaroid-entry');
const selSyncMode          = document.getElementById('sel-sync-mode');

const statPhotos = document.getElementById('stat-photos');
const statBeats  = document.getElementById('stat-beats');
const statBpm    = document.getElementById('stat-bpm');
const statDur    = document.getElementById('stat-dur');

/* ── Stage helpers ──────────────────────────────────── */
function showStage(stage) {
  [stageUpload, stageProcessing, stageDone, stageError].forEach(s => s.classList.remove('active'));
  stage.classList.add('active');
}

/* ── File inputs ─────────────────────────────────────── */
function updateProcessButton() {
  const hasPhotos = inputPhotos.files && inputPhotos.files.length > 0;
  const hasMusic  = inputMusic.files  && inputMusic.files.length  > 0;
  btnProcess.disabled = !(hasPhotos && hasMusic);
}

inputPhotos.addEventListener('change', () => {
  const n = inputPhotos.files.length;
  photosStatus.textContent = n > 0 ? `${n} photo${n > 1 ? 's' : ''} selected` : 'No photos selected';
  updateProcessButton();
});

inputMusic.addEventListener('change', () => {
  const f = inputMusic.files[0];
  musicStatus.textContent = f ? f.name : 'No file selected';
  if (f) loadWaveform(f);
  updateProcessButton();
});

/* ── Drag-and-drop for upload zones ─────────────────── */
function setupDrop(zone, input) {
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const dt = e.dataTransfer;
    if (dt.files.length) {
      try {
        input.files = dt.files;
        input.dispatchEvent(new Event('change'));
      } catch (_) {
        console.warn('Could not assign dropped files to input');
      }
    }
  });
  zone.addEventListener('click', e => {
    // The label's `for` attribute already triggers the input on label clicks;
    // only call input.click() for clicks outside the label to avoid double dialog.
    const label = zone.querySelector('label');
    if (label && label.contains(e.target)) return;
    if (e.target !== input) input.click();
  });
}

setupDrop(document.getElementById('zone-photos'), inputPhotos);
setupDrop(document.getElementById('zone-music'),  inputMusic);

/* ── Polaroid entry visibility ──────────────────────── */
function updatePolaroidVisibility() {
  polaroidEntryItem.style.display = selTransition.value === 'polaroid' ? '' : 'none';
}
selTransition.addEventListener('change', updatePolaroidVisibility);

/* ── Live range value display ───────────────────────── */
rangeSensitivity.addEventListener('input', () => {
  valSensitivity.textContent = parseFloat(rangeSensitivity.value).toFixed(1);
});
rangeZoom.addEventListener('input', () => {
  valZoom.textContent = parseFloat(rangeZoom.value).toFixed(2);
});

/* ── Process ─────────────────────────────────────────── */
btnProcess.addEventListener('click', async () => {
  const fd = new FormData();
  Array.from(inputPhotos.files).forEach(f => fd.append('images', f));
  fd.append('music',       inputMusic.files[0]);
  fd.append('resolution',     document.getElementById('sel-resolution').value);
  fd.append('transition',     selTransition.value);
  fd.append('polaroid_entry', selPolaroidEntry.value);
  fd.append('sync_mode',   selSyncMode.value);
  fd.append('sensitivity', rangeSensitivity.value);
  fd.append('zoom',        rangeZoom.value);
  fd.append('fps',         '24');
  // Section selector: 0/0 = full song sentinel
  if (songDuration > 0) {
    const sT = reelStartPct < 0.005 ? 0 : reelStartPct * songDuration;
    const eT = reelEndPct   > 0.995 ? 0 : reelEndPct   * songDuration;
    fd.append('start_time', sT.toFixed(2));
    fd.append('end_time',   eT.toFixed(2));
  } else {
    fd.append('start_time', '0');
    fd.append('end_time',   '0');
  }

  showStage(stageProcessing);
  logBox.innerHTML = '';
  progressBar.style.width = '0%';
  progressLabel.textContent = 'Starting...';

  let res;
  try {
    res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Upload failed');
    currentJobId = data.job_id;
  } catch (err) {
    showError(err.message);
    return;
  }

  startSSE(currentJobId);
});

/* ── SSE ─────────────────────────────────────────────── */
function startSSE(jobId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/progress/${jobId}`);

  eventSource.onmessage = (e) => {
    const data = JSON.parse(e.data);

    if (data.error) {
      eventSource.close();
      showError(data.error);
      return;
    }

    if (data.msg === 'cancelled') {
      eventSource.close();
      resetToUpload();
      return;
    }

    progressBar.style.width = data.pct + '%';
    progressLabel.textContent = data.msg;
    appendLog(data.msg);

    if (data.pct >= 100 && data.msg === 'done' && data.file) {
      eventSource.close();
      showDone(data);
    }
  };

  eventSource.onerror = () => {
    eventSource.close();
    if (stageProcessing.classList.contains('active')) {
      showError('Lost connection to server. Check the terminal for errors.');
    }
  };
}

function appendLog(msg) {
  const line = document.createElement('span');
  line.className = 'log-line';
  line.textContent = `› ${msg}`;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

/* ── Cancel ─────────────────────────────────────────── */
btnCancel.addEventListener('click', async () => {
  if (!currentJobId) return;
  await fetch(`/cancel/${currentJobId}`, { method: 'POST' });
  if (eventSource) eventSource.close();
  resetToUpload();
});

/* ── Done ────────────────────────────────────────────── */
function showDone(data) {
  const stats = data.stats || {};
  statPhotos.textContent = stats.photos_used ?? '—';
  statBeats.textContent  = stats.beats       ?? '—';
  statBpm.textContent    = stats.bpm ? Math.round(stats.bpm) : '—';
  statDur.textContent    = stats.duration ? `${stats.duration.toFixed(1)}s` : '—';
  btnDownload.href = data.file;
  showStage(stageDone);
}

btnAgain.addEventListener('click', resetToUpload);

/* ── Error ───────────────────────────────────────────── */
function showError(msg) {
  errorMsg.textContent = msg;
  showStage(stageError);
}

btnRetry.addEventListener('click', resetToUpload);

/* ── Reel section selector ───────────────────────────── */
let songDuration  = 0;   // seconds decoded from audio file
let reelStartPct  = 0;   // 0.0 – 1.0
let reelEndPct    = 1;
let waveformPeaks = null; // Float32Array of normalised peak amplitudes
let reelPlayPct   = null; // null = not playing, 0.0–1.0 = playhead position
let audioObjectURL = null;
let reelAudio      = null; // HTMLAudioElement used for preview playback
let reelPlayRAF    = null; // requestAnimationFrame handle

const btnReelPlay   = document.getElementById('btn-reel-play');
const iconPlay      = document.getElementById('icon-play');
const iconPause     = document.getElementById('icon-pause');
const reelWrap      = document.getElementById('reel-wrap');
const waveformCanvas= document.getElementById('waveform-canvas');
const reelSelection = document.getElementById('reel-selection');
const handleStart   = document.getElementById('handle-start');
const handleEnd     = document.getElementById('handle-end');
const reelTimeStart = document.getElementById('reel-time-start');
const reelTimeEnd   = document.getElementById('reel-time-end');
const reelHint      = document.getElementById('reel-hint');
const valSection    = document.getElementById('val-section');

// Decorative placeholder bars (shown before audio loads)
const PLACEHOLDER_PEAKS = Array.from({ length: 120 }, (_, i) => {
  const t = i / 120;
  return 0.18 + 0.55 * Math.abs(Math.sin(t * 18.4 + 1.1) * Math.cos(t * 7.3 + 0.5));
});

function fmtTime(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function drawWaveformCanvas() {
  if (!waveformCanvas) return;
  const dpr  = window.devicePixelRatio || 1;
  const rect = waveformCanvas.getBoundingClientRect();
  if (!rect.width) return;

  const W = rect.width;
  const H = rect.height;
  waveformCanvas.width  = Math.round(W * dpr);
  waveformCanvas.height = Math.round(H * dpr);

  const ctx  = waveformCanvas.getContext('2d');
  const peaks = waveformPeaks || PLACEHOLDER_PEAKS;
  const N    = peaks.length;
  const gap  = 1.5;
  const barW = Math.max(1, (W * dpr - gap * (N - 1)) / N);

  for (let i = 0; i < N; i++) {
    const iPct   = i / N;
    const inSel  = waveformPeaks && iPct >= reelStartPct && iPct < reelEndPct;
    const peak   = peaks[i];
    const barH   = Math.max(2, peak * H * dpr * 0.82);
    const x      = i * (barW + gap);
    const y      = (H * dpr - barH) / 2;

    if (waveformPeaks) {
      // Dim bars that the playhead has already passed
      const played = reelPlayPct !== null && iPct < reelPlayPct;
      ctx.fillStyle = inSel
        ? played
          ? `rgba(200,255,0,${0.18 + peak * 0.2})`
          : `rgba(200,255,0,${0.45 + peak * 0.5})`
        : `rgba(107,107,128,${0.12 + peak * 0.22})`;
    } else {
      ctx.fillStyle = `rgba(107,107,128,${0.07 + peak * 0.1})`;
    }
    ctx.fillRect(x, y, barW, barH);
  }

  // Draw playhead line
  if (reelPlayPct !== null) {
    const px = reelPlayPct * W * dpr;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth   = Math.round(1.5 * dpr);
    ctx.shadowColor = 'rgba(200,255,0,0.6)';
    ctx.shadowBlur  = 6 * dpr;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, H * dpr);
    ctx.stroke();
    ctx.restore();
  }
}

function updateReelDisplay() {
  reelSelection.style.left  = (reelStartPct * 100) + '%';
  reelSelection.style.width = ((reelEndPct - reelStartPct) * 100) + '%';

  if (songDuration > 0) {
    const startSec = reelStartPct * songDuration;
    const endSec   = reelEndPct   * songDuration;
    const durSec   = endSec - startSec;
    const fullSong = reelStartPct < 0.005 && reelEndPct > 0.995;

    reelTimeStart.innerHTML = `<span class="t">${fmtTime(startSec)}</span>`;
    reelTimeEnd.innerHTML   =
      `<span class="t">${fmtTime(endSec)}</span>` +
      `<span style="color:var(--muted);margin-left:6px">· ${fmtTime(durSec)}</span>`;
    valSection.textContent  = fullSong
      ? 'Full song'
      : `${fmtTime(startSec)} → ${fmtTime(endSec)}`;
  }

  drawWaveformCanvas();
}

async function loadWaveform(file) {
  reelHint.textContent = 'Analysing audio…';
  // Revoke previous blob URL
  if (audioObjectURL) { URL.revokeObjectURL(audioObjectURL); audioObjectURL = null; }
  audioObjectURL = URL.createObjectURL(file);
  try {
    const buf      = await file.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded  = await audioCtx.decodeAudioData(buf);
    songDuration   = decoded.duration;
    btnReelPlay.disabled = false;

    const ch        = decoded.getChannelData(0);
    const N         = 120;
    const blockSize = Math.floor(ch.length / N);
    const raw       = new Float32Array(N);
    let   maxP      = 0;
    for (let i = 0; i < N; i++) {
      let m = 0, off = i * blockSize;
      for (let j = 0; j < blockSize; j++) { const v = Math.abs(ch[off + j]); if (v > m) m = v; }
      raw[i] = m;
      if (m > maxP) maxP = m;
    }
    waveformPeaks = Array.from(raw).map(p => maxP > 0 ? p / maxP : 0.5);

    reelWrap.classList.remove('reel-disabled');
    reelStartPct = 0;
    reelEndPct   = 1;
    updateReelDisplay();

    reelTimeStart.innerHTML = `<span class="t">0:00</span>`;
    reelTimeEnd.innerHTML   =
      `<span class="t">${fmtTime(songDuration)}</span>` +
      `<span style="color:var(--muted);margin-left:6px">· ${fmtTime(songDuration)}</span>`;
    valSection.textContent  = 'Full song';
    reelHint.textContent    = `${fmtTime(songDuration)} total · drag handles to trim`;

    await audioCtx.close();
  } catch (err) {
    console.warn('Waveform decode failed:', err);
    // Fallback: enable reel without real waveform
    songDuration  = 0;
    waveformPeaks = Array.from(PLACEHOLDER_PEAKS);
    reelWrap.classList.remove('reel-disabled');
    btnReelPlay.disabled = false;
    reelHint.textContent = 'Drag handles to trim';
    drawWaveformCanvas();
  }
}

/* ── Reel play/pause ─────────────────────────────────── */
function stopReelPlayback() {
  if (reelAudio) { reelAudio.pause(); reelAudio = null; }
  if (reelPlayRAF) { cancelAnimationFrame(reelPlayRAF); reelPlayRAF = null; }
  reelPlayPct = null;
  iconPlay.style.display  = '';
  iconPause.style.display = 'none';
  drawWaveformCanvas();
}

btnReelPlay.addEventListener('click', () => {
  // If already playing, stop
  if (reelAudio && !reelAudio.paused) { stopReelPlayback(); return; }

  if (!audioObjectURL) return;

  const startSec = reelStartPct * songDuration;
  const endSec   = reelEndPct   * songDuration;

  reelAudio = new Audio(audioObjectURL);
  reelAudio.currentTime = startSec;

  iconPlay.style.display  = 'none';
  iconPause.style.display = '';

  reelAudio.play().catch(err => { console.warn('Audio play error:', err); stopReelPlayback(); });

  function tick() {
    if (!reelAudio) return;
    const t = reelAudio.currentTime;
    if (t >= endSec) { stopReelPlayback(); return; }
    reelPlayPct = t / songDuration;
    drawWaveformCanvas();
    reelPlayRAF = requestAnimationFrame(tick);
  }
  reelPlayRAF = requestAnimationFrame(tick);

  reelAudio.onended = stopReelPlayback;
});

function setupHandleDrag(handle, isStart) {
  let active = false;

  function onStart(e) { active = true; handle.classList.add('dragging'); e.preventDefault(); }
  function onStop()   { active = false; handle.classList.remove('dragging'); }
  function onMove(clientX) {
    if (!active) return;
    const rect = reelWrap.getBoundingClientRect();
    let pct = (clientX - rect.left) / rect.width;
    pct = Math.max(0, Math.min(1, pct));
    if (isStart) {
      reelStartPct = Math.min(pct, reelEndPct - 0.02);
    } else {
      reelEndPct   = Math.max(pct, reelStartPct + 0.02);
    }
    updateReelDisplay();
  }

  handle.addEventListener('mousedown', onStart);
  document.addEventListener('mouseup',   onStop);
  document.addEventListener('mousemove', e => onMove(e.clientX));

  handle.addEventListener('touchstart', onStart, { passive: false });
  document.addEventListener('touchend',  onStop);
  document.addEventListener('touchmove', e => { if (active) { e.preventDefault(); onMove(e.touches[0].clientX); } }, { passive: false });
}

setupHandleDrag(handleStart, true);
setupHandleDrag(handleEnd,   false);

// Redraw canvas on window resize
window.addEventListener('resize', drawWaveformCanvas);

// Draw placeholder immediately
drawWaveformCanvas();

function resetReel() {
  stopReelPlayback();
  if (audioObjectURL) { URL.revokeObjectURL(audioObjectURL); audioObjectURL = null; }
  songDuration  = 0;
  reelStartPct  = 0;
  reelEndPct    = 1;
  waveformPeaks = null;
  reelWrap.classList.add('reel-disabled');
  btnReelPlay.disabled      = true;
  reelSelection.style.left  = '0%';
  reelSelection.style.width = '100%';
  reelTimeStart.innerHTML   = '—';
  reelTimeEnd.innerHTML     = '—';
  valSection.textContent    = 'Full song';
  reelHint.textContent      = 'Load a song to select a section';
  drawWaveformCanvas();
}

/* ── Reset ───────────────────────────────────────────── */
function resetToUpload() {
  currentJobId = null;
  if (eventSource) { eventSource.close(); eventSource = null; }

  try {
    const dt = new DataTransfer();
    inputPhotos.files = dt.files;
    inputMusic.files  = dt.files;
  } catch (_) {}

  photosStatus.textContent = 'No photos selected';
  musicStatus.textContent  = 'No file selected';
  btnProcess.disabled      = true;

  resetReel();
  selTransition.value         = 'flash';
  selPolaroidEntry.value      = 'drop';
  selSyncMode.value           = 'beat';
  updatePolaroidVisibility();

  progressBar.style.width     = '0%';
  progressLabel.textContent   = 'Starting...';
  logBox.innerHTML            = '';
  errorMsg.textContent        = '';
  btnDownload.href            = '#';

  showStage(stageUpload);
}

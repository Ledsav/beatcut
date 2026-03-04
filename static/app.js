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
const rangeDuration        = document.getElementById('range-duration');
const valDuration          = document.getElementById('val-duration');
const selTransition        = document.getElementById('sel-transition');
const polaroidEntryItem    = document.getElementById('polaroid-entry-item');
const selPolaroidEntry     = document.getElementById('sel-polaroid-entry');

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
rangeDuration.addEventListener('input', () => {
  const v = parseInt(rangeDuration.value);
  valDuration.textContent = v >= 600 ? 'Full song' : `${v}s`;
});

/* ── Process ─────────────────────────────────────────── */
btnProcess.addEventListener('click', async () => {
  const fd = new FormData();
  Array.from(inputPhotos.files).forEach(f => fd.append('images', f));
  fd.append('music',       inputMusic.files[0]);
  fd.append('resolution',     document.getElementById('sel-resolution').value);
  fd.append('transition',     selTransition.value);
  fd.append('polaroid_entry', selPolaroidEntry.value);
  fd.append('sensitivity', rangeSensitivity.value);
  fd.append('zoom',         rangeZoom.value);
  fd.append('fps',          '24');
  // 600 = "full song" sentinel — server will just use full beat list
  const durVal = parseInt(rangeDuration.value);
  fd.append('max_duration', durVal >= 600 ? '0' : String(durVal));

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

  rangeDuration.value         = '600';
  valDuration.textContent     = 'Full song';
  selTransition.value         = 'flash';
  selPolaroidEntry.value      = 'drop';
  updatePolaroidVisibility();

  progressBar.style.width     = '0%';
  progressLabel.textContent   = 'Starting...';
  logBox.innerHTML            = '';
  errorMsg.textContent        = '';
  btnDownload.href            = '#';

  showStage(stageUpload);
}

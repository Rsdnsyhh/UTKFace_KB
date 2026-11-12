'use strict';

/* ==========================================================================
   Pengaturan. Ubah nilai di sini bila model diganti atau dilatih ulang.
   ========================================================================== */
const CONFIG = {
  MODEL_URL: 'model/model.json',
  INFO_URL: 'model/model_info.json',   // dibuat oleh notebook (Bagian F)
  INPUT_SIZE: 128,          // ukuran sisi citra masukan model (piksel)
  MAE: 6.9,                 // nilai cadangan (tahun) bila model_info.json tidak ditemukan
  MAX_FILE_MB: 15,
  MIN_IMAGE_PX: 64,         // sisi terpendek foto yang masih diterima
  MAX_HISTORY: 8,
  UNSTABLE_DIFF: 4,         // selisih (tahun) prediksi asli vs cermin yang dianggap kurang stabil
  PREDICT_DELAY_MS: 250,    // jeda sebelum prediksi ulang saat kotak digeser
  SELFTEST_TOL: 0.05,       // toleransi relatif uji mandiri (minimal 1 tahun)
  GROUPS: [
    { name: 'Anak-anak', max: 12 },
    { name: 'Remaja', max: 19 },
    { name: 'Dewasa', max: 59 },
    { name: 'Lansia', max: Infinity },
  ],
};

/* ==========================================================================
   Utilitas dasar
   ========================================================================== */
const $ = (sel) => document.querySelector(sel);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Galat dengan pesan yang sudah ramah dibaca pengguna. */
class AppError extends Error {}

const els = {
  status: $('#status'), statusText: $('#status-text'), retryModel: $('#retry-model'),
  themeToggle: $('#theme-toggle'),
  alert: $('#alert'), alertText: $('#alert-text'), alertClose: $('#alert-close'),
  tabUpload: $('#tab-upload'), tabCamera: $('#tab-camera'),
  panelUpload: $('#panel-upload'), panelCamera: $('#panel-camera'),
  dropzone: $('#dropzone'), fileInput: $('#file-input'),
  video: $('#video'), cameraBox: document.querySelector('.camera-box'),
  camStart: $('#cam-start'), camShot: $('#cam-shot'), camStop: $('#cam-stop'),
  editor: $('#editor'), stage: $('#stage'), stageWrap: $('#stage-wrap'), zoom: $('#zoom'),
  btnCenter: $('#btn-center'), btnDetect: $('#btn-detect'), btnReset: $('#btn-reset'),
  stateEmpty: $('#state-empty'), stateResult: $('#state-result'),
  preview: $('#preview'), ageBlock: $('#age-block'), ageValue: $('#age-value'), ageGroup: $('#age-group'),
  scaleRange: $('#scale-range'), scaleMarker: $('#scale-marker'),
  factRange: $('#fact-range'), factConsistency: $('#fact-consistency'), factTime: $('#fact-time'),
  notes: $('#notes'), btnSave: $('#btn-save'),
  history: $('#history'), historyEmpty: $('#history-empty'), btnClear: $('#btn-clear'),
  toast: $('#toast'),
};

const state = {
  engine: null,            // mesin inferensi aktif: { name, label, predict(canvas) }
  info: null,              // isi model_info.json (MAE keseluruhan dan per kelompok usia)
  source: null,            // HTMLImageElement atau HTMLCanvasElement
  srcW: 0, srcH: 0,
  crop: { x: 0, y: 0, s: 0 },
  scale: 1,                // rasio piksel foto asli ke piksel kanvas (tampilan)
  drag: null,
  stream: null,
  timer: null,
  runId: 0,
  last: null,              // hasil prediksi terakhir
  shownAge: 0,
  history: [],
};

/* ==========================================================================
   Pesan galat dan notifikasi
   ========================================================================== */
function showError(message) {
  els.alertText.textContent = message;
  els.alert.hidden = false;
}
function clearError() { els.alert.hidden = true; }

let toastTimer = null;
function toast(message) {
  els.toast.textContent = message;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { els.toast.hidden = true; }, 2600);
}

/** Menampilkan galat apa pun dengan pesan yang tepat. */
function handleError(err, fallback) {
  console.error(err);
  showError(err instanceof AppError ? err.message : fallback);
}

window.addEventListener('unhandledrejection', (e) => {
  console.error(e.reason);
  showError('Terjadi kesalahan yang tidak terduga. Muat ulang halaman bila masalah berlanjut.');
  e.preventDefault();
});

/* ==========================================================================
   Tema tampilan
   ========================================================================== */
function applyTheme(theme) { document.documentElement.dataset.theme = theme; }

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem('theme'); } catch (_) { /* penyimpanan dinonaktifkan */ }
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(saved || (prefersDark ? 'dark' : 'light'));
  els.themeToggle.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try { localStorage.setItem('theme', next); } catch (_) { /* abaikan */ }
  });
}

/* ==========================================================================
   Pemuatan model
   ========================================================================== */
function setStatus(kind, text) {
  els.status.className = `pill pill-${kind}`;
  els.statusText.textContent = text;
  els.retryModel.hidden = kind !== 'error';
}

/** Mesin utama: TensorFlow.js. */
async function createTfEngine() {
  if (typeof tf === 'undefined') throw new Error('TensorFlow.js tidak tersedia.');
  const webglOk = await tf.setBackend('webgl').catch(() => false);
  if (!webglOk) await tf.setBackend('cpu');
  await tf.ready();

  const model = await tf.loadLayersModel(CONFIG.MODEL_URL);
  const S = CONFIG.INPUT_SIZE;
  const inShape = model.inputs[0].shape.slice(1).join('x');
  if (inShape !== `${S}x${S}x3`) {
    throw new AppError(`Ukuran masukan model (${inShape}) tidak sesuai dengan pengaturan aplikasi (${S}x${S}x3).`);
  }

  const run = (batch) => {
    const out = model.predict(batch);
    return out.data().then((v) => { out.dispose(); return v; });
  };

  return {
    name: 'tfjs',
    label: `TensorFlow.js, ${tf.getBackend()}`,
    /** Mengembalikan prediksi untuk gambar asli dan versi cerminnya. */
    async predict(canvas) {
      const batch = tf.tidy(() => {
        const x = tf.browser.fromPixels(canvas).toFloat().div(255).expandDims(0);
        return tf.concat([x, tf.reverse(x, [2])], 0);
      });
      try {
        const v = await run(batch);
        return [v[0], v[1]];
      } finally { batch.dispose(); }
    },
    /** Menjalankan citra uji deterministik (Float32Array HWC). */
    async predictRaw(image) {
      const batch = tf.tidy(() => tf.tensor4d(image, [1, S, S, 3]));
      try { return (await run(batch))[0]; } finally { batch.dispose(); }
    },
  };
}

/** Mesin cadangan: JavaScript murni (js/engine.js), tanpa TensorFlow.js. */
async function createJsEngine() {
  if (typeof JsEngine === 'undefined') throw new AppError('Berkas js/engine.js tidak dapat dimuat.');
  let json, buffer;
  try {
    const res = await fetch(CONFIG.MODEL_URL, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    json = await res.json();
    const binUrl = new URL(json.weightsManifest[0].paths[0], new URL(CONFIG.MODEL_URL, location.href));
    const binRes = await fetch(binUrl, { cache: 'no-cache' });
    if (!binRes.ok) throw new Error(`HTTP ${binRes.status}`);
    buffer = await binRes.arrayBuffer();
  } catch (err) {
    console.error(err);
    throw new AppError('Berkas model tidak dapat dimuat. Pastikan folder "model" berisi model.json dan ' +
      'group1-shard1of1.bin, lalu klik "Coba lagi".');
  }

  const S = CONFIG.INPUT_SIZE;
  const model = JsEngine.createModel(json, buffer);
  if (model.input.h !== S || model.input.w !== S || model.input.c !== 3) {
    throw new AppError(`Ukuran masukan model (${model.input.h}x${model.input.w}x${model.input.c}) tidak sesuai ` +
      `dengan pengaturan aplikasi (${S}x${S}x3).`);
  }
  const tick = () => new Promise((r) => setTimeout(r, 0));   // memberi napas pada antarmuka

  return {
    name: 'js',
    label: 'JavaScript murni',
    async predict(canvas) {
      const px = canvas.getContext('2d').getImageData(0, 0, S, S).data;
      const img = new Float32Array(S * S * 3);
      for (let i = 0, j = 0; i < px.length; i += 4) {
        img[j++] = px[i] / 255; img[j++] = px[i + 1] / 255; img[j++] = px[i + 2] / 255;
      }
      const a = await model.run(img, tick);
      const b = await model.run(JsEngine.flipHorizontal(img, S, S, 3), tick);
      return [a, b];
    },
    predictRaw(image) { return model.run(image, tick); },
  };
}

/** Membandingkan keluaran mesin dengan nilai acuan yang dihitung saat konversi model. */
async function selfTest(engine) {
  const ref = state.info && state.info.selftest && state.info.selftest.expected;
  const S = CONFIG.INPUT_SIZE;
  const image = (typeof JsEngine !== 'undefined') ? JsEngine.selftestImage(S) : null;
  if (!Number.isFinite(ref) || !image) return { checked: false, ok: true };
  const value = await engine.predictRaw(image);
  const tol = Math.max(1, Math.abs(ref) * CONFIG.SELFTEST_TOL);
  const ok = Number.isFinite(value) && Math.abs(value - ref) <= tol;
  if (!ok) console.warn(`Uji mandiri mesin '${engine.name}' gagal: hasil ${value}, acuan ${ref}.`);
  return { checked: true, ok };
}

async function loadModel() {
  setStatus('loading', 'Memuat model…');
  state.engine = null;
  try {
    if (location.protocol === 'file:') {
      throw new AppError('Halaman dibuka langsung dari berkas (file://), sehingga browser memblokir pemuatan model. ' +
        'Jalankan lewat Live Server di VS Code, lalu buka alamat yang diberikan (misalnya http://127.0.0.1:5500).');
    }
    await loadModelInfo();

    let engine = null, note = '';
    if (typeof tf === 'undefined') {
      note = 'TensorFlow.js tidak dapat dimuat (periksa koneksi internet). Aplikasi memakai mesin JavaScript murni.';
    } else {
      try {
        engine = await createTfEngine();
        const test = await selfTest(engine);
        if (!test.ok) {
          note = 'Hasil TensorFlow.js tidak lolos uji mandiri. Aplikasi memakai mesin JavaScript murni.';
          engine = null;
        }
      } catch (err) {
        console.warn('Mesin TensorFlow.js gagal:', err);
        if (err instanceof AppError) throw err;
        note = 'TensorFlow.js gagal memuat model. Aplikasi memakai mesin JavaScript murni.';
        engine = null;
      }
    }

    if (!engine) {
      engine = await createJsEngine();
      const test = await selfTest(engine);
      if (!test.ok) {
        throw new AppError('Berkas model tidak konsisten dengan nilai acuan hasil konversi. ' +
          'Ulangi konversi model dengan convert_model.py.');
      }
      if (note) toast(note);
    } else {
      // Pemanasan agar prediksi pertama tidak terasa lambat
      await engine.predictRaw(new Float32Array(CONFIG.INPUT_SIZE * CONFIG.INPUT_SIZE * 3));
    }

    state.engine = engine;
    setStatus('ready', `Model siap (${engine.label})`);
    if (state.source) schedulePredict(0);
  } catch (err) {
    setStatus('error', 'Model gagal dimuat');
    handleError(err, 'Model gagal dimuat karena kesalahan yang tidak diketahui.');
  }
}

/** Membaca ringkasan model. Bersifat opsional: bila gagal, nilai cadangan dipakai. */
async function loadModelInfo() {
  state.info = null;
  try {
    const res = await fetch(CONFIG.INFO_URL, { cache: 'no-store' });
    if (res.ok) {
      const data = await res.json();
      if (data && typeof data === 'object') {
        const groups = (Array.isArray(data.groups) ? data.groups : [])
          .filter((g) => Number.isFinite(g.max) && Number.isFinite(g.mae) && g.mae > 0)
          .sort((a, b) => a.max - b.max);
        state.info = { ...data, groups };
      }
    }
  } catch (err) {
    console.warn('model_info.json tidak dapat dibaca, memakai nilai cadangan.', err);
  }
  renderModelInfo();
}

function renderModelInfo() {
  const el = $('#model-info');
  if (!el) return;
  const i = state.info;
  if (!i || !Number.isFinite(i.mae)) {
    el.textContent = `Model: perkiraan selisih rata-rata sekitar ${CONFIG.MAE.toFixed(1)} tahun ` +
      '(nilai cadangan, berkas model_info.json tidak ditemukan).';
    return;
  }
  const parts = [`MAE uji ${i.mae.toFixed(2)} tahun`];
  if (Number.isFinite(i.rmse)) parts.push(`RMSE ${i.rmse.toFixed(2)}`);
  if (Number.isFinite(i.r2)) parts.push(`R² ${i.r2.toFixed(3)}`);
  if (Number.isFinite(i.baseline_mae)) parts.push(`baseline ${i.baseline_mae.toFixed(2)} tahun`);
  el.textContent = `Model: ${parts.join(' · ')}`;
}

/** Selisih rata-rata model untuk usia tertentu: per kelompok bila tersedia, jika tidak keseluruhan. */
function errorFor(age) {
  const overall = state.info && Number.isFinite(state.info.mae) ? state.info.mae : CONFIG.MAE;
  const g = state.info && state.info.groups.find((item) => Math.round(age) <= item.max);
  return { mae: g ? g.mae : overall, label: g ? g.label : null, overall };
}

/* ==========================================================================
   Memuat foto
   ========================================================================== */
function loadFile(file) {
  clearError();
  if (!file) return;
  if (!file.type || !file.type.startsWith('image/')) {
    return showError('Berkas yang dipilih bukan gambar. Gunakan format JPG, PNG, atau WebP.');
  }
  if (file.size === 0) return showError('Berkas kosong dan tidak dapat dibuka.');
  if (file.size > CONFIG.MAX_FILE_MB * 1024 * 1024) {
    return showError(`Ukuran berkas melebihi ${CONFIG.MAX_FILE_MB} MB. Pilih foto yang lebih kecil.`);
  }

  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    URL.revokeObjectURL(url);
    setSource(img, img.naturalWidth, img.naturalHeight);
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    showError('Gambar tidak dapat dibaca. Berkas mungkin rusak, atau formatnya (misalnya HEIC) tidak didukung ' +
      'browser. Coba simpan ulang sebagai JPG atau PNG.');
  };
  img.src = url;
}

async function setSource(source, w, h) {
  if (Math.min(w, h) < CONFIG.MIN_IMAGE_PX) {
    return showError(`Resolusi foto terlalu kecil (${w}×${h} px). Gunakan foto dengan sisi terpendek minimal ${CONFIG.MIN_IMAGE_PX} px.`);
  }
  state.source = source;
  state.srcW = w;
  state.srcH = h;
  const s = Math.min(w, h);
  state.crop = { x: (w - s) / 2, y: (h - s) / 2, s };

  els.editor.hidden = false;
  els.stateEmpty.hidden = true;
  els.stateResult.hidden = false;
  fitStage();
  onCropChanged();
  els.editor.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  const found = await detectFace(true);
  if (!found && 'FaceDetector' in window) toast('Wajah tidak terdeteksi otomatis. Atur kotak secara manual.');
}

function resetSource() {
  state.source = null;
  state.last = null;
  state.runId++;
  clearTimeout(state.timer);
  els.editor.hidden = true;
  els.stateResult.hidden = true;
  els.stateEmpty.hidden = false;
  els.fileInput.value = '';
  clearError();
}

/* ==========================================================================
   Kotak potong wajah
   ========================================================================== */
const HANDLE = 16; // radius area sentuh sudut (piksel kanvas)

function minCrop() { return Math.max(24, Math.min(state.srcW, state.srcH) * 0.12); }
function maxCrop() { return Math.min(state.srcW, state.srcH); }

function setCrop(x, y, s) {
  s = clamp(s, minCrop(), maxCrop());
  x = clamp(x, 0, state.srcW - s);
  y = clamp(y, 0, state.srcH - s);
  state.crop = { x, y, s };
}

function zoomAround(factor) {
  const c = state.crop;
  const cx = c.x + c.s / 2, cy = c.y + c.s / 2;
  const s = clamp(c.s * factor, minCrop(), maxCrop());
  setCrop(cx - s / 2, cy - s / 2, s);
  onCropChanged();
}

function syncSlider() {
  const lo = minCrop(), hi = maxCrop();
  els.zoom.value = hi === lo ? 0 : Math.round(((hi - state.crop.s) / (hi - lo)) * 100);
}

function fitStage() {
  const maxW = Math.max(160, els.stageWrap.clientWidth - 16);
  const maxH = 440;
  const scale = Math.min(maxW / state.srcW, maxH / state.srcH, 1);
  state.scale = scale;
  const dpr = window.devicePixelRatio || 1;
  els.stage.width = Math.round(state.srcW * scale * dpr);
  els.stage.height = Math.round(state.srcH * scale * dpr);
  els.stage.style.width = `${Math.round(state.srcW * scale)}px`;
  els.stage.style.height = `${Math.round(state.srcH * scale)}px`;
}

function renderStage() {
  if (!state.source) return;
  const cv = els.stage, ctx = cv.getContext('2d');
  const k = cv.width / state.srcW;          // piksel foto -> piksel kanvas
  const { x, y, s } = state.crop;

  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(state.source, 0, 0, cv.width, cv.height);

  // Redupkan area di luar kotak
  ctx.save();
  ctx.fillStyle = 'rgba(8, 10, 20, 0.55)';
  ctx.beginPath();
  ctx.rect(0, 0, cv.width, cv.height);
  ctx.rect(x * k, y * k, s * k, s * k);
  ctx.fill('evenodd');
  ctx.restore();

  // Bingkai
  const bx = x * k, by = y * k, bs = s * k;
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  ctx.strokeStyle = '#ffffff';
  ctx.strokeRect(bx, by, bs, bs);

  // Panduan oval wajah
  ctx.save();
  ctx.setLineDash([6, 6]);
  ctx.lineWidth = 1.2 * (window.devicePixelRatio || 1);
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.beginPath();
  ctx.ellipse(bx + bs / 2, by + bs / 2, bs * 0.34, bs * 0.44, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  // Pegangan sudut
  const r = 6 * (window.devicePixelRatio || 1);
  ctx.fillStyle = '#ffffff';
  for (const [cx, cy] of corners()) {
    ctx.beginPath();
    ctx.arc(cx * k, cy * k, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

function corners() {
  const { x, y, s } = state.crop;
  return [[x, y], [x + s, y], [x, y + s], [x + s, y + s]];
}

/** Mengubah posisi pointer menjadi koordinat pada foto asli. */
function toImageCoords(ev) {
  const rect = els.stage.getBoundingClientRect();
  return {
    x: ((ev.clientX - rect.left) / rect.width) * state.srcW,
    y: ((ev.clientY - rect.top) / rect.height) * state.srcH,
  };
}

function hitCorner(p) {
  const tol = HANDLE / state.scale;
  let best = -1, bestD = tol;
  corners().forEach(([cx, cy], i) => {
    const d = Math.hypot(p.x - cx, p.y - cy);
    if (d <= bestD) { best = i; bestD = d; }
  });
  return best;
}

function insideCrop(p) {
  const { x, y, s } = state.crop;
  return p.x >= x && p.x <= x + s && p.y >= y && p.y <= y + s;
}

function onPointerDown(ev) {
  if (!state.source) return;
  els.stage.setPointerCapture(ev.pointerId);
  const p = toImageCoords(ev);
  const corner = hitCorner(p);

  if (corner >= 0) {
    // Titik tetap adalah sudut yang berseberangan
    const [ax, ay] = corners()[3 - corner];
    state.drag = { mode: 'resize', ax, ay, dirX: corner % 2 === 0 ? -1 : 1, dirY: corner < 2 ? -1 : 1 };
  } else {
    if (!insideCrop(p)) setCrop(p.x - state.crop.s / 2, p.y - state.crop.s / 2, state.crop.s);
    state.drag = { mode: 'move', dx: p.x - state.crop.x, dy: p.y - state.crop.y };
    onCropChanged();
  }
}

function onPointerMove(ev) {
  if (!state.source) return;
  const p = toImageCoords(ev);

  if (!state.drag) {
    const corner = hitCorner(p);
    els.stage.style.cursor = corner >= 0 ? (corner === 0 || corner === 3 ? 'nwse-resize' : 'nesw-resize')
      : insideCrop(p) ? 'move' : 'crosshair';
    return;
  }

  const d = state.drag;
  if (d.mode === 'move') {
    setCrop(p.x - d.dx, p.y - d.dy, state.crop.s);
  } else {
    const roomX = d.dirX > 0 ? state.srcW - d.ax : d.ax;
    const roomY = d.dirY > 0 ? state.srcH - d.ay : d.ay;
    const want = Math.max(Math.abs(p.x - d.ax), Math.abs(p.y - d.ay));
    const s = clamp(want, minCrop(), Math.min(roomX, roomY));
    setCrop(d.dirX > 0 ? d.ax : d.ax - s, d.dirY > 0 ? d.ay : d.ay - s, s);
  }
  onCropChanged();
}

function onPointerUp() { state.drag = null; }

function onCropChanged(predict = true) {
  syncSlider();
  renderStage();
  drawPreview();
  if (predict) schedulePredict();
}

function drawPreview() {
  const S = CONFIG.INPUT_SIZE, ctx = els.preview.getContext('2d');
  const { x, y, s } = state.crop;
  ctx.fillStyle = '#808080';                 // latar netral untuk gambar transparan
  ctx.fillRect(0, 0, S, S);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(state.source, x, y, s, s, 0, 0, S, S);
}

/* ==========================================================================
   Deteksi wajah otomatis (bila browser mendukung FaceDetector)
   ========================================================================== */
async function detectFace(silent = false) {
  if (!('FaceDetector' in window) || !state.source) return false;
  try {
    const detector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
    const faces = await detector.detect(state.source);
    if (!faces.length) return false;
    const b = faces[0].boundingBox;
    const s = Math.max(b.width, b.height) * 1.5;
    setCrop(b.x + b.width / 2 - s / 2, b.y + b.height / 2 - s / 2, s);
    onCropChanged();
    if (!silent) toast('Wajah terdeteksi.');
    return true;
  } catch (err) {
    console.warn('Deteksi wajah tidak tersedia:', err);
    return false;
  }
}

/* ==========================================================================
   Prediksi
   ========================================================================== */
function schedulePredict(delay = CONFIG.PREDICT_DELAY_MS) {
  clearTimeout(state.timer);
  els.ageBlock.classList.add('is-busy');
  state.timer = setTimeout(runPrediction, delay);
}

function groupOf(age) {
  const g = CONFIG.GROUPS.find((item) => Math.round(age) <= item.max);
  return g ? g.name : '';
}

async function runPrediction() {
  if (!state.source) return;
  if (!state.engine) {
    els.ageBlock.classList.add('is-busy');
    const li = document.createElement('li');
    li.textContent = 'Menunggu model siap. Estimasi akan muncul otomatis setelah model berhasil dimuat.';
    els.notes.replaceChildren(li);
    return; // dipanggil ulang otomatis setelah model selesai dimuat
  }
  const id = ++state.runId;
  const t0 = performance.now();

  try {
    // Dua masukan: foto asli dan versi cerminnya, lalu dirata-rata.
    const values = await state.engine.predict(els.preview);
    if (id !== state.runId) return;           // ada prediksi yang lebih baru

    if (values.length < 2 || !Number.isFinite(values[0]) || !Number.isFinite(values[1])) {
      throw new AppError('Model menghasilkan nilai yang tidak valid. Coba foto atau area wajah yang lain.');
    }
    const age = clamp((values[0] + values[1]) / 2, 1, 116);
    const diff = Math.abs(values[0] - values[1]);
    showResult(age, diff, performance.now() - t0);
  } catch (err) {
    handleError(err, 'Prediksi gagal dijalankan. Coba pilih ulang foto atau muat ulang halaman.');
  } finally {
    if (id === state.runId) els.ageBlock.classList.remove('is-busy');
  }
}

/* ==========================================================================
   Tampilan hasil
   ========================================================================== */
function animateAge(to) {
  const from = state.shownAge;
  state.shownAge = to;
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduce || !Number.isFinite(from)) { els.ageValue.textContent = to.toFixed(1); return; }
  const start = performance.now(), dur = 450;
  const step = (now) => {
    const t = clamp((now - start) / dur, 0, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    els.ageValue.textContent = (from + (to - from) * eased).toFixed(1);
    if (t < 1 && state.shownAge === to) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function showResult(age, diff, ms) {
  state.last = { age, diff };
  animateAge(age);
  els.ageGroup.textContent = groupOf(age);

  const err = errorFor(age);
  const lo = Math.max(0, age - err.mae), hi = Math.min(100, age + err.mae);
  els.scaleRange.style.left = `${lo}%`;
  els.scaleRange.style.width = `${Math.max(1, hi - lo)}%`;
  els.scaleMarker.style.left = `${clamp(age, 0, 100)}%`;

  els.factRange.textContent = `${Math.max(1, Math.round(age - err.mae))} – ${Math.round(age + err.mae)} th`;
  const stable = diff <= CONFIG.UNSTABLE_DIFF;
  els.factConsistency.textContent = stable ? 'Stabil' : 'Kurang stabil';
  els.factTime.textContent = `${Math.round(ms)} ms`;

  const notes = [];
  if (!stable) {
    notes.push({ warn: true, text: `Prediksi dari foto asli dan versi cerminnya berbeda ${diff.toFixed(1)} tahun. ` +
      'Coba foto yang lebih jelas atau atur ulang kotak wajah.' });
  }
  if (age < 6 || age > 70) {
    notes.push({ warn: true, text: 'Data latih pada rentang usia ini lebih sedikit, sehingga hasilnya cenderung kurang akurat.' });
  }
  if (state.crop.s < 48) {
    notes.push({ warn: true, text: 'Area wajah pada foto asli berukuran kecil. Foto dengan wajah lebih besar biasanya memberi hasil lebih baik.' });
  }
  notes.push({ warn: false, text: err.label
    ? `Pada data uji, selisih rata-rata model untuk kelompok usia ${err.label} sekitar ${err.mae.toFixed(1)} tahun ` +
      `(keseluruhan ${err.overall.toFixed(1)} tahun). Anggap hasil ini sebagai perkiraan.`
    : `Selisih rata-rata model pada data uji sekitar ${err.mae.toFixed(1)} tahun, jadi anggap hasil ini sebagai perkiraan.` });

  els.notes.replaceChildren(...notes.map((n) => {
    const li = document.createElement('li');
    li.textContent = n.text;
    if (n.warn) li.className = 'warn';
    return li;
  }));
}

/* ==========================================================================
   Riwayat
   ========================================================================== */
function renderHistory() {
  els.history.replaceChildren(...state.history.map((item) => {
    const li = document.createElement('li');
    const img = document.createElement('img');
    img.src = item.thumb;
    img.alt = `Foto dengan perkiraan usia ${item.age.toFixed(1)} tahun`;
    const box = document.createElement('div');
    const b = document.createElement('b');
    b.textContent = `${item.age.toFixed(1)} tahun`;
    const span = document.createElement('span');
    span.textContent = `${item.group} · ${item.time}`;
    box.append(b, span);
    li.append(img, box);
    return li;
  }));
  els.historyEmpty.hidden = state.history.length > 0;
  els.btnClear.hidden = state.history.length === 0;
}

function saveToHistory() {
  if (!state.last) return toast('Belum ada hasil untuk disimpan.');
  const thumb = document.createElement('canvas');
  thumb.width = thumb.height = 96;
  thumb.getContext('2d').drawImage(els.preview, 0, 0, 96, 96);
  state.history.unshift({
    age: state.last.age,
    group: groupOf(state.last.age),
    time: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' }),
    thumb: thumb.toDataURL('image/jpeg', 0.8),
  });
  state.history.length = Math.min(state.history.length, CONFIG.MAX_HISTORY);
  renderHistory();
  toast('Hasil disimpan ke riwayat.');
}

/* ==========================================================================
   Kamera
   ========================================================================== */
function cameraMessage(err) {
  switch (err && err.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Izin kamera ditolak. Izinkan akses kamera pada pengaturan situs di browser, lalu coba lagi.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'Kamera tidak ditemukan pada perangkat ini.';
    case 'NotReadableError':
    case 'AbortError':
      return 'Kamera sedang dipakai aplikasi lain atau tidak dapat dibuka. Tutup aplikasi tersebut lalu coba lagi.';
    default:
      return 'Kamera tidak dapat diaktifkan. Pastikan halaman dibuka lewat localhost atau https.';
  }
}

async function startCamera() {
  clearError();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return showError('Browser ini tidak mendukung akses kamera, atau halaman tidak dibuka lewat alamat yang aman (localhost atau https).');
  }
  try {
    stopCamera();
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
    els.video.srcObject = state.stream;
    await els.video.play();
    els.cameraBox.classList.add('is-live');
    els.camStart.disabled = true;
    els.camShot.disabled = false;
    els.camStop.disabled = false;
  } catch (err) {
    stopCamera();
    console.error(err);
    showError(cameraMessage(err));
  }
}

function stopCamera() {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  els.video.srcObject = null;
  els.cameraBox.classList.remove('is-live');
  els.camStart.disabled = false;
  els.camShot.disabled = true;
  els.camStop.disabled = true;
}

function takePhoto() {
  const w = els.video.videoWidth, h = els.video.videoHeight;
  if (!w || !h) return showError('Kamera belum siap. Tunggu sebentar lalu coba lagi.');
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d').drawImage(els.video, 0, 0, w, h);
  stopCamera();
  setSource(canvas, w, h);
}

/* ==========================================================================
   Tab dan pemasangan event
   ========================================================================== */
function selectTab(which) {
  const upload = which === 'upload';
  els.tabUpload.classList.toggle('is-active', upload);
  els.tabCamera.classList.toggle('is-active', !upload);
  els.tabUpload.setAttribute('aria-selected', String(upload));
  els.tabCamera.setAttribute('aria-selected', String(!upload));
  els.panelUpload.hidden = !upload;
  els.panelCamera.hidden = upload;
  if (upload) stopCamera();
}

function bindEvents() {
  els.alertClose.addEventListener('click', clearError);
  els.retryModel.addEventListener('click', () => { clearError(); loadModel(); });

  els.tabUpload.addEventListener('click', () => selectTab('upload'));
  els.tabCamera.addEventListener('click', () => selectTab('camera'));

  els.fileInput.addEventListener('change', (e) => loadFile(e.target.files[0]));

  ['dragenter', 'dragover'].forEach((t) => els.dropzone.addEventListener(t, (e) => {
    e.preventDefault();
    els.dropzone.classList.add('is-over');
  }));
  ['dragleave', 'drop'].forEach((t) => els.dropzone.addEventListener(t, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove('is-over');
  }));
  els.dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length > 1) toast('Hanya satu foto yang diproses. Foto pertama dipakai.');
    loadFile(files && files[0]);
  });
  // Mencegah browser membuka foto bila terlepas di luar area unggah
  ['dragover', 'drop'].forEach((t) => window.addEventListener(t, (e) => e.preventDefault()));

  window.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData ? e.clipboardData.items : [])].find((i) => i.type.startsWith('image/'));
    if (item) { selectTab('upload'); loadFile(item.getAsFile()); }
  });

  els.camStart.addEventListener('click', startCamera);
  els.camShot.addEventListener('click', takePhoto);
  els.camStop.addEventListener('click', stopCamera);
  window.addEventListener('pagehide', stopCamera);

  els.stage.addEventListener('pointerdown', onPointerDown);
  els.stage.addEventListener('pointermove', onPointerMove);
  els.stage.addEventListener('pointerup', onPointerUp);
  els.stage.addEventListener('pointercancel', onPointerUp);
  els.stage.addEventListener('wheel', (e) => {
    if (!state.source) return;
    e.preventDefault();
    zoomAround(e.deltaY < 0 ? 0.93 : 1.07);
  }, { passive: false });

  els.zoom.addEventListener('input', () => {
    const t = Number(els.zoom.value) / 100;
    const lo = minCrop(), hi = maxCrop();
    const s = hi - t * (hi - lo);
    const c = state.crop;
    setCrop(c.x + c.s / 2 - s / 2, c.y + c.s / 2 - s / 2, s);
    renderStage();
    drawPreview();
    schedulePredict();
  });

  els.btnCenter.addEventListener('click', () => {
    const c = state.crop;
    setCrop((state.srcW - c.s) / 2, (state.srcH - c.s) / 2, c.s);
    onCropChanged();
  });
  els.btnDetect.addEventListener('click', async () => {
    const ok = await detectFace(false);
    if (!ok) toast('Wajah tidak terdeteksi. Atur kotak secara manual.');
  });
  els.btnReset.addEventListener('click', resetSource);

  els.btnSave.addEventListener('click', saveToHistory);
  els.btnClear.addEventListener('click', () => { state.history = []; renderHistory(); });

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (state.source) { fitStage(); renderStage(); } }, 120);
  });
}

/* ==========================================================================
   Mulai
   ========================================================================== */
function init() {
  initTheme();
  bindEvents();
  renderHistory();
  els.btnDetect.hidden = !('FaceDetector' in window);
  window.addEventListener('load', loadModel); // menunggu pustaka tf selesai dimuat
}

init();

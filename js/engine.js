/* ==========================================================================
   Mesin inferensi JavaScript murni (tanpa TensorFlow.js).

   Dipakai sebagai cadangan bila TensorFlow.js tidak dapat dimuat atau hasilnya
   tidak lolos uji mandiri. Membaca model.json dan berkas bobot yang sama dengan
   TensorFlow.js, jadi tidak ada berkas tambahan yang diperlukan.

   Lapisan yang didukung: Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten,
   Dense, Dropout, dan BatchNormalization (model Sequential, channels_last).
   ========================================================================== */
(function (root) {
  'use strict';

  const SUPPORTED = ['Conv2D', 'MaxPooling2D', 'GlobalAveragePooling2D', 'Flatten',
    'Dense', 'Dropout', 'BatchNormalization'];

  /** Citra uji deterministik; rumusnya sama dengan selftest_input() di convert_model.py. */
  function selftestImage(size) {
    const out = new Float32Array(size * size * 3);
    let i = 0;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        for (let c = 0; c < 3; c++) {
          out[i++] = 0.5 + 0.4 * Math.sin(0.07 * x + 1.3 * c) * Math.cos(0.05 * y + 0.7 * c);
        }
      }
    }
    return out;
  }

  function flipHorizontal(x, h, w, c) {
    const out = new Float32Array(x.length);
    for (let y = 0; y < h; y++) {
      for (let col = 0; col < w; col++) {
        const src = (y * w + col) * c, dst = (y * w + (w - 1 - col)) * c;
        for (let k = 0; k < c; k++) out[dst + k] = x[src + k];
      }
    }
    return out;
  }

  function activate(arr, name) {
    if (name === 'relu') {
      for (let i = 0; i < arr.length; i++) if (arr[i] < 0) arr[i] = 0;
    } else if (name === 'sigmoid') {
      for (let i = 0; i < arr.length; i++) arr[i] = 1 / (1 + Math.exp(-arr[i]));
    } else if (name === 'tanh') {
      for (let i = 0; i < arr.length; i++) arr[i] = Math.tanh(arr[i]);
    } else if (name && name !== 'linear') {
      throw new Error(`Fungsi aktivasi '${name}' belum didukung mesin JavaScript.`);
    }
    return arr;
  }

  function conv2d(x, H, W, C, k, kh, kw, F, bias, same, sy, sx) {
    const oh = same ? Math.ceil(H / sy) : Math.floor((H - kh) / sy) + 1;
    const ow = same ? Math.ceil(W / sx) : Math.floor((W - kw) / sx) + 1;
    const padH = same ? Math.max((oh - 1) * sy + kh - H, 0) : 0;
    const padW = same ? Math.max((ow - 1) * sx + kw - W, 0) : 0;
    const top = padH >> 1, left = padW >> 1;
    const out = new Float32Array(oh * ow * F);

    for (let oy = 0; oy < oh; oy++) {
      for (let ox = 0; ox < ow; ox++) {
        const base = (oy * ow + ox) * F;
        if (bias) for (let f = 0; f < F; f++) out[base + f] = bias[f];
        for (let ky = 0; ky < kh; ky++) {
          const iy = oy * sy + ky - top;
          if (iy < 0 || iy >= H) continue;
          for (let kx = 0; kx < kw; kx++) {
            const ix = ox * sx + kx - left;
            if (ix < 0 || ix >= W) continue;
            const xi = (iy * W + ix) * C;
            const kb = (ky * kw + kx) * C * F;
            for (let c = 0; c < C; c++) {
              const v = x[xi + c];
              if (v === 0) continue;              // lompatan cepat untuk keluaran ReLU yang nol
              const ko = kb + c * F;
              for (let f = 0; f < F; f++) out[base + f] += v * k[ko + f];
            }
          }
        }
      }
    }
    return { data: out, h: oh, w: ow, c: F };
  }

  function maxPool(x, H, W, C, ph, pw, sy, sx) {
    const oh = Math.floor((H - ph) / sy) + 1, ow = Math.floor((W - pw) / sx) + 1;
    const out = new Float32Array(oh * ow * C);
    for (let oy = 0; oy < oh; oy++) {
      for (let ox = 0; ox < ow; ox++) {
        for (let c = 0; c < C; c++) {
          let m = -Infinity;
          for (let py = 0; py < ph; py++) {
            for (let px = 0; px < pw; px++) {
              const v = x[((oy * sy + py) * W + (ox * sx + px)) * C + c];
              if (v > m) m = v;
            }
          }
          out[(oy * ow + ox) * C + c] = m;
        }
      }
    }
    return { data: out, h: oh, w: ow, c: C };
  }

  function dense(x, k, units, bias) {
    const n = x.length, out = new Float32Array(units);
    if (bias) out.set(bias);
    for (let i = 0; i < n; i++) {
      const v = x[i];
      if (v === 0) continue;
      const ko = i * units;
      for (let j = 0; j < units; j++) out[j] += v * k[ko + j];
    }
    return out;
  }

  const pair = (v, d) => (v == null ? d : (typeof v === 'number' ? [v, v] : v));

  class JsModel {
    constructor(layers, weights, inputShape) {
      this.layers = layers;
      this.weights = weights;
      this.input = { h: inputShape[1], w: inputShape[2], c: inputShape[3] };
    }

    /** Menjalankan satu citra (Float32Array HWC, nilai 0..1) dan mengembalikan angka keluaran pertama. */
    async run(image, tick) {
      let t = { data: image, h: this.input.h, w: this.input.w, c: this.input.c };
      let flat = null;

      for (const layer of this.layers) {
        const cls = layer.class_name, cfg = layer.config, name = cfg.name;
        const wt = this.weights;

        if (cls === 'Conv2D') {
          const k = wt[`${name}/kernel`];
          const [kh, kw] = pair(cfg.kernel_size, [3, 3]);
          const [sy, sx] = pair(cfg.strides, [1, 1]);
          const bias = cfg.use_bias === false ? null : wt[`${name}/bias`].data;
          t = conv2d(t.data, t.h, t.w, t.c, k.data, kh, kw, k.shape[3], bias, cfg.padding === 'same', sy, sx);
          activate(t.data, cfg.activation);
        } else if (cls === 'MaxPooling2D') {
          const [ph, pw] = pair(cfg.pool_size, [2, 2]);
          const [sy, sx] = pair(cfg.strides, [ph, pw]);
          t = maxPool(t.data, t.h, t.w, t.c, ph, pw, sy, sx);
        } else if (cls === 'GlobalAveragePooling2D') {
          const out = new Float32Array(t.c), n = t.h * t.w;
          for (let i = 0; i < n; i++) for (let c = 0; c < t.c; c++) out[c] += t.data[i * t.c + c] / n;
          flat = out;
        } else if (cls === 'Flatten') {
          flat = t.data;
        } else if (cls === 'Dense') {
          const k = wt[`${name}/kernel`];
          const bias = cfg.use_bias === false ? null : wt[`${name}/bias`].data;
          flat = activate(dense(flat, k.data, k.shape[1], bias), cfg.activation);
        } else if (cls === 'BatchNormalization') {
          const mean = wt[`${name}/moving_mean`].data, variance = wt[`${name}/moving_variance`].data;
          const gamma = wt[`${name}/gamma`] ? wt[`${name}/gamma`].data : null;
          const beta = wt[`${name}/beta`] ? wt[`${name}/beta`].data : null;
          const eps = cfg.epsilon == null ? 1e-3 : cfg.epsilon;
          const arr = flat || t.data, ch = mean.length;
          for (let i = 0; i < arr.length; i++) {
            const c = i % ch;
            arr[i] = (arr[i] - mean[c]) / Math.sqrt(variance[c] + eps) * (gamma ? gamma[c] : 1) + (beta ? beta[c] : 0);
          }
        }
        // Dropout tidak berpengaruh saat inferensi
        if (tick) await tick();
      }
      if (!flat || !Number.isFinite(flat[0])) throw new Error('Keluaran model tidak valid.');
      return flat[0];
    }
  }

  /** Membangun model dari isi model.json dan berkas bobot (ArrayBuffer). */
  function createModel(modelJson, buffer) {
    const layers = modelJson.modelTopology.model_config.config.layers;
    for (const l of layers) {
      if (!SUPPORTED.includes(l.class_name)) {
        throw new Error(`Lapisan '${l.class_name}' belum didukung mesin JavaScript.`);
      }
    }
    const inputShape = layers[0].config.batch_input_shape;
    if (!inputShape || inputShape.length !== 4) throw new Error('Ukuran masukan model tidak ditemukan.');

    const weights = {};
    let offset = 0;
    for (const w of modelJson.weightsManifest[0].weights) {
      const n = w.shape.reduce((a, b) => a * b, 1);
      if (offset + n * 4 > buffer.byteLength) throw new Error('Berkas bobot lebih pendek dari yang diharapkan.');
      weights[w.name] = { shape: w.shape, data: new Float32Array(buffer, offset, n) };
      offset += n * 4;
    }
    return new JsModel(layers, weights, inputShape);
  }

  const api = { createModel, selftestImage, flipHorizontal };
  root.JsEngine = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);

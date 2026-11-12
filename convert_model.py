"""
Mengubah model Keras (.h5 atau .keras) menjadi format TensorFlow.js
(model.json + berkas bobot .bin) tanpa memerlukan paket tensorflowjs.

Kebutuhan : pip install h5py numpy
Penggunaan: python convert_model.py model_estimasi_usia.h5
Keluaran  : folder model/ berisi model.json dan group1-shard1of1.bin.
            Bila model_info.json (dari notebook) ada di folder yang sama dengan
            model atau folder kerja, berkas itu ikut disalin ke folder keluaran.

Lapisan yang didukung: Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
BatchNormalization, dan GlobalAveragePooling2D (model Sequential).
"""
import argparse
import json
import os
import sys
import tempfile
import zipfile

import numpy as np

SUPPORTED = {"Conv2D", "MaxPooling2D", "Flatten", "Dense", "Dropout",
             "BatchNormalization", "GlobalAveragePooling2D"}

# Urutan bobot per jenis lapisan, sesuai yang diharapkan TensorFlow.js
WEIGHT_ORDER = {
    "Conv2D": ["kernel", "bias"],
    "Dense": ["kernel", "bias"],
    "BatchNormalization": ["gamma", "beta", "moving_mean", "moving_variance"],
}

VS_INIT = {"class_name": "VarianceScaling",
           "config": {"scale": 1.0, "mode": "fan_avg",
                      "distribution": "uniform", "seed": None}}
ZEROS = {"class_name": "Zeros", "config": {}}
ONES = {"class_name": "Ones", "config": {}}


def pair(value, default):
    """Mengubah angka tunggal atau daftar menjadi pasangan [a, b]."""
    if value is None:
        return list(default)
    if isinstance(value, int):
        return [value, value]
    return [int(v) for v in value]


# ---------------------------------------------------------------- pembacaan
def read_h5(path):
    import h5py
    with h5py.File(path, "r") as f:
        raw = f.attrs.get("model_config")
        if raw is None:
            sys.exit("Berkas ini tidak memuat arsitektur model. Gunakan berkas "
                     "hasil model.save(), bukan save_weights().")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        config = json.loads(raw)
        group = f["model_weights"] if "model_weights" in f else f
        weights = {}
        for layer in config["config"]["layers"]:
            name = layer["config"]["name"]
            if name not in group:
                continue
            found = {}

            def visit(_, obj):
                if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                    key = obj.name.split("/")[-1].split(":")[0]
                    found[key] = np.array(obj, dtype=np.float32)

            group[name].visititems(visit)
            if found:
                weights[name] = found
    return config, weights


def read_keras_zip(path):
    import h5py
    with zipfile.ZipFile(path) as z:
        config = json.loads(z.read("config.json"))
        with tempfile.TemporaryDirectory() as tmp:
            z.extract("model.weights.h5", tmp)
            weights = {}
            with h5py.File(os.path.join(tmp, "model.weights.h5"), "r") as f:
                for layer in config["config"]["layers"]:
                    cls, name = layer["class_name"], layer["config"]["name"]
                    order = WEIGHT_ORDER.get(cls)
                    if not order or name not in f["layers"]:
                        continue
                    vars_ = f["layers"][name]["vars"]
                    weights[name] = {k: np.array(vars_[str(i)], dtype=np.float32)
                                     for i, k in enumerate(order) if str(i) in vars_}
    return config, weights


# ------------------------------------------------------------------ topologi
def input_shape_from(config, default_size):
    layers = config["config"]["layers"]
    for layer in layers:
        cfg = layer["config"]
        shape = cfg.get("batch_shape") or cfg.get("batch_input_shape")
        if shape:
            return [None] + [int(v) for v in shape[1:]]
    build = config["config"].get("build_input_shape")
    if build:
        return [None] + [int(v) for v in build[1:]]
    print(f"Peringatan: ukuran input tidak ditemukan, memakai {default_size}x{default_size}x3.")
    return [None, default_size, default_size, 3]


def tfjs_layer(layer, batch_shape=None):
    cls, cfg = layer["class_name"], layer["config"]
    out = {"name": cfg["name"], "trainable": True}
    if batch_shape is not None:
        out["batch_input_shape"] = batch_shape
        out["dtype"] = "float32"

    if cls == "Conv2D":
        out.update({
            "filters": int(cfg["filters"]),
            "kernel_size": pair(cfg.get("kernel_size"), (3, 3)),
            "strides": pair(cfg.get("strides"), (1, 1)),
            "padding": cfg.get("padding", "valid"),
            "data_format": "channels_last",
            "dilation_rate": pair(cfg.get("dilation_rate"), (1, 1)),
            "activation": cfg.get("activation", "linear"),
            "use_bias": bool(cfg.get("use_bias", True)),
            "kernel_initializer": VS_INIT, "bias_initializer": ZEROS,
            "kernel_regularizer": None, "bias_regularizer": None,
            "activity_regularizer": None,
            "kernel_constraint": None, "bias_constraint": None})
    elif cls == "MaxPooling2D":
        pool = pair(cfg.get("pool_size"), (2, 2))
        out.update({"pool_size": pool,
                    "padding": cfg.get("padding", "valid"),
                    "strides": pair(cfg.get("strides"), pool),
                    "data_format": "channels_last"})
    elif cls == "GlobalAveragePooling2D":
        out.update({"data_format": "channels_last"})
    elif cls == "Flatten":
        out.update({"data_format": "channels_last"})
    elif cls == "Dense":
        out.update({
            "units": int(cfg["units"]),
            "activation": cfg.get("activation", "linear"),
            "use_bias": bool(cfg.get("use_bias", True)),
            "kernel_initializer": VS_INIT, "bias_initializer": ZEROS,
            "kernel_regularizer": None, "bias_regularizer": None,
            "activity_regularizer": None,
            "kernel_constraint": None, "bias_constraint": None})
    elif cls == "Dropout":
        out.update({"rate": float(cfg.get("rate", 0.5)),
                    "noise_shape": None, "seed": None})
    elif cls == "BatchNormalization":
        axis = cfg.get("axis", -1)
        if isinstance(axis, (list, tuple)):
            axis = axis[0]
        out.update({
            "axis": int(axis), "momentum": float(cfg.get("momentum", 0.99)),
            "epsilon": float(cfg.get("epsilon", 1e-3)),
            "center": bool(cfg.get("center", True)),
            "scale": bool(cfg.get("scale", True)),
            "beta_initializer": ZEROS, "gamma_initializer": ONES,
            "moving_mean_initializer": ZEROS, "moving_variance_initializer": ONES,
            "beta_regularizer": None, "gamma_regularizer": None,
            "beta_constraint": None, "gamma_constraint": None})
    return {"class_name": cls, "config": out}


def build_model_json(config, weights, default_size=128):
    if config.get("class_name") != "Sequential":
        sys.exit("Skrip ini hanya mendukung model Sequential.")
    batch_shape = input_shape_from(config, default_size)
    topology, manifest, blobs = [], [], []

    for layer in config["config"]["layers"]:
        cls = layer["class_name"]
        if cls == "InputLayer":
            continue
        if cls not in SUPPORTED:
            sys.exit(f"Lapisan '{cls}' belum didukung oleh skrip konversi ini.")
        name = layer["config"]["name"]
        topology.append(tfjs_layer(layer, batch_shape if not topology else None))

        for key in WEIGHT_ORDER.get(cls, []):
            if cls == "BatchNormalization" and key == "gamma" and not layer["config"].get("scale", True):
                continue
            if cls == "BatchNormalization" and key == "beta" and not layer["config"].get("center", True):
                continue
            if cls in ("Conv2D", "Dense") and key == "bias" and not layer["config"].get("use_bias", True):
                continue
            arr = weights.get(name, {}).get(key)
            if arr is None:
                sys.exit(f"Bobot '{key}' untuk lapisan '{name}' tidak ditemukan di berkas.")
            manifest.append({"name": f"{name}/{key}", "shape": list(arr.shape),
                             "dtype": "float32"})
            blobs.append(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    model_json = {
        "format": "layers-model",
        "generatedBy": "keras v2.15.0",
        "convertedBy": "convert_model.py",
        "modelTopology": {
            "keras_version": "2.15.0", "backend": "tensorflow",
            "model_config": {"class_name": "Sequential",
                             "config": {"name": config["config"].get("name", "model"),
                                        "layers": topology}}},
        "weightsManifest": [{"paths": ["group1-shard1of1.bin"], "weights": manifest}],
    }
    return model_json, b"".join(blobs), manifest, batch_shape


# ------------------------------------------------------------ uji mandiri
def selftest_input(size):
    """Citra uji deterministik; rumus yang sama dipakai di js/engine.js."""
    ys, xs, cs = np.meshgrid(np.arange(size), np.arange(size), np.arange(3), indexing="ij")
    img = 0.5 + 0.4 * np.sin(0.07 * xs + 1.3 * cs) * np.cos(0.05 * ys + 0.7 * cs)
    return img.astype(np.float32)[None]


def _windows(x, kh, kw, sy, sx):
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(x, (kh, kw), axis=(1, 2))      # N,oh,ow,C,kh,kw
    return w[:, ::sy, ::sx]


def _pad_same(x, kh, kw, sy, sx):
    h, w = x.shape[1:3]
    oh, ow = -(-h // sy), -(-w // sx)
    ph, pw = max((oh - 1) * sy + kh - h, 0), max((ow - 1) * sx + kw - w, 0)
    return np.pad(x, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)))


def numpy_forward(config, weights, x):
    """Perhitungan maju model Sequential memakai numpy (acuan independen)."""
    for layer in config["config"]["layers"]:
        cls, cfg = layer["class_name"], layer["config"]
        name = cfg["name"]
        if cls == "Conv2D":
            kh, kw = pair(cfg.get("kernel_size"), (3, 3))
            sy, sx = pair(cfg.get("strides"), (1, 1))
            if cfg.get("padding", "valid") == "same":
                x = _pad_same(x, kh, kw, sy, sx)
            win = _windows(x, kh, kw, sy, sx).transpose(0, 1, 2, 4, 5, 3)   # N,oh,ow,kh,kw,C
            n, oh, ow = win.shape[:3]
            k = weights[name]["kernel"]
            y = win.reshape(n, oh, ow, -1) @ k.reshape(-1, k.shape[-1])
            if "bias" in weights[name]:
                y = y + weights[name]["bias"]
            x = np.maximum(y, 0) if cfg.get("activation") == "relu" else y
        elif cls == "MaxPooling2D":
            ph, pw = pair(cfg.get("pool_size"), (2, 2))
            sy, sx = pair(cfg.get("strides"), (ph, pw))
            x = _windows(x, ph, pw, sy, sx).max(axis=(-2, -1))
        elif cls == "GlobalAveragePooling2D":
            x = x.mean(axis=(1, 2))
        elif cls == "Flatten":
            x = x.reshape(x.shape[0], -1)
        elif cls == "Dense":
            y = x @ weights[name]["kernel"]
            if "bias" in weights[name]:
                y = y + weights[name]["bias"]
            act = cfg.get("activation", "linear")
            x = np.maximum(y, 0) if act == "relu" else (1 / (1 + np.exp(-y)) if act == "sigmoid" else y)
        elif cls == "BatchNormalization":
            w = weights[name]
            x = (x - w["moving_mean"]) / np.sqrt(w["moving_variance"] + float(cfg.get("epsilon", 1e-3)))
            x = x * w.get("gamma", 1.0) + w.get("beta", 0.0)
        # Dropout tidak berpengaruh saat inferensi
    return x


def write_info(out_dir, config, weights, input_shape, extra_info_path=None):
    """Menulis model_info.json: menggabungkan berkas info dari notebook dan hasil uji mandiri."""
    info = {}
    if extra_info_path:
        with open(extra_info_path, encoding="utf-8") as f:
            info = json.load(f)
    size = int(input_shape[1])
    expected = float(numpy_forward(config, weights, selftest_input(size)).reshape(-1)[0])
    info["input_size"] = size
    info["selftest"] = {"expected": round(expected, 5)}
    with open(os.path.join(out_dir, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return expected


def main():
    parser = argparse.ArgumentParser(description="Konversi model Keras ke TensorFlow.js")
    parser.add_argument("model", help="berkas model (.h5 atau .keras)")
    parser.add_argument("--out", default="model", help="folder keluaran (bawaan: model)")
    parser.add_argument("--size", type=int, default=128,
                        help="ukuran input cadangan bila tidak terbaca dari berkas")
    parser.add_argument("--info", default=None,
                        help="lokasi model_info.json (bawaan: dicari di sebelah model)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"Berkas tidak ditemukan: {args.model}")
    try:
        import h5py  # noqa: F401
    except ImportError:
        sys.exit("Paket h5py belum terpasang. Jalankan: pip install h5py numpy")

    reader = read_keras_zip if args.model.endswith(".keras") else read_h5
    config, weights = reader(args.model)
    model_json, blob, manifest, input_shape = build_model_json(config, weights, args.size)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "model.json"), "w", encoding="utf-8") as f:
        json.dump(model_json, f)
    with open(os.path.join(args.out, "group1-shard1of1.bin"), "wb") as f:
        f.write(blob)

    total = sum(int(np.prod(w["shape"])) for w in manifest)
    print(f"Selesai. {len(manifest)} tensor bobot, total {total:,} parameter.")
    print(f"Ukuran berkas bobot: {len(blob) / 1e6:.1f} MB")
    print(f"Hasil disimpan di folder: {os.path.abspath(args.out)}")
    print("Model dari notebook proyek ini seharusnya berjumlah 2.338.497 parameter.")

    candidates = [args.info] if args.info else [
        os.path.join(os.path.dirname(os.path.abspath(args.model)), "model_info.json"),
        "model_info.json",
        os.path.join(args.out, "model_info.json")]
    info_path = next((c for c in candidates if c and os.path.isfile(c)), None)
    expected = write_info(args.out, config, weights, input_shape, info_path)
    if info_path:
        print(f"model_info.json dibuat dari: {os.path.abspath(info_path)}")
        if os.path.abspath(info_path) == os.path.abspath(os.path.join(args.out, "model_info.json")):
            print("Catatan: yang dipakai adalah model_info.json lama di folder keluaran. "
                  "Bila model baru, salin model_info.json terbaru dari notebook ke folder ini.")
    else:
        print("Peringatan: model_info.json dari notebook tidak ditemukan. "
              "Aplikasi akan memakai nilai MAE cadangan.")
    print(f"Nilai uji mandiri (numpy): {expected:.4f}")

if __name__ == "__main__":
    main()

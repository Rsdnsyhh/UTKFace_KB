# Estimasi Usia dari Wajah

Aplikasi web untuk memperkirakan usia seseorang dari foto wajah. Modelnya adalah CNN buatan sendiri (tanpa transfer learning) yang dilatih pada dataset UTKFace untuk kasus regresi, sesuai notebook `Proyek_Estimasi_Usia_UTKFace_Final.ipynb`. Seluruh proses prediksi berjalan di browser, sehingga tidak diperlukan server Python.

Model yang dipakai pada folder ini adalah model hasil pelatihan di bagian D notebook (`model_estimasi_usia.h5`): empat blok Conv2D dan MaxPooling2D, satu lapisan Dense 256 neuron, Dropout 0,5, dan satu neuron keluaran, dengan 2.338.497 parameter. Hasil evaluasinya pada data uji adalah MAE 6,86 tahun, RMSE 9,57 tahun, dan R² 0,768. Angka-angka tersebut diambil dari keluaran notebook dan ditulis pada `model/model_info.json`.

## Fitur

Foto dapat dipilih dengan mengunggah berkas, menyeret foto ke halaman, menempel dari clipboard (Ctrl+V), atau mengambil gambar lewat kamera. Setelah foto dipilih, muncul kotak potong yang dapat digeser, ditarik sudutnya, dan diperbesar atau diperkecil. Kotak ini diperlukan karena model dilatih dengan foto wajah yang sudah dipotong rapat, sehingga hasilnya paling baik bila wajah memenuhi kotak.

Setiap kali kotak diubah, perkiraan usia diperbarui otomatis. Halaman hasil menampilkan:

- perkiraan usia dalam tahun beserta kelompoknya (anak-anak, remaja, dewasa, lansia),
- rentang wajar, yaitu perkiraan usia dikurangi dan ditambah MAE model pada data uji,
- tingkat konsistensi, yang didapat dengan membandingkan prediksi foto asli dan versi cerminnya,
- gambar 128 × 128 piksel yang benar-benar diterima model,
- catatan peringatan bila hasil kurang meyakinkan, misalnya wajah terlalu kecil atau usia berada di rentang yang datanya sedikit.

Di bagian bawah halaman tertulis ringkasan model (MAE, RMSE, R², dan MAE baseline). Hasil dapat disimpan ke riwayat selama halaman masih terbuka. Foto tidak dikirim ke mana pun dan tidak disimpan permanen.

## Struktur folder

```
webapp/
├── index.html           halaman utama
├── css/style.css        tampilan (mendukung tema terang dan gelap)
├── js/app.js            logika antarmuka dan prediksi
├── js/engine.js         mesin cadangan berbasis JavaScript murni
├── model/
│   ├── model.json               arsitektur model
│   ├── group1-shard1of1.bin     bobot model (sekitar 9 MB)
│   └── model_info.json          ringkasan evaluasi dan nilai uji mandiri
├── convert_model.py     skrip konversi bila model diganti
└── README.md
```

Folder `model/` sudah berisi hasil konversi dari model pada notebook, jadi aplikasi dapat langsung dijalankan tanpa langkah konversi.

## Cara menjalankan

Perangkat yang dibutuhkan adalah VS Code dengan ekstensi Live Server dan sebuah browser modern (Chrome, Edge, atau Firefox). Koneksi internet dianjurkan karena TensorFlow.js dimuat dari CDN. Tanpa koneksi internet, aplikasi tetap berjalan memakai mesin cadangan, hanya sedikit lebih lambat.

1. Ekstrak berkas zip, lalu buka folder `webapp` di VS Code (File, lalu Open Folder).
2. Pasang ekstensi Live Server dari Marketplace bila belum ada.
3. Klik kanan `index.html`, lalu pilih Open with Live Server.
4. Browser terbuka pada alamat seperti `http://127.0.0.1:5500`.

Indikator di pojok kanan atas berubah dari "Memuat model" menjadi "Model siap", disertai nama mesin yang dipakai. Setelah itu aplikasi dapat digunakan.

Halaman tidak boleh dibuka dengan klik ganda pada `index.html`. Browser memblokir pemuatan model bila halaman dibuka lewat alamat `file://`, dan aplikasi akan menampilkan pesan yang menjelaskan hal ini.

## Cara memakai

1. Pilih foto lewat tab Unggah atau Kamera.
2. Atur kotak agar wajah berada di dalamnya dan memenuhi sebagian besar kotak. Garis putus-putus berbentuk oval membantu sebagai panduan.
3. Baca hasil di panel sebelah kanan. Gunakan tombol Simpan ke riwayat untuk membandingkan beberapa foto.

Agar hasil lebih baik, gunakan foto yang menghadap ke depan, pencahayaan cukup, wajah tidak tertutup, dan resolusi tidak terlalu kecil.

## Cara kerja

Foto yang dipilih dipotong sesuai kotak, diubah menjadi 128 × 128 piksel, lalu nilai pikselnya dibagi 255 agar berada pada rentang 0 sampai 1. Langkah ini sama dengan prapemrosesan saat pelatihan di notebook. Gambar tersebut dimasukkan ke model bersama versi cerminnya, lalu dua hasilnya dirata-rata. Selisih antara kedua hasil dipakai sebagai petunjuk konsistensi: semakin besar selisihnya, semakin kurang stabil prediksi untuk foto tersebut.

Aplikasi memiliki dua mesin untuk menjalankan model yang sama:

- **TensorFlow.js** sebagai mesin utama. Mesin ini memakai GPU melalui WebGL sehingga prediksi selesai dalam hitungan milidetik.
- **JavaScript murni** (`js/engine.js`) sebagai cadangan. Mesin ini dipakai bila TensorFlow.js tidak dapat dimuat, misalnya karena tidak ada internet, atau bila hasilnya tidak lolos uji mandiri. Waktu prediksinya sekitar setengah detik.

Pada saat model dimuat, aplikasi menjalankan sebuah citra uji dan membandingkan hasilnya dengan nilai acuan yang dihitung saat konversi (disimpan pada `model_info.json`). Bila hasil TensorFlow.js menyimpang, aplikasi pindah ke mesin cadangan secara otomatis dan memberi tahu pengguna. Dengan cara ini, hasil prediksi tidak bergantung pada kebetulan versi library atau perangkat yang dipakai.

## Pesan galat yang mungkin muncul

| Pesan | Penyebab dan cara mengatasi |
|---|---|
| Halaman dibuka langsung dari berkas (file://) | Jalankan lewat Live Server, bukan klik ganda. |
| TensorFlow.js tidak dapat dimuat (muncul sebagai pemberitahuan) | Tidak ada koneksi internet atau CDN terblokir. Aplikasi tetap berjalan dengan mesin cadangan. |
| Berkas model tidak dapat dimuat | Folder `model/` tidak lengkap. Pastikan `model.json` dan `group1-shard1of1.bin` ada di dalamnya. |
| Ukuran masukan model tidak sesuai | Model memakai ukuran masukan selain 128 × 128. Sesuaikan `INPUT_SIZE` di `js/app.js`. |
| Berkas model tidak konsisten dengan nilai acuan | `model_info.json` berasal dari model yang berbeda. Ulangi konversi dengan `convert_model.py`. |
| Berkas yang dipilih bukan gambar | Gunakan JPG, PNG, atau WebP. |
| Gambar tidak dapat dibaca | Berkas rusak atau berformat HEIC. Simpan ulang sebagai JPG atau PNG. |
| Resolusi foto terlalu kecil | Gunakan foto dengan sisi terpendek minimal 64 piksel. |
| Izin kamera ditolak | Izinkan akses kamera pada pengaturan situs di browser. |
| Kamera sedang dipakai aplikasi lain | Tutup aplikasi yang memakai kamera, misalnya aplikasi rapat daring. |

## Mengganti model

Bila model dilatih ulang, simpan dengan `model.save('nama_model.h5')` pada notebook. Salin berkas tersebut ke folder `webapp`, lalu jalankan:

```
pip install h5py numpy
python convert_model.py nama_model.h5 --info model_info.json
```

Opsi `--info` bersifat opsional dan menunjuk ke berkas JSON berisi hasil evaluasi model baru, dengan bentuk seperti pada `model/model_info.json` (kolom `mae`, `rmse`, `r2`, `baseline_mae`). Bila berkas itu tidak diberikan, skrip memakai berkas `model_info.json` yang sudah ada, dan angka MAE pada aplikasi akan tetap angka model lama. Karena itu, perbarui angkanya setiap kali model diganti.

Skrip akan membuat ulang isi folder `model/` beserta nilai uji mandiri yang baru. Bila ingin rentang wajar berbeda untuk tiap kelompok usia, tambahkan kolom `groups` pada berkas info, misalnya `[{"label": "60+ (lansia)", "max": 999, "mae": 9.5}]`. Aplikasi akan memakai MAE kelompok yang sesuai dengan usia hasil prediksi.

Skrip konversi mendukung model Sequential dengan lapisan Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, dan GlobalAveragePooling2D. Berkas `.keras` juga dapat dibaca.

## Menayangkan secara online (opsional)

Aplikasi ini hanya terdiri dari berkas statis, jadi dapat ditayangkan di layanan hosting statis tanpa pengaturan tambahan. Dua cara yang paling mudah adalah GitHub Pages (unggah isi folder `webapp` ke repositori, lalu aktifkan Pages) atau Netlify Drop (seret folder `webapp` ke halaman netlify.com/drop). Alamat yang dihasilkan memakai https sehingga fitur kamera juga dapat dipakai.

## Keterbatasan

- Hasil berupa perkiraan. Pada data uji, model rata-rata meleset sekitar 6,9 tahun, dan hanya sekitar setengah prediksi yang meleset paling banyak 5 tahun.
- Data latih paling banyak berada pada usia dewasa muda, sehingga hasil untuk anak kecil dan lansia cenderung kurang akurat. Pada versi ini besar kesalahan per kelompok usia belum diukur, sehingga rentang wajar memakai satu nilai MAE untuk semua usia.
- Dataset UTKFace berisi wajah yang sudah dipotong dan diluruskan. Foto dengan sudut miring, pencahayaan buruk, atau wajah kecil dapat memberi hasil yang kurang baik.
- Deteksi wajah otomatis hanya aktif di browser yang menyediakan fitur FaceDetector. Pada browser lain, kotak diatur secara manual.

# Olist Delivery Delay Analysis & Risk Prioritization

Proyek ini menganalisis pengiriman e-commerce yang terlambat dan menguji apakah tim operasional bisa menemukan lebih banyak order berisiko dengan kapasitas 20 review per hari. Prediksi dibuat saat order disetujui, lalu dipakai untuk menyusun antrean harian dalam simulasi historis.

## Cara menjalankan

Buka `Notebook/olist_delivery.ipynb` dari folder proyek di VS Code atau Jupyter. Pilih interpreter `.venv\Scripts\python.exe` sebagai kernel, lalu jalankan semua sel dari atas.

Buat environment lokal dengan Python 3.11 dan pasang paket dari folder proyek.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Pipeline/requirements.txt matplotlib ipykernel
```

Unduh dataset dari Kaggle. Dari arsipnya, ambil kedua CSV di bawah. Simpan masing-masing sebagai ZIP terpisah dengan nama yang tercantum, lalu letakkan keduanya di `Data/`. Setiap ZIP berisi satu CSV.

| File | Isi yang dipakai |
| --- | --- |
| `olist_orders_dataset.csv.zip` | ID dan status order, waktu approval, tanggal estimasi, dan tanggal barang diterima |
| `olist_customers_dataset.csv.zip` | ID pelanggan dan negara bagian tujuan |

Sumber data adalah [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

## Menjalankan app Streamlit

Jalankan dari folder proyek. Jika `Pipeline/output/models.joblib` atau `Pipeline/output/replay_metrics.csv` belum ada, jalankan perintah `train` di bagian berikut.

```powershell
.\.venv\Scripts\python.exe -m pip install -r App/requirements.txt
.\.venv\Scripts\python.exe -m streamlit run App/app.py
```

Pilih tanggal simulasi, kapasitas 20, 50, atau 100, dan strategi prioritas. App memperbarui antrean **Berisiko telat** dan daftar **Sudah lewat estimasi** lewat fungsi scoring pipeline. Bagian evaluasi di bawah daftar merangkum kinerja strategi dan kapasitas terpilih untuk seluruh periode uji.

## Menjalankan pipeline

Jalankan dari folder proyek dengan Python di `.venv`. Paket pipeline bisa dipasang lewat `pip install -r Pipeline/requirements.txt`.

```powershell
.\.venv\Scripts\python.exe Pipeline/run.py train
.\.venv\Scripts\python.exe Pipeline/run.py score --date 2018-08-15 --capacity 20 --policy carrier
```

| Tahap | Yang dikerjakan |
| --- | --- |
| `ingestion.py` | Membaca dua ZIP, mengecek ID order, dan menggabungkan wilayah pelanggan. |
| `preprocessing.py` | Merapikan timestamp, membuat fitur saat approval dan label historis, lalu membagi data berdasarkan waktu. |
| `training.py` | Melatih logistic regression default dan tuned. Nilai `C` dipilih lewat validasi April. |
| `evaluation.py` | Mengukur hasil uji saat approval dan replay harian pada 20, 50, dan 100 slot review. |
| `scoring.py` | Membuat antrean risiko dan daftar lewat estimasi berdasarkan informasi yang terlihat hingga akhir hari. |

Perintah `train` menyimpan `models.joblib`, `tuning.csv`, `approval_metrics.csv`, dan `replay_metrics.csv` di `Pipeline/output/`. Perintah `score` memuat model yang sudah tersimpan dan menulis `risk_queue.csv` serta `overdue.csv` di folder berdasarkan tanggal, kebijakan, dan kapasitas. Output antrean hanya berisi informasi untuk review pada tanggal yang dipilih.

Tanggal replay didukung dari 1 Juni sampai 31 Agustus 2018. Pilihan kebijakannya `baseline` untuk skor wilayah, `approval` untuk logistic regression default, dan `carrier` untuk model default dengan prioritas tambahan berdasarkan catatan kurir. Scoring memutar ulang hari-hari sebelumnya untuk mengecualikan order yang sudah pernah dipilih dalam kebijakan dan kapasitas tersebut. Karena waktu pembatalan tidak tersedia, order yang akhirnya batal masih bisa terlihat aktif dalam replay.

## Isi notebook

1. **Import dan Read Data** untuk membaca dua CSV dan melihat contoh baris, status order, serta nilai kosong.
2. **Preprocessing Awal** untuk menggabungkan data, mengubah tipe tanggal, mencatat dan menangani null, lalu membuat label.
3. **EDA** untuk cek cakupan, distribusi label, tren bulanan, negara bagian, kota, distribusi hari telat, durasi per tahap, dan kelompok estimasi pengiriman.
4. **Split Data dan Preprocessing Fitur** untuk menyiapkan data berdasarkan waktu dan membentuk tiga fitur yang tersedia ketika order disetujui.
5. **Training dan Tuning** untuk membandingkan aturan risiko per wilayah dengan logistic regression default dan mencari nilai regularisasi `C` melalui data validasi April.
6. **Evaluation dan Simpan Model** untuk menguji ketiga metode pada Juni–Agustus, melihat contoh antrean, dan menyimpan model untuk dipakai lagi.

## Kenapa memilih teknik ini?

| Teknik | Alasan |
| --- | --- |
| Join dengan validasi kardinalitas | Tabel pelanggan menambahkan wilayah tujuan. Validasi join dan cek `order_id` memastikan satu order tetap dihitung sekali. |
| Penanganan null | Baris berstatus delivered tanpa tanggal wajib dikeluarkan dari data berlabel, wilayah kosong diisi dengan kategori tetap, dan null tanggal serah ke kurir hanya memengaruhi analisis durasi per tahap. Tabel audit di notebook menunjukkan jumlah null sebelum dan sesudah. |
| Bandingkan tanggal pada tingkat hari | Order yang diterima pada hari estimasi tetap tepat waktu meski jamnya berbeda. Persentase dihitung dari order yang sudah diterima dan punya tanggal yang diperlukan. |
| Lihat jumlah dan persentase keterlambatan | Jumlah menunjukkan banyaknya order terdampak. Persentase memberi konteks saat volume antarwilayah berbeda. |
| Pisahkan data berdasarkan waktu | Melatih model dari masa lalu dan mengujinya pada Juni–Agustus 2018 meniru urutan penggunaan model. Order train harus sudah diterima sebelum 1 Juni agar labelnya tersedia saat training. |
| Fitur tersedia saat approval | Negara bagian, hari approval, dan selisih hari ke estimasi bisa dipakai saat memberi skor. Tanggal barang diterima dipakai untuk membuat label. |
| One-hot encoding dan standardisasi | One-hot encoding memberi kolom terpisah untuk kategori tanpa memaksakan urutan. Standardisasi menyetarakan skala fitur numerik. Pipeline scikit-learn memastikan keduanya di-fit memakai train. |
| Tuning regularisasi | Uji beberapa nilai `C` pada validasi April dan pilih berdasarkan average precision. Model default tetap ikut evaluasi untuk melihat apakah tuning memberi manfaat. |
| Logistic regression | Model awal yang sederhana untuk klasifikasi biner dan memberi skor untuk mengurutkan order. |
| Baseline historis per wilayah | Tim ops bisa memakai aturan ini tanpa melatih model. Model dinilai terhadap aturan ini pada kandidat dan kapasitas review yang sama. |
| Order telat ditemukan per hari | Metrik ini menjawab langsung apakah metode menemukan lebih banyak order yang akhirnya telat dengan kapasitas review yang sama. Precision antrean melaporkan bagian antrean yang memang akhirnya telat. Average precision merangkum kualitas ranking pada berbagai ambang skor. |

## Temuan EDA

EDA mencakup 27.162 order yang disetujui Januari–April 2018 dan punya label sebelum 1 Juni. Jumlah ini mencakup 97,1% order yang disetujui pada periode tersebut.

- **Maret perlu ditelusuri.** Ada 1.353 order telat (19,1%), dibanding 238 (3,6%) pada April. Pecah Maret per minggu dan wilayah untuk mempersempit investigasi.
- **Prioritaskan Rio de Janeiro.** Negara bagian RJ mencatat 734 order telat, sekitar seperempat dari total. Kota Rio de Janeiro menyumbang 335 kasus dan menjadi titik awal yang konkret untuk mengambil sampel.
- **Cek perjalanan setelah paket diserahkan ke kurir.** Di RJ, median waktu kurir ke tujuan adalah 34,0 hari pada order telat dan 10,4 hari pada order tepat waktu. Median approval ke kurir hanya berbeda sekitar 0,4 hari. Perbandingan ini memberi arah investigasi. Penyebabnya masih perlu diuji dengan data rute dan kurir.
- **Kasus berat cukup banyak.** Dari 2.873 order telat, 1.572 tiba minimal tujuh hari setelah estimasi.

## Hasil evaluasi

Pada 18.645 order uji Juni–Agustus, 672 order akhirnya telat. Masing-masing metode mengecek 1.724 order dengan kapasitas maksimal 20 review per hari.

| Metode | Order telat ditemukan | Telat ditemukan per hari | Precision antrean | Average precision |
| --- | ---: | ---: | ---: | ---: |
| Baseline tingkat telat per wilayah | 54 | 0,59 | 3,13% | 0,0382 |
| Logistic regression default | 115 | 1,25 | 6,67% | 0,0776 |
| Logistic regression tuned | 115 | 1,25 | 6,67% | 0,0777 |

Validasi memilih `C=10`. Di data test, kedua model menemukan jumlah order telat yang sama. Peningkatan dari 54 menjadi 115 kasus datang saat beralih dari aturan wilayah ke logistic regression. Tuning memberi selisih sangat kecil pada average precision dan hasil antreannya setara.

Model tuned dan preprocessing disimpan bersama sebagai `Notebook/late_delivery_model.joblib`. Input yang dibutuhkan adalah `state`, `weekday` dalam bahasa Inggris, dan `days_to_estimate`. Notebook memuat ulang file ini dan mencocokkan prediksinya dengan model sebelum disimpan.

### Cara membaca antrean review

Setiap hari, tim memprioritaskan maksimal 20 order dari skor risiko tertinggi. Status aktual baru digunakan setelah pengiriman untuk memeriksa hasilnya. Order yang akhirnya telat tetapi berada di luar antrean dihitung sebagai kasus terlewat.

Dari 672 order yang akhirnya telat, baseline menangkap 54 (recall 8,0%) dan melewatkan 618. Model default dan tuned masing-masing menangkap 115 (recall 17,1%) dan melewatkan 557. Jika melewatkan keterlambatan lebih merugikan, recall membantu menilai berapa banyak kasus yang masih luput. Precision menunjukkan berapa bagian dari antrean yang akhirnya telat.

## Simulasi antrean order berjalan

Replay Juni–Agustus mencakup 18.960 order yang baru disetujui pada periode tersebut. Pada akhir setiap hari, order yang belum tercatat tiba dipisahkan menjadi **sudah lewat estimasi** dan **masih berisiko**. Daftar risiko memakai maksimal 20, 50, atau 100 review per hari. Satu order memakai satu slot selama simulasi untuk setiap kebijakan.

Dari 672 kasus telat terkonfirmasi, 670 sempat memenuhi syarat untuk peringatan sebelum tenggat. Tabel ini menunjukkan jumlah kasus yang ditangkap dan persentasenya dari 670.

| Slot review per hari | Baseline wilayah | Model approval | Model + aturan kurir |
| ---: | ---: | ---: | ---: |
| 20 | 73 (10,9%) | 128 (19,1%) | 313 (46,7%) |
| 50 | 161 (24,0%) | 208 (31,0%) | 425 (63,4%) |
| 100 | 268 (40,0%) | 333 (49,7%) | 498 (74,3%) |

Aturan tambahan yang diuji memberi prioritas pada order dengan satu atau dua hari tersisa dan belum ada catatan serah ke kurir. Pada 20 slot per hari, aturan ini menangkap 185 kasus lebih banyak daripada model approval, tetapi juga memilih 105 order yang akhirnya batal atau tidak tersedia, dibanding 9 untuk model approval. Pada kapasitas yang sama, median waktu peringatan untuk kasus yang tertangkap adalah 2 hari sebelum tenggat dengan aturan kurir dan 14 hari dengan skor approval saja. Data hanya punya status akhir, jadi waktu pembatalan tidak bisa dipakai untuk membersihkan antrean saat replay.

Aturan dua hari masih eksploratif. Hasil ini menunjukkan manfaat potensial untuk memilih order yang perlu ditinjau. Penggunaan operasional memerlukan pembaruan status yang lengkap dan pengukuran dampak tindak lanjut terhadap keterlambatan aktual.

## Batas interpretasi

Evaluasi awal saat approval memakai order historis yang tercatat sudah diterima. Order batal, belum diterima, atau tanpa tanggal yang diperlukan berada di luar evaluasi, sehingga hasil bisa berubah saat semua order baru masuk antrean. Data historis juga belum mengukur apakah tindakan review dari tim operasional bisa mencegah keterlambatan.

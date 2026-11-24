# 💬 Analisis Sentimen Ulasan Aplikasi GoPay

Mengukur sentimen 5.000 ulasan pengguna GoPay dari Google Play Store — dan menemukan bahwa satu baris `dropna()` ternyata membuang 94,6% ulasan positif sebelum model sempat melihatnya.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK%20%2B%20Sastrawi-green)](https://www.nltk.org/)
[![Dataset](https://img.shields.io/badge/Data-Google%20Play%20Scraper-3DDC84?logo=googleplay&logoColor=white)](https://pypi.org/project/google-play-scraper/)

**📦 Dataset:** [`dataset_scaraping_5000.csv`](dataset_scaraping_5000.csv) (5.000 ulasan, tersedia di repo)

---

## 📌 Ringkasan Eksekutif

| Pertanyaan | Jawaban |
|---|---|
| Bagaimana sentimen pengguna GoPay? | **63,4% negatif**, 29,1% positif, 7,5% netral — keluhan mendominasi |
| Berapa akurasi model terbaik? | **87,46%** (SVM TF-IDF dan Random Forest Word2Vec, identik) |
| Seberapa baik sebenarnya? | **Buruk.** Baseline "selalu jawab negatif" sudah **87,03%** |
| Kelas mana yang gagal total? | **Positif dan netral** — F1 = **0,00** untuk keduanya |
| Apa penyebabnya? | `dropna()` membuang semua baris tanpa balasan developer — **94,6% ulasan positif** ikut terhapus |

> **Insight utama:** proyek ini punya satu kesalahan yang menentukan seluruh hasilnya. Setelah preprocessing, kelas positif menyusut dari **1.457 menjadi 78 baris** — hanya **5,4%** yang bertahan. Penyebabnya bukan preprocessing teks, melainkan `dropna()` yang dijalankan pada seluruh dataframe: kolom `replyContent` dan `repliedAt` kosong pada 94,6% ulasan positif (developer jarang membalas pujian), sehingga baris-baris itu ikut terbuang. Akibatnya model dilatih pada data yang secara artifisial menjadi 87,4% negatif, dan akurasi 87,46% yang dilaporkan **hanya 0,43 poin di atas** menebak "negatif" untuk semua ulasan.

---

## 🎯 Business Understanding

### Latar Belakang

GoPay adalah salah satu dompet digital dengan basis pengguna terbesar di Indonesia. Ulasan di Google Play Store adalah sumber umpan balik yang paling langsung dan paling murah — ribuan pengguna menulis keluhan dan pujian tanpa diminta. Tetapi volume sebesar ini tidak mungkin dibaca manual.

Analisis sentimen otomatis menjanjikan dua hal: **pemantauan kesehatan produk secara berkelanjutan**, dan **deteksi dini masalah** dengan mengelompokkan keluhan berdasarkan tema. Proyek ini membangun fondasi untuk itu, sekaligus menguji apakah fondasinya sudah cukup kuat.

### Problem Statement

1. Bagaimana distribusi sentimen ulasan pengguna GoPay di Google Play Store?
2. Algoritma dan representasi teks mana yang paling baik untuk klasifikasi sentimen berbahasa Indonesia?
3. Apakah model yang dibangun benar-benar menambah nilai di atas tebakan naif?

### Goals

- Mengumpulkan ulasan GoPay secara terprogram dari Google Play Store.
- Membangun pipeline NLP bahasa Indonesia: cleaning, normalisasi slang, stemming, stopword removal.
- Membandingkan **tiga kombinasi** representasi dan algoritma: TF-IDF+SVM, Word2Vec+Random Forest, TF-IDF+Logistic Regression.
- Mengevaluasi dengan metrik yang tahan terhadap ketidakseimbangan kelas.

### Metrik Keberhasilan

- **Metrik bisnis:** kemampuan mengangkat keluhan yang benar-benar baru dari lautan ulasan — bukan sekadar menandai semua ulasan sebagai negatif.
- **Metrik model:** **macro-F1** dan **recall per kelas**. Akurasi tidak dipakai sebagai metrik utama karena kelas negatif mendominasi 87%; baseline naif dicantumkan eksplisit sebagai pembanding.

### Cakupan

**Termasuk:** scraping 5.000 ulasan, EDA, preprocessing bahasa Indonesia (termasuk kamus slang), tiga kombinasi model, evaluasi per kelas, analisis penyebab kegagalan.
**Tidak termasuk:** topic modeling, analisis aspek, data ulasan App Store, data transaksi internal, deployment, model transformer (IndoBERT).

---

## 🗂️ Dataset

| Atribut | Keterangan |
|---|---|
| **Sumber** | Google Play Store — aplikasi **GoPay** (`com.gojek.gopay`) via `google-play-scraper` |
| **Metode** | `reviews_all()` dengan `sort=MOST_RELEVANT`, `lang='id'`, `country='id'`, diambil 5.000 teratas |
| **Jumlah baris** | **5.000** ulasan |
| **Kolom** | 11 — `reviewId`, `userName`, `userImage`, `content`, `score`, `thumbsUpCount`, `reviewCreatedVersion`, `at`, `replyContent`, `repliedAt`, `appVersion` |
| **Periode** | **8 Juni 2023 – 15 April 2025** (84,7% berasal dari 2025) |
| **Duplikat** | 0 pada `reviewId`, 1 pada teks |
| **Teks kosong** | 0 |

**Pelabelan sentimen** dilakukan dari `score` bintang:

| Skor | Sentimen | Jumlah | Proporsi |
|---|---|---:|---:|
| 1–2 | Negatif | **3.168** | **63,4%** |
| 3 | Netral | **375** | **7,5%** |
| 4–5 | Positif | **1.457** | **29,1%** |

Perhatikan bentuknya: bukan sekadar tidak seimbang, tapi **bimodal** — 54,6% ulasan memberi bintang 1, sementara 24,5% memberi bintang 5. Hanya 7,5% memilih bintang 3. Pola ini khas ulasan aplikasi: pengguna menulis ketika sangat marah atau sangat puas.

**Catatan kualitas data:**

- **`userName` dan `userImage` tidak informatif** — hanya satu nilai unik (`"Pengguna Google"`) untuk seluruh 5.000 baris, karena scraper tidak mengambil identitas reviewer.
- **`content` terpotong di 500 karakter** — 32 ulasan tepat berhenti di 500 karakter, menandakan batas dari API Play Store. Keluhan panjang bisa terpotong di tengah.
- **`replyContent` kosong pada 28,5% baris** (1.425 dari 5.000), dan distribusinya sangat timpang antar kelas — ini akar masalah seluruh proyek, dibahas di [Keterbatasan](#-keterbatasan--asumsi) poin 1.

---

## 🔬 Metodologi

```
Scraping → Pelabelan dari Score → Cleaning → Normalisasi Slang → Case Folding
→ Stemming Sastrawi → Tokenizing → Stopword Removal → Split 80:20
→ Representasi (TF-IDF / Word2Vec) → Modeling (SVM / RF / LR) → Evaluasi
```

1. **Scraping** — `reviews_all()` mengambil seluruh ulasan lalu dipotong 5.000 teratas berdasarkan relevansi. Disimpan ke CSV dengan `csv.writer`.
2. **Pelabelan** — `score ≥ 4` → positif, `score == 3` → netral, `score ≤ 2` → negatif.
3. **Cleaning** — hapus mention, hashtag, RT, URL, angka, emoji (library `emoji`), dan tanda baca.
4. **Normalisasi slang** — kamus slang Indonesia (~3.000 entri) dari [`louisowen6/NLP_bahasa_resources`](https://github.com/louisowen6/NLP_bahasa_resources), diterapkan per kata. Ini krusial untuk ulasan Play Store yang penuh singkatan (`gk`, `udh`, `bgt`, `gopay`).
5. **Case folding** — lowercase.
6. **Stemming** — Sastrawi, mereduksi kata ke bentuk dasar (`menggunakannya` → `guna`).
7. **Tokenizing + stopword removal** — NLTK `word_tokenize` + daftar stopword bahasa Indonesia.
8. **Deduplikasi** — hapus baris dengan `clean_text` identik.
9. **Split** — 80:20 stratified (`random_state=42`) → **2.774 train / 694 test**.
10. **Representasi & modeling** — tiga kombinasi:
    - **TF-IDF** (`max_features=3468`) → **SVM** (`kernel='linear'`, `C=1.0`)
    - **Word2Vec** (`vector_size=100`, `window=5`, `sg=1`) dirata-rata per dokumen → **Random Forest** (100 pohon)
    - **TF-IDF** → **Logistic Regression** (`max_iter=1000`)
11. **Evaluasi** — classification report per kelas, akurasi train vs test, dan pembandingan terhadap baseline naif.

**Kenapa tiga kombinasi?** TF-IDF menangkap kata kunci secara eksplisit dan cocok untuk SVM linear pada data sparse. Word2Vec menangkap kemiripan makna antar kata, sehingga Random Forest bisa mengenali ulasan yang memakai kata berbeda dengan maksud sama. Logistic Regression ditambahkan sebagai baseline linear yang mudah diinterpretasi. Perbandingannya justru mengungkap bahwa **ketiganya gagal dengan cara yang sama** — lihat [Hasil Model](#-hasil-model).

---

## 📊 Temuan Utama (EDA)

### 1. Sentimen didominasi keluhan

| Sentimen | Jumlah | Proporsi |
|---|---:|---:|
| Negatif (1–2) | 3.168 | **63,4%** |
| Positif (4–5) | 1.457 | 29,1% |
| Netral (3) | 375 | 7,5% |

Hampir dua pertiga ulasan bernada negatif. Bagi tim produk, ini bukan sekadar angka — dengan 5.000 ulasan "paling relevan", artinya keluhan adalah konten yang paling banyak di-*upvote* dan paling terlihat oleh calon pengguna baru.

### 2. Distribusi bimodal, bukan normal

Bintang 1 (54,6%) dan bintang 5 (24,5%) mendominasi; bintang 2, 3, dan 4 bersama-sama hanya 20,8%. Ini pola khas ulasan aplikasi dan punya konsekuensi metodologis: **kelas netral terlalu kecil (7,5%) untuk bisa dipelajari**, dan setiap model yang dioptimasi untuk akurasi akan mengabaikannya sepenuhnya.

### 3. Panjang ulasan berbeda sistematis antar kelas

| Sentimen | Rata-rata panjang | Median |
|---|---:|---:|
| Negatif | 196 karakter | 168 |
| Netral | 181 karakter | 149 |
| **Positif** | **130 karakter** | **102** |

Ulasan positif **jauh lebih pendek** daripada negatif. Ini masuk akal: pengguna yang puas menulis "mantap, lancar" dan pergi; yang kecewa menulis kronologi lengkap. Konsekuensinya, ulasan positif punya lebih sedikit token untuk dipelajari model — dan setelah stopword removal, sebagian menjadi sangat tipis. Ini memperparah masalah kelas minoritas.

---

## 🤖 Hasil Model

### Perbandingan performa

| Model | Accuracy Train | Accuracy Test | Macro-F1 | Selisih vs baseline |
|---|---:|---:|---:|---:|
| **SVM** (TF-IDF) | 0,8767 | **0,8746** | 0,31 | **+0,43 poin** |
| **Random Forest** (Word2Vec) | 1,0000 | **0,8746** | *tidak valid* | **+0,43 poin** |
| **Logistic Regression** (TF-IDF) | 0,8796 | 0,8689 | 0,31 | **−0,14 poin** |
| *Baseline "selalu negatif"* | — | *0,8703* | *0,23* | *—* |

**Tiga hal yang harus dibaca dari tabel ini:**

1. **Semua model praktis setara dengan baseline naif.** SVM dan Random Forest unggul 0,43 poin; Logistic Regression justru **kalah** 0,14 poin dari sekadar menebak "negatif" untuk semua ulasan.
2. **Random Forest mengalami overfitting ekstrem** — akurasi train **1,0000** sempurna, test 0,8746. Model menghafal data latih, bukan belajar pola.
3. **Macro-F1 hanya 0,31**, jauh dari akurasi 0,87. Jarak lebar ini adalah tanda pasti bahwa performa tinggi berasal dari kelas mayoritas saja.

> **Kenapa macro-F1 Random Forest ditandai tidak valid?** Notebook menghitungnya dengan `classification_report(y_pred_test_rf, y_pred_bal)` — membandingkan **dua vektor prediksi** satu sama lain, bukan prediksi dengan label sebenarnya. Angka 0,36 yang muncul di notebook tidak mengukur performa apa pun. Akurasi test Random Forest (0,8746) tetap valid karena dihitung dengan `accuracy_score(y_pred_test_rf, y_test_w2v)`.

### Classification report — di sinilah kegagalannya terlihat

**SVM (TF-IDF):**

| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negatif | 0,87 | 1,00 | 0,93 | 604 |
| **Netral** | **0,00** | **0,00** | **0,00** | 76 |
| **Positif** | **0,00** | **0,00** | **0,00** | 14 |
| **Macro avg** | **0,29** | **0,33** | **0,31** | 694 |

**Logistic Regression (TF-IDF):**

| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negatif | 0,87 | 1,00 | 0,93 | 604 |
| **Netral** | **0,00** | **0,00** | **0,00** | 76 |
| **Positif** | **0,00** | **0,00** | **0,00** | 14 |
| **Macro avg** | **0,29** | **0,33** | **0,31** | 694 |

**Cara membaca:** kedua model **tidak pernah sekali pun** memprediksi "netral" atau "positif". Recall 0,00 berarti dari 76 ulasan netral dan 14 ulasan positif di test set, **nol** yang berhasil dikenali. Model secara efektif menjadi **detektor biner**: negatif, atau bukan apa-apa. Untuk keperluan memahami sentimen pengguna, model ini tidak menghasilkan informasi apa pun di luar "mayoritas ulasan itu negatif" — yang sudah diketahui sebelum model dilatih.

### Analisis kegagalan: satu baris yang merusak semuanya

Perhatikan distribusi kelas sebelum dan sesudah preprocessing:

| Sentimen | Sebelum | Sesudah | Hilang | % Hilang |
|---|---:|---:|---:|---:|
| Negatif | 3.168 | 3.033 | 135 | **4,3%** |
| Netral | 375 | 357 | 18 | **4,8%** |
| **Positif** | **1.457** | **78** | **1.379** | **94,6%** |

Penghapusan 94,6% ulasan positif tidak mungkin disebabkan oleh cleaning teks — teks positif tetap berisi kata. Penyebabnya ada pada urutan operasi:

```python
app_reviews_df = app_reviews_df[app_reviews_df['clean_text'] != '']   # buang teks kosong
app_reviews_df = app_reviews_df.dropna()                              # <-- ini masalahnya
app_reviews_df = app_reviews_df.drop_duplicates(subset=['clean_text'])
```

`dropna()` tanpa argumen membuang baris yang punya nilai kosong di **kolom mana pun** — termasuk `replyContent` dan `repliedAt`. Dan di data ini, kolom balasan developer kosong secara sangat tidak acak:

| Sentimen | Ulasan tanpa balasan developer |
|---|---:|
| Negatif | 1,2% |
| Netral | 2,1% |
| **Positif** | **94,6%** |

Developer membalas hampir semua keluhan, tetapi hampir tidak pernah membalas pujian. Karena `dropna()` dijalankan **setelah** kolom `clean_text` dibuat tetapi **pada seluruh dataframe**, filter yang seharusnya hanya menyaring kualitas teks justru menyaring berdasarkan ada/tidaknya balasan. Hasilnya bukan sampel acak, melainkan **sampel yang bias ke arah keluhan** — dan model dilatih pada dunia yang sudah dipermutasi sebelum ia mulai belajar.

**Perbaikannya sederhana:** `dropna(subset=['clean_text'])`. Satu argumen itu akan mempertahankan 1.457 ulasan positif dan mengubah kesimpulan proyek secara menyeluruh.

**Catatan transparansi:** seluruh angka di atas diambil langsung dari output sel yang tersimpan di notebook dan diverifikasi ulang terhadap `dataset_scaraping_5000.csv` yang disertakan di repo ini.

---

## ⚠️ Keterbatasan & Asumsi

### 1. `dropna()` membuang 94,6% kelas positif — ini merusak validitas keseluruhan

Dibahas lengkap di atas. Konsekuensinya berantai:

- Distribusi kelas berubah dari 63:29:8 menjadi **87:10:2**, membuat kelas positif terlalu kecil untuk dipelajari (hanya 78 baris, 14 di antaranya masuk test set).
- Akurasi yang dilaporkan (87,46%) hampir seluruhnya berasal dari kelas mayoritas.
- Seluruh kesimpulan tentang performa model **tidak berlaku** untuk populasi ulasan yang sebenarnya.

### 2. SMOTE diterapkan pada label prediksi, bukan label sebenarnya

Pada cell Random Forest, baris ini:

```python
X_train_bal, y_train_bal = smote.fit_resample(X_train_w2v, y_pred_train_rf)
```

`y_pred_train_rf` adalah **hasil prediksi model pada data latih**, bukan `y_train_w2v` yang sebenarnya. SMOTE kemudian menyeimbangkan data menggunakan label yang sudah diprediksi — termasuk semua kesalahan prediksi train. Ini bukan hanya bug kecil; ia membuat oversampling mereplikasi bias model, bukan menyeimbangkan kelas nyata. Selain itu, hasilnya tidak pernah benar-benar dievaluasi: `rf_model_bal` dilatih pada data ini, tetapi classification report di bawahnya membandingkan **dua vektor prediksi** (`y_pred_test_rf` vs `y_pred_bal`), bukan prediksi dengan label sebenarnya.

### 3. SMOTE tidak cocok untuk data teks

Meskipun labelnya diperbaiki, SMOTE tetap bermasalah di sini. SMOTE menginterpolasi linear antar tetangga terdekat. Pada embedding Word2Vec, rata-rata dua ulasan bukan ulasan yang bermakna — ia vektor yang tidak mewakili kalimat apa pun. Untuk teks, `class_weight='balanced'` hampir selalu lebih tepat dan lebih dapat dijelaskan.

### 4. Label dari bintang bukan sentimen yang tervalidasi

Asumsi `score ≥ 4 = positif` adalah heuristik, bukan kebenaran. Ulasan bintang 1 sering berisi sarkasme atau keluhan tentang hal lain (misalnya kualitas jaringan), dan ulasan bintang 5 kadang berisi keluhan yang diselipkan. Tidak ada validasi manual pada sampel mana pun, sehingga akurasi label dasar **tidak diketahui** — dan model tidak bisa lebih baik dari kualitas labelnya.

### 5. Data hanya 5.000 ulasan "paling relevan", bukan sampel acak

`reviews_all()` dipotong 5.000 teratas berdasarkan `MOST_RELEVANT`. Ini bukan sampel acak dari seluruh ulasan GoPay — ini ulasan yang paling banyak di-*upvote*. Ulasan keluhan ekstrem cenderung mendapat lebih banyak upvote, sehingga dataset ini **kemungkinan melebih-lebihkan proporsi negatif**. Kesimpulan "63,4% pengguna tidak puas" tidak boleh digeneralisasi ke seluruh basis pengguna.

### 6. Konsentrasi temporal ekstrem

**84,7% ulasan berasal dari tahun 2025**, sementara 2023 hanya menyumbang 21 ulasan. Analisis ini praktis merupakan potret awal 2025, bukan tren dua tahun. Tidak ada validasi temporal, sehingga tidak diketahui apakah pola sentimen stabil antar periode.

### 7. Teks terpotong di 500 karakter

32 ulasan berhenti tepat di 500 karakter karena batas API Play Store. Untuk keluhan panjang yang berisi kronologi lengkap, bagian akhir — yang sering memuat inti masalah — hilang. Ini memengaruhi kelas negatif paling banyak, karena ulasan negatif adalah yang paling panjang.

### 8. `userName` dan `userImage` tidak bisa dipakai

Hanya ada satu nilai unik untuk seluruh 5.000 baris. Tidak ada analisis per pengguna, tidak ada deteksi ulasan berulang dari akun yang sama, dan tidak ada pembobotan berdasarkan reputasi reviewer.

### 9. Tidak ada validasi silang

Seluruh hasil berasal dari satu split 80:20. Tidak ada cross-validation, sehingga tidak ada estimasi variabilitas. Perbedaan 0,43 poin antara model dan baseline berada jauh di dalam rentang yang bisa muncul hanya dari pemilihan split.

### 10. Kamus slang diambil dengan `eval()` dari URL eksternal

```python
slangwords = eval(response.text)
```

`eval()` pada konten yang diunduh dari internet adalah risiko keamanan: jika URL tersebut dikompromikan, kode arbitrer akan dieksekusi. Untuk proyek portofolio dampaknya kecil, tetapi dalam pipeline produksi ini harus diganti `ast.literal_eval()` atau format data yang aman seperti JSON. Selain itu, kamus ini tidak di-*pin* versinya — isinya bisa berubah kapan saja dan membuat hasil tidak reproducible.

---

## 💡 Rekomendasi

### Untuk perbaikan model (prioritas utama, berurutan)

1. **Ganti `dropna()` menjadi `dropna(subset=['clean_text'])`.** Ini perbaikan satu baris dengan dampak terbesar — mempertahankan 1.457 ulasan positif dan mengubah distribusi kelas dari 87:10:2 menjadi 63:29:8.
2. **Perbaiki bug SMOTE:** gunakan `y_train` yang sebenarnya, bukan `y_pred_train_rf`. Lalu evaluasi dengan `classification_report(y_test, y_pred)`.
3. **Ganti SMOTE dengan `class_weight='balanced'`** pada SVM dan Logistic Regression, atau hentikan oversampling dan optimalkan threshold probabilitas. Untuk teks, ini lebih tepat dan jauh lebih mudah dijelaskan.
4. **Ubah metrik evaluasi utama menjadi macro-F1**, dan sertakan baseline naif di setiap tabel hasil. Tanpa baris baseline, akurasi 87% akan selalu terlihat baik padahal tidak berarti apa pun.
5. **Tambahkan cross-validation** (Stratified 5-fold) dengan `Pipeline` agar TF-IDF di-*fit* per fold — bukan sekali di seluruh data.
6. **Pertimbangkan IndoBERT.** Untuk teks berbahasa Indonesia, representasi kontekstual biasanya melampaui TF-IDF secara signifikan. Perbandingannya akan menunjukkan apakah batas performa di sini berasal dari metode atau dari data.

### Untuk implementasi (setelah model diperbaiki)

1. **Jangan pakai untuk keputusan otomatis.** Gunakan sebagai **prioritisasi antrean** — model menyortir ulasan yang perlu dibaca tim CX lebih dulu. Keputusan tetap pada manusia.
2. **Manfaatkan `thumbsUpCount` sebagai sinyal tambahan.** 2.046 ulasan (40,9%) punya upvote, maksimum 18.607. Ulasan dengan upvote tinggi adalah keluhan yang paling banyak disetujui pengguna lain — jauh lebih bernilai untuk ditindaklanjuti daripada ulasan acak. Fitur ini tersedia tetapi belum dipakai sama sekali.
3. **Pisahkan analisis per periode.** Dengan 84,7% data di 2025, pantau sentimen per bulan untuk mendeteksi lonjakan keluhan setelah rilis versi baru (`appVersion` tersedia, 29 versi berbeda).
4. **Bangun proses pelabelan yang tervalidasi.** Ambil 300 ulasan acak, minta dua orang melabeli secara independen, ukur kesepakatan antar-pelabel. Tanpa ini, tidak ada cara mengetahui seberapa akurat label bintang sebagai proksi sentimen.

### KPI yang Disarankan

| KPI | Baseline (test set) | Target |
|---|---|---|
| Macro-F1 | 0,31 | **≥ 0,70** |
| Recall kelas positif | **0,00** (dari 14 sampel) | **≥ 0,75** |
| Recall kelas netral | **0,00** (dari 76 sampel) | **≥ 0,60** |
| Akurasi di atas baseline naif | +0,43 poin | **+10 poin** |
| Jumlah ulasan positif setelah preprocessing | 78 (dari 1.457) | **≥ 1.300** |
| Selisih akurasi train vs test | 12,5 poin (RF) | **< 5 poin** |

---

## 🚀 Cara Menjalankan

### 1. Prasyarat

- Python **3.10+**

### 2. Install dependensi

```bash
pip install -r requirement.txt
```

### 3. Jalankan pipeline

**Langkah 1 — Scraping (opsional, dataset sudah tersedia):**

```bash
jupyter notebook file_kode_scraping.ipynb
```

> Menghasilkan `dataset_scaraping_5000.csv` dengan 5.000 ulasan terbaru dari Play Store. Karena datanya *live*, hasilnya akan berbeda dari CSV yang ada di repo ini.

**Langkah 2 — Pelatihan model:**

```bash
jupyter notebook file_pelatihan_model.ipynb
```

> **Catatan:** notebook menggunakan path absolut `/content/dataset_scaraping_5000.csv` (lingkungan Colab). Untuk eksekusi lokal, sesuaikan path atau jalankan langsung di [Google Colab](https://colab.research.google.com/). Notebook juga mengunduh resource NLTK (`punkt`, `punkt_tab`, `stopwords`) dan kamus slang dari GitHub pada saat pertama dijalankan.

---

## 🗂️ Struktur Proyek

```
indonesian-sentiment-analysis/
│
├── file_kode_scraping.ipynb      # Scraping ulasan GoPay dari Play Store
├── file_pelatihan_model.ipynb    # Pelabelan, preprocessing, 3 model, evaluasi
├── dataset_scaraping_5000.csv    # Dataset mentah (5.000 ulasan, 11 kolom)
├── requirement.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Kategori | Tools |
|---|---|
| Bahasa | Python 3.10+ |
| Scraping | google-play-scraper |
| Analisis data | pandas, NumPy |
| NLP | NLTK (tokenize, stopwords), Sastrawi (stemming), emoji, kamus slang |
| Representasi teks | TF-IDF (scikit-learn), Word2Vec (Gensim) |
| Machine learning | scikit-learn (SVM, Random Forest, Logistic Regression) |
| Penanganan imbalance | imbalanced-learn (SMOTE) |
| Environment | Jupyter / Google Colab |

---

## 👤 Author

**Zulfahmi M. Ardianto**
Laskar ID: A548YBM523

*Project ini merupakan bagian dari portfolio Data Analyst.*

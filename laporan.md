# Laporan Proyek Machine Learning Rekomendasi Buku Menggunakan Content Based Filtering - Muhammad Lutfi Hakim

## Project Overview

Pada proyek ini, sistem **Content-Based Filtering** dikembangkan untuk memberikan rekomendasi buku berdasarkan kesamaan konten seperti genre, deskripsi, penulis, dan judul buku. Rekomendasi berbasis konten adalah salah satu metode dalam sistem rekomendasi yang digunakan untuk membantu pengguna menemukan item yang relevan berdasarkan atribut atau fitur yang dimiliki oleh item tersebut. Dalam proyek ini, dua metode pengukuran kemiripan digunakan: **Cosine Similarity** dan **Jaccard Distance**, yang keduanya digunakan untuk menganalisis teks dari kolom yang menggabungkan informasi genre, deskripsi, penulis, dan judul buku.

**Mengapa masalah ini perlu diselesaikan?**
Rekomendasi buku yang baik dapat membantu pembaca menemukan buku yang sesuai dengan minat dan preferensi mereka tanpa harus melalui pencarian panjang di berbagai platform. Dengan sistem rekomendasi yang efektif, pembaca dapat lebih mudah menemukan buku yang ingin dibaca sesuai dengan preferensi pribadi mereka. Dalam proyek ini, kami menggunakan dua metode pengukuran kemiripan, yaitu **Cosine Similarity** dan **Jaccard Distance**, untuk memberikan hasil rekomendasi berbasis konten.

**Sumber Referensi:**

* Ricci, F., Rokach, L., & Shapira, B. (2011). *Introduction to Recommender Systems Handbook*. Springer.
* Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer Society.

---

## Business Understanding

### Problem Statements

* **Pernyataan Masalah 1**: Bagaimana cara memberikan rekomendasi buku yang relevan kepada pengguna berdasarkan kesamaan isi buku, seperti genre, deskripsi, penulis, dan judul buku?
* **Pernyataan Masalah 2**: Bagaimana cara meningkatkan kualitas rekomendasi buku dengan memperhitungkan tingkat kesamaan antara buku-buku berdasarkan berbagai atribut, agar buku yang lebih sesuai dengan preferensi pengguna dapat direkomendasikan dengan lebih baik?

### Goals

* **Goal 1**: Mengembangkan sistem rekomendasi buku yang dapat menyarankan buku berdasarkan kesamaan isi atau konten buku yang relevan dengan minat pengguna.
* **Goal 2**:  Meningkatkan akurasi rekomendasi dengan menggunakan metode untuk mengukur seberapa mirip buku yang satu dengan yang lainnya berdasarkan atribut-atribut penting seperti deskripsi, genre, dan penulis.

### Solution Approach

Untuk mencapai tujuan tersebut, dua pendekatan sistem rekomendasi digunakan:

1. **Cosine Similarity**: Menggunakan metode pengukuran kemiripan antara buku berdasarkan kesamaan keseluruhan konten buku. Ini melibatkan analisis teks dari kolom seperti genre, deskripsi, penulis, dan judul buku untuk melihat buku mana yang paling mirip dengan buku yang dicari oleh pengguna.
2. **Jaccard Distance**: Menggunakan metode yang lebih fokus pada kesamaan elemen spesifik dari teks, seperti kata kunci atau genre tertentu dalam deskripsi buku. Dengan pendekatan ini, kita menghitung kemiripan berdasarkan kemiripan genre dan kata yang muncul dalam deskripsi buku.

---

## Data Understanding

Dataset yang digunakan berisi informasi mengenai buku-buku dengan fitur-fitur berikut:

* **Book**: Nama buku.
* **Author**: Nama penulis buku.
* **Description**: Deskripsi atau sinopsis buku.
* **Genres**: Genre atau kategori buku.
* **Avg\_Rating**: Rata-rata rating buku.
* **Num\_Ratings**: Jumlah ulasan buku.

**Jumlah Data dan Kondisi Data:**
Dataset yang digunakan berisi **10.000 entri** dan **8 kolom**. Berikut adalah kondisi spesifik dari dataset:

1. **Jumlah Missing Value**: Terdapat **77 missing values** pada kolom **'Description'**.
2. **Jumlah Duplikat**: Tidak ditemukan data duplikat (**0 duplikat**).
3. **Kolom yang Perlu Diperhatikan**:
  * **Description**: Kolom ini harus ditangani dengan mengisi missing value dengan value **No Description** pada data yang kosong.
  * **Num\_Ratings**: Kolom ini memiliki tipe data **object** (seharusnya **float64**) karena mengandung koma yang perlu dihapus dan dikonversi menjadi tipe numerik.
  * **Genres**: Kolom ini berisi string dengan beberapa genre yang digabungkan dalam format yang memerlukan pengolahan untuk mengubahnya menjadi list.


**Sumber Dataset:**
[Best Books 10k Multi Genre Data - Kaggle](https://www.kaggle.com/datasets/ishikajohari/best-books-10k-multi-genre-data)

---

## Data Preparation

Pada bagian ini, berikut adalah tahapan **data preparation** yang dilakukan:

* **Menghapus Kolom yang Tidak Diperlukan**: Kolom **`Unnamed: 0`** dan **`URL`** dihapus karena tidak relevan untuk analisis.
* **Mengonversi `Num_Ratings` ke Tipe Numerik**: Koma dalam **`Num_Ratings`** dihapus untuk memastikan angka bisa dihitung dengan benar, kemudian mengonversi kolom tersebut menjadi **`float64`**.
* **Mengisi Missing Value di Kolom `Description`**: Nilai kosong di kolom **`Description`** diisi dengan string **"No Description"** untuk menghindari missing values yang bisa mengganggu model.
* **Mengonversi `Genres` Menjadi List**: Kolom **`Genres`** yang berbentuk string diubah menjadi list agar lebih mudah diproses dan digunakan dalam model.
* **Membersihkan dan Menggabungkan Kolom**: Kolom **`Genres`**, **`Description`**, **`Author`**, dan **`Book`** digabungkan menjadi satu kolom **`combined`** yang digunakan sebagai input untuk model rekomendasi berbasis konten.

**Alasan Data Preparation:**

* **Menghapus kolom yang tidak relevan** membantu mengurangi dimensi dataset dan mempercepat pemrosesan data.
* **Mengonversi `Num_Ratings`** ke tipe numerik memungkinkan kita melakukan perhitungan, seperti pengurutan berdasarkan jumlah rating.
* **Menangani nilai kosong di `Description`** penting agar model tidak terkendala dengan missing values.
* **Mengonversi `Genres` menjadi list** memastikan kolom tersebut dapat digunakan untuk menghitung kemiripan antar buku dengan akurat.
* **Menggabungkan kolom `Genres`, `Description`, `Author`, dan `Book`** menjadi satu kolom **`combined`** memungkinkan model menggunakan informasi komprehensif untuk menghitung kemiripan antar buku.

---

## Modeling

### 1. **Cosine Similarity**:

* **TF-IDF Vectorization** digunakan untuk mengonversi teks dari kolom **`combined`** menjadi representasi vektor numerik.

* **Cosine Similarity** dihitung untuk mengukur kemiripan antar buku berdasarkan teks yang telah terkonversi.

* Rumus **Cosine Similarity**:

  $$
  \text{Cosine Similarity} = \frac{A \cdot B}{\|A\| \|B\|}
  $$

  Di mana **A** dan **B** adalah dua vektor teks yang dihitung menggunakan **TF-IDF**.

* **Fungsi `recommend_books_cosine`** mengembalikan **top-5 rekomendasi buku** yang paling mirip dengan query yang diberikan.

**Hasil yang diperoleh :** 

Rekomendasi yang diberikan berdasarkan **Cosine Similarity** menunjukkan buku yang memiliki kesamaan **kontekstual** dalam hal deskripsi, genre, dan penulis.

Buku yang direkomendasikan:

* **Virtual Light** oleh William Gibson
* **Forever Peace** oleh Joe Haldeman
* **Dear and Glorious Physician** oleh Taylor Caldwell
* **The Enigma of Arrival** oleh V.S. Naipaul
* **Collected Fiction** oleh Ruskin Bond

**Rating dan Popularitas**: Buku yang lebih banyak di-review (seperti **Virtual Light**) lebih tinggi dalam rekomendasi karena pengurutan berdasarkan **`Num_Ratings`**.


### 2. **Jaccard Distance**:

* **CountVectorizer** digunakan untuk mengonversi teks menjadi representasi biner.

* **Jaccard Distance** dihitung antara buku dan query untuk mengukur kesamaan antar buku berdasarkan elemen spesifik dalam deskripsi dan genre.

* Rumus **Jaccard Distance**:

  $$
  \text{Jaccard Distance} = 1 - \frac{|A \cap B|}{|A \cup B|}
  $$

  Di mana **A** dan **B** adalah dua set kata dalam deskripsi buku yang dibandingkan.

* **Fungsi `recommend_books_jaccard`** mengembalikan **top-5 rekomendasi buku** berdasarkan Jaccard Distance.

**Hasil yang Diperoleh:**

Rekomendasi yang diberikan berdasarkan **Jaccard Distance** menunjukkan buku yang memiliki kesamaan **kata kunci** atau elemen **spesifik** dalam deskripsi dan genre.

Buku yang direkomendasikan:

* **The Shipping News** oleh E. Annie Proulx
* **SHADOW PANTHEON** oleh Eric Nierstedt
* **A Taste for Green Tangerines** oleh Barbara Bisco
* **Making Amends** oleh D.J. Callaghan
* **Chasing Dreams** oleh Aaron Jennings

**Rating dan Popularitas**: Buku dengan rating tinggi (misalnya **The Shipping News**) lebih tinggi dalam rekomendasi karena pengurutan berdasarkan **`Num_Ratings`**.


**Kelebihan dan Kekurangan:**

* **Cosine Similarity**:

  * **Kelebihan:** Mampu menangkap **kesamaan konteks keseluruhan** dari teks buku.
  * **Kekurangan:** Mungkin kurang efektif jika terdapat perbedaan kecil dalam deskripsi atau genre meskipun ada kesamaan konteks.
* **Jaccard Distance**:

  * **Kelebihan:** Lebih menekankan pada **kesamaan elemen spesifik** dalam deskripsi dan genre buku, cocok untuk buku dengan **kesamaan kata kunci**.
  * **Kekurangan:** Mungkin kurang efektif untuk menangkap konteks keseluruhan dari teks.

---

## Evaluation

Metrik yang digunakan untuk evaluasi adalah **Precision at K**, yang mengukur seberapa banyak buku yang relevan (misalnya, dengan rating tinggi) muncul dalam **top-5 rekomendasi**.

**Precision at K:**

$$
\text{Precision at K} = \frac{\text{Number of relevant items in K}}{\text{Total number of items in K}}
$$

Hasil evaluasi:

* **Precision at K for Cosine Similarity: 0.4**
* **Precision at K for Jaccard Distance: 0.6**

**Penjelasan Lebih Lanjut:**

1. **Precision at K for Cosine Similarity (0.4):**

   * **Interpretasi:** Hanya **40%** dari **top-5 rekomendasi** berdasarkan **Cosine Similarity** yang relevan dengan preferensi pengguna (misalnya, buku dengan rating lebih tinggi atau genre yang sesuai).
   * **Kemungkinan Penyebab:**

     * **Cosine Similarity** mengukur **kemiripan konteks** secara keseluruhan, tetapi terkadang bisa memberi hasil yang kurang relevan jika **konteks teksnya mirip** namun tidak berhubungan dengan **preferensi rating tinggi** atau **genre** tertentu.
     * Ini bisa terjadi jika **relevansi** yang didasarkan pada **rating tinggi** atau **genre yang cocok** tidak cukup berhubungan dengan teks buku (misalnya, genre atau deskripsi).

2. **Precision at K for Jaccard Distance (0.6):**

   * **Interpretasi:** **60%** dari **top-5 rekomendasi** berdasarkan **Jaccard Distance** relevan dengan preferensi pengguna (misalnya, buku dengan rating lebih tinggi atau genre yang sesuai).
   * **Kemungkinan Penyebab:**

     * **Jaccard Distance** lebih fokus pada **kesamaan elemen spesifik** (kata atau genre), yang membuatnya lebih **berorientasi pada kesamaan langsung** dalam deskripsi buku.
     * Karena **Jaccard Distance** mengukur **kesamaan kata atau genre**, sistem ini mungkin lebih efektif dalam menangkap **kesamaan yang lebih langsung** dengan **preferensi pengguna** yang lebih sederhana, seperti **rating tinggi** dan genre.



**Implikasi dari Hasil Evaluasi:**

1. **Cosine Similarity:**

   * **Kelebihan:** Cosine Similarity cocok ketika kita ingin memberikan rekomendasi berdasarkan **kesamaan konteks keseluruhan** (deskripsi, genre, penulis, dll). Namun, jika pengguna lebih tertarik pada buku dengan **rating tinggi** atau **genre tertentu**, sistem ini mungkin tidak seefektif Jaccard Distance.
   * **Peningkatan:** Kamu bisa **menambahkan bobot** untuk genre atau rating agar sistem lebih menekankan pada buku dengan **rating lebih tinggi** dan **kesesuaian genre**.

2. **Jaccard Distance:**

   * **Kelebihan:** Jaccard lebih menekankan pada **kesamaan elemen langsung**, seperti kata kunci atau genre dalam deskripsi buku. Dengan demikian, **precision** lebih tinggi karena rekomendasi lebih relevan dengan **kesamaan kata** yang spesifik.
   * **Peningkatan:** Agar hasil lebih baik, kamu bisa **memperkenalkan bobot** berdasarkan **rating** atau **jumlah rating**, memastikan buku yang lebih populer dan relevan tetap mendapatkan prioritas meskipun mungkin tidak memiliki kesamaan teks yang terlalu tinggi.


**Kesimpulan:**

* **Jaccard Distance** memberikan hasil yang lebih baik dalam hal relevansi rekomendasi, namun kedua metode ini dapat digabungkan atau disesuaikan lebih lanjut untuk meningkatkan kualitas rekomendasi.
* **Jaccard Distance** memberikan hasil yang lebih baik dalam hal **Precision at K**, karena ia lebih fokus pada **kesamaan kata atau genre** yang lebih jelas dan langsung.
* **Cosine Similarity** meskipun bermanfaat untuk memahami **kesamaan konteks keseluruhan**, mungkin tidak cukup efektif dalam menghasilkan **rekomendasi yang relevan** jika kita lebih mengutamakan **rating tinggi** atau **genre tertentu**.

---
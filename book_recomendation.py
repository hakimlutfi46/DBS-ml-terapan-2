# Import Data
import pandas as pd
import seaborn as sns
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load Data
df = pd.read_csv('goodreads_data.csv')


# Data Understanding
df.head()
df.shape
df.info()
df.isnull().sum()
df.duplicated().sum()
df['Genres'].unique()


# Data Preparation

# Menghapus kolom yang tidak diperlukan
df_cleaned = df.drop(columns=['Unnamed: 0', 'URL'])

# Menghapus koma pada kolom 'Num_Ratings' (jika ada)
df_cleaned['Num_Ratings'] = df_cleaned['Num_Ratings'].str.replace(',', '')

# Mengonversi kolom 'Num_Ratings' menjadi float64
df_cleaned['Num_Ratings'] = pd.to_numeric(df_cleaned['Num_Ratings'], errors='coerce')

# Mengisi nilai kosong di kolom 'Description' dengan string kosong
df_cleaned['Description'].fillna('No Description', inplace=True)

# Mengonversi kolom Genres yang berupa string menjadi list jika perlu
df_cleaned['Genres'] = df_cleaned['Genres'].apply(lambda x: x.strip("[]").replace("'", "").split(',') if isinstance(x, str) else x)

# Membersihkan genre untuk memastikan kesesuaian format (menggabungkan list menjadi string)
df_cleaned['Genres'] = df_cleaned['Genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

# Menggabungkan kolom 'Genres', 'Description', 'Author', dan 'Book' menjadi satu kolom teks
df_cleaned['combined'] = df_cleaned['Genres'] + ' ' + df_cleaned['Description'] + ' ' + df_cleaned['Author'] + ' ' + df_cleaned['Book']

# Melihat data yang telah digabungkan
print(df_cleaned[['Book', 'combined']].head())

# Feature Engineering

# Menggunakan TfidfVectorizer untuk menghitung representasi TF-IDF dari teks yang telah digabungkan
tfidf = TfidfVectorizer(stop_words='english')

# Menghitung matriks TF-IDF
tfidf_matrix = tfidf.fit_transform(df_cleaned['combined'])

# Menghitung cosine similarity antar buku menggunakan TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Menggunakan CountVectorizer untuk menghitung representasi biner dari teks (deskripsi buku)
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(df_cleaned['combined'])


# Modeling

# Cosine Similarity

# Fungsi untuk rekomendasi berdasarkan Cosine Similarity
def recommend_books_cosine(query, cosine_sim=cosine_sim, top_n=5):
    query_tfidf = tfidf.transform([query])
    cosine_sim_query = cosine_similarity(query_tfidf, tfidf_matrix)
    
    sim_scores = list(enumerate(cosine_sim_query[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[0:top_n]
    book_indices = [i[0] for i in sim_scores]
    
    recommended_books = df_cleaned[['Book', 'Author', 'Genres', 'Avg_Rating', 'Num_Ratings']].iloc[book_indices]
    recommended_books['Num_Ratings'] = recommended_books['Num_Ratings'].apply(pd.to_numeric, errors='coerce')
    recommended_books = recommended_books.sort_values(by='Num_Ratings', ascending=False)
    
    return recommended_books.head(top_n)

# Menggunakan contoh query
recommended_books_cosine = recommend_books_cosine('fiction')
print("Top 5 recommended books based on Cosine Similarity:")
print(recommended_books_cosine)


# Jacard Distance

# Fungsi untuk menghitung Jaccard Distance
def jaccard_distance(vec1, vec2):
    intersection = np.sum(np.minimum(vec1, vec2))  # Jumlah kata yang ada di kedua buku
    union = np.sum(np.maximum(vec1, vec2))  # Jumlah kata yang ada di salah satu buku
    return 1 - intersection / union  # Jaccard Distance

# Fungsi untuk rekomendasi berdasarkan Jaccard Distance
def recommend_books_jaccard(query, count_matrix=count_matrix, top_n=5):
    query_vec = count_vectorizer.transform([query]).toarray().flatten()  # Mengubah query menjadi vektor biner
    jaccard_scores = []
    
    # Menghitung Jaccard Distance antara query dan setiap buku dalam dataset
    for i in range(count_matrix.shape[0]):
        book_vec = count_matrix[i].toarray().flatten()
        score = jaccard_distance(query_vec, book_vec)
        jaccard_scores.append((i, score))
    
    # Mengurutkan berdasarkan skor Jaccard Distance (semakin kecil nilai, semakin mirip)
    jaccard_scores = sorted(jaccard_scores, key=lambda x: x[1])
    
    # Memilih top_n buku teratas
    top_books_indices = [i[0] for i in jaccard_scores[:top_n]]
    recommended_books = df_cleaned[['Book', 'Author', 'Genres', 'Avg_Rating', 'Num_Ratings']].iloc[top_books_indices]
    recommended_books['Num_Ratings'] = recommended_books['Num_Ratings'].apply(pd.to_numeric, errors='coerce')
    recommended_books = recommended_books.sort_values(by='Num_Ratings', ascending=False)
    
    return recommended_books.head(top_n)

# Menggunakan contoh query
recommended_books_jaccard = recommend_books_jaccard('fiction')
print("Top 5 recommended books based on Jaccard Distance:")
print(recommended_books_jaccard)


# Evaluation

# Fungsi untuk menghitung Precision at K
def precision_at_k(recommended_books, relevant_books, k):
    """
    Menghitung Precision at K
    - recommended_books: Daftar buku yang direkomendasikan
    - relevant_books: Daftar buku relevan yang sesuai dengan preferensi pengguna
    - k: Jumlah buku yang dipertimbangkan dalam rekomendasi
    """
    recommended_at_k = recommended_books[:k]
    relevant_at_k = [book for book in recommended_at_k if book in relevant_books]
    precision = len(relevant_at_k) / k
    return precision

# Misalnya, relevan buku adalah buku dengan rating tinggi (> 4)
relevant_books_cosine = recommended_books_cosine[recommended_books_cosine['Avg_Rating'] > 4]['Book'].tolist()
relevant_books_jaccard = recommended_books_jaccard[recommended_books_jaccard['Avg_Rating'] > 4]['Book'].tolist()

# Tentukan k (jumlah rekomendasi)
k = 5

# Menghitung Precision at K untuk Cosine Similarity
precision_cosine = precision_at_k(recommended_books_cosine['Book'].tolist(), relevant_books_cosine, k)
print(f"Precision at K for Cosine Similarity: {precision_cosine}")

# Menghitung Precision at K untuk Jaccard Distance
precision_jaccard = precision_at_k(recommended_books_jaccard['Book'].tolist(), relevant_books_jaccard, k)
print(f"Precision at K for Jaccard Distance: {precision_jaccard}")
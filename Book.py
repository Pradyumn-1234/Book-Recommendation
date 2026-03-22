import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches


# -------- CSS --------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#667eea,#764ba2);
    color: white;
}

h1 {
    text-align:center;
    font-size:50px;
    font-weight:bold;
}

.stButton>button {
    background: linear-gradient(45deg,#ff416c,#ff4b2b);
    color:white;
    border-radius:20px;
    height:3em;
    width:200px;
    font-size:18px;
}

.stTextInput>div>div>input {
    border-radius:10px;
    padding:8px;
}

</style>
""", unsafe_allow_html=True)


st.title("📚 Book Recommendation System")

# Load dataset
df = pd.read_csv("book.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.write("Data Size:", df.size)
st.write("Shape:", df.shape)

# Select required columns
final_cols = [
    'book_id',
    'title',
    'authors',
    'original_publication_year',
    'language_code',
    'average_rating',
    'description'
]

df = df[final_cols]
df = df.drop(columns=['Unnamed: 0', 'image_url'], errors='ignore')

# Graphs
st.subheader("📊 Data Visualizations")

fig1, ax1 = plt.subplots()
sns.histplot(df['average_rating'], bins=20, kde=True, ax=ax1)
st.pyplot(fig1)


# -------- Heatmap  --------
st.subheader("Heatmap Correlation")
fig2, ax2 = plt.subplots()

sns.heatmap(
    df[['original_publication_year', 'average_rating']].corr(),
    annot=True,
    cmap='coolwarm',
    ax=ax2
)

st.pyplot(fig2)


# Data cleaning
df['title'] = df['title'].astype(str)
df['title'] = df['title'].str.replace(r"\s*\(.*?\)", "", regex=True)
df['title'] = df['title'].str.strip()

df = df[['title', 'authors', 'original_publication_year',
         'average_rating', 'description']].dropna()

df.columns = ['title', 'author', 'year', 'rating', 'description']
df['year'] = df['year'].astype(int)

# TFIDF
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(
    df['title'] + " " + df['description']
)

similarity = cosine_similarity(tfidf_matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=df['title'],
    columns=df['title']
)

# Recommendation function
def recommend_book(book_name):

    book_name = book_name.lower()

    book = df[df['title'].str.lower().str.contains(book_name)]

    if book.empty:

        words = book_name.split()

        for w in words:
            book = df[df['title'].str.lower().str.contains(w)]
            if not book.empty:
                break

    if book.empty:
        st.error("Book not found")
        return

    book_index = book.index[0]

    scores = list(enumerate(similarity[book_index]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    similar_books = scores[1:15]

    books = df.iloc[[i[0] for i in similar_books]]

    books = books.sort_values(by='rating', ascending=False)

    top_books = books.head(5)

    st.subheader("📚 Recommended Books")

    for i in range(len(top_books)):

        title = top_books.iloc[i]['title']
        rating = top_books.iloc[i]['rating']
        year = top_books.iloc[i]['year']
        author = top_books.iloc[i]['author']
        desc = top_books.iloc[i]['description']

        st.markdown(f"### 📘 {title}")
        st.write("⭐ Rating:", rating)
        st.write("📅 Year:", year)
        st.write("✍ Author:", author)
        st.write("📝 Description:", desc)
        st.write("----------------------------------")


# -------- User Input --------
st.subheader("🔎 Search Book")

user_input = st.text_input("Enter Book Name")

if st.button("Recommend"):
    recommend_book(user_input)
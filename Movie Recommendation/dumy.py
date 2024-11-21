import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

movies = pd.read_csv('movie_data.csv')
movies = movies[['Movie Name', 'Genre', 'Description']]
movies.dropna(inplace=True)

ps = PorterStemmer()

def preprocess(text):
    tokens = text.split()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

movies['Genre'] = movies['Genre'].apply(preprocess)
movies['Description'] = movies['Description'].apply(preprocess)

tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_vectors = tfidf_vectorizer.fit_transform(movies['Genre'] + ' ' + movies['Description'])

def recommend_with_plot_summary(query, k=5):
    query_vector = tfidf_vectorizer.transform([preprocess(query)])
    similarity = cosine_similarity(query_vector, tfidf_vectors)[0]
    indices = np.argsort(similarity)[-k:][::-1]
    recommended_movies = movies.iloc[indices]['Movie Name'].tolist()
    return recommended_movies

recommendations = recommend_with_plot_summary('''Mini isn't eager to wed the rich suitor whos been chosen for her, so she stages her own kidnapping and runs off with a man who works for her father.''')
print(recommendations)

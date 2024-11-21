import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 

import nltk
from nltk.stem.porter import PorterStemmer 


# Load the movie data
movies = pd.read_csv('movie_data.csv')

# Select relevant columns
movies = movies[['Movie Name', 'Genre', 'Description']]

# Handle missing values
movies.isnull().sum()
movies.dropna(inplace=True)

# Convert genre to lists of words (Corrected)
def convert(obj):
    L = []
    for i in obj.split(', '):  # Split the string by ', '
        L.append(i.replace(" ", ""))  # Remove spaces
    return L

movies['Genre'] = movies['Genre'].apply(convert) 

# Preprocess the data
def preprocess(text):
    text = text.split()
    text = [i.replace(" ", "") for i in text]
    return text

movies['Description'] = movies['Description'].apply(preprocess)
movies['tags'] = movies['Genre'] + movies['Description']
# Create a new dataframe with movie name and tags
new_df = movies[['Movie Name', 'tags']]
new_df = movies[['Movie Name', 'tags']].copy()  # Create an explicit copy

new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Apply stemming
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return ' '.join(y)

new_df['tags'] = new_df['tags'].apply(stem)

# Create a CountVectorizer object
cv = CountVectorizer(max_features=5000, stop_words='english')

# Fit and transform the tags
vectors = cv.fit_transform(new_df['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vectors)

# Create a recommendation function
def recommend(movie):
    movie_index = new_df[new_df['Movie Name'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    a = []
    for i in movies_list:
        a.append(new_df.iloc[i[0]]['Movie Name'])
    return a

# Get movie recommendations (uncomment to use)
# a = input('Movie name: ')
# recommendations = recommend(a)
# print(recommendations)
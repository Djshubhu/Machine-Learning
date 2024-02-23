import numpy as np
import pandas as pd
import seaborn as sns
movies=pd.read_csv('movie_data.csv')

movies=movies[['Movie Name','Genre','Description']]
movies.isnull().sum()
movies.dropna(inplace=True)
movies.shape
import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
# movies['Genre']=movies['Genre'].apply(convert)
# movies['Description']=movies['Description'].apply(convert)
# movies.head()
movies['Genre']=movies['Genre'].apply(lambda x:x.split())
movies['Description']=movies['Description'].apply(lambda x:x.split())
movies['Genre']=movies['Genre'].apply(lambda x:[i.replace(" ","")for i in x])
movies['Description']=movies['Description'].apply(lambda x:[i.replace(" ","")for i in x])
movies['tags']=movies['Genre']+movies['Description']

new_df=movies[['Movie Name','tags']]
new_df['tags']=new_df['tags'].apply(lambda x:' '.join(x))
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
new_df['tags']
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')
cv.fit_transform(new_df['tags']).toarray().shape
vectors=cv.fit_transform(new_df['tags']).toarray()
vectors[0]
(cv.get_feature_names_out())
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return ' '.join(y)
new_df['tags']=new_df['tags'].apply(stem)
# import re
# def extract_alpha(input_str):
#     return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', input_str)
# for i in range(0,935):
#     new_df['Movie Name'][i]=extract_alpha(new_df['Movie Name'][i])
# new_df['Movie_Name']=new_df['Movie Name']
new_df['Movie_Name']=new_df['Movie Name']
new_df['Movie_Name'][3]
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(vectors)
similarity=cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]
def recommed(movie):
    movie_index = new_df[new_df['Movie_Name'] == movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].Movie_Name)
a=input("Movie Name ")
recommed(a)
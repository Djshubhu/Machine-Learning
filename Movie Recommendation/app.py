from flask import Flask, render_template, request
from Movie import recommend,new_df
import pandas as pd
import random


app = Flask(__name__)
movies = new_df['Movie Name'].tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        random_movies = random.sample(movies, 3)
    elif request.method == 'POST':
        movie_name = request.form['movie_name']
        recommendations = recommend(movie_name)
        return render_template('result.html', movie_name=movie_name, recommendations=recommendations)
    
    return render_template('index.html', random_movies=random_movies)



if __name__ == '__main__':
    app.run(debug=True)

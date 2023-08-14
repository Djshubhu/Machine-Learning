from flask import Flask, render_template, request
from models import get_prediction  # Import your prediction function from model.py

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image uploaded"
        
        image = request.files['image']
        if image.filename == '':
            return "No image selected"

        prediction = get_prediction(image)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
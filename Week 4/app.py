from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load the model
with open('linear_model_.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract feature values from the form
        medinc = float(request.form['medinc'])
        houseage = float(request.form['houseage'])
        averooms = float(request.form['averooms'])
        avebedrms = float(request.form['avebedrms'])
        population = float(request.form['population'])
        aveoccup = float(request.form['aveoccup'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])

        features = [medinc, houseage, averooms, avebedrms, population, aveoccup, latitude, longitude]
        prediction = model.predict([features])[0]  # Predict expects a 2D array, predict first instance
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

    return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction*100000:.2f}')

if __name__ == "__main__":
    app.run(debug=True)

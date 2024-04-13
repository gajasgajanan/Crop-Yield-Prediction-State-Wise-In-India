from flask import Flask, request, render_template # type: ignore
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler # type: ignore

# Load models and preprocessors
dtr = pickle.load(open('./dtree_model', 'rb'))
preprocessor = pickle.load(open('./pp', 'rb'))

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extracting features from the form
        State_Name = request.form['State_Name']
        District_Name = request.form['District_Name']
        Crop_Year = int(request.form['Crop_Year'])
        Season = request.form['Season']
        Crop = request.form['Crop']
        Area = float(request.form['Area'])

        # Creating feature array
        features = np.array([[State_Name, District_Name, Crop_Year, Season, Crop, Area]])

        # Preprocessing the input features
        transformed_features = preprocessor.transform(features)

        # Making predictions
        prediction = dtr.predict(transformed_features)

        # Reshaping prediction if necessary
        prediction = prediction.reshape(-1, 1)

        return render_template('index.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
import json
import pickle

from flask import Flask, request, app, jsonify, url_for, render_template, redirect
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        # Handle manual input
        data = [float(x) for x in request.form.values() if x]
        if len(data) == 10:  # Ensure all 10 features are provided
            final_input = scalar.transform(np.array(data).reshape(1, -1))
            output = regmodel.predict(final_input)[0]
            return render_template("home.html", prediction_text="The House price prediction is {}".format(output))
        else:
            return render_template("home.html", prediction_text="Please provide all 10 features or upload a CSV file.")
    else:
        # Handle file input
        file = request.files['file']
        if file.filename == '':
            return render_template("home.html", prediction_text="No file selected for uploading")

        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            # Ensure this path exists on your server
            filepath = os.path.join('/tmp', filename)
            file.save(filepath)
            data = pd.read_csv(filepath)
            # Ensure the CSV contains the required columns
            required_columns = ['t3', 'rh_5', 't8', 'rh_6', 't2',
                                'press_mm_hg', 'rh_1', 'rh_8', 'rh_out', 'lights']
            if all(column in data.columns for column in required_columns):
                model_test_X = data[required_columns]
                model_test_X_scaled = scalar.transform(model_test_X)
                predictions = regmodel.predict(model_test_X_scaled)
                data['predicted_appliances'] = predictions
                output_filepath = os.path.join('/tmp', 'predictions.csv')
                data.to_csv(output_filepath, index=False)
                return render_template("home.html", prediction_text="Predictions saved to predictions.csv")
            else:
                return render_template("home.html", prediction_text="CSV file does not contain the required columns")
        else:
            return render_template("home.html", prediction_text="Please upload a valid CSV file")


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model and expected columns
model = joblib.load('income_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = pd.read_csv('model_columns.csv', header=None)[0].tolist()


# Load column names (used for encoding)
data_columns = pd.read_csv("data.csv")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input from form
        input_data = {
            'age': int(request.form['age']),
            'workclass': request.form['workclass'],
            'fnlwgt': int(request.form['fnlwgt']),
            'education': request.form['education'],
            'educational-num': int(request.form['educational-num']),
            'marital-status': request.form['marital-status'],
            'occupation': request.form['occupation'],
            'relationship': request.form['relationship'],
            'race': request.form['race'],
            'gender': request.form['gender'],
            'capital-gain': int(request.form['capital-gain']),
            'capital-loss': int(request.form['capital-loss']),
            'hours-per-week': int(request.form['hours-per-week']),
            'native-country': request.form['native-country']
        }

        df_input = pd.DataFrame([input_data])

        # One-hot encode & match model's training columns
        df_encoded = pd.get_dummies(df_input)
        # Align with training columns
        df_encoded = df_encoded.reindex(columns=expected_columns, fill_value=0)

        # Scale
        scaled_input = scaler.transform(df_encoded)

        # Predict
        prediction = model.predict(scaled_input)[0]
        predicted_label = '>50K' if prediction == 1 else '<=50K'

        return render_template('result.html', prediction=predicted_label, user_input=input_data)

@app.route('/save', methods=['POST'])
def save():
    # Save user input into the dataset
    input_data = request.form.to_dict()
    df = pd.read_csv('data.csv')
    new_data = pd.DataFrame([input_data])
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv('data.csv', index=False)
    return render_template('thankyou.html')

if __name__ == '__main__':
    app.run(debug=True)

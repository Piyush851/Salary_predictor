from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model, scaler, and column structure
model = joblib.load('salary_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = pd.read_csv('model_columns.csv', header=None).squeeze().tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        age = float(request.form['age'])
        gender = request.form['gender']
        education = request.form['education']
        job_title = request.form['job_title']
        experience = float(request.form['experience'])

        # Create dictionary with proper column names
        input_dict = {
            'Age': age,
            'Years of Experience': experience,
            'Gender': gender,
            'Education Level': education,
            'Job Title': job_title
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Apply one-hot encoding (same as training)
        input_df = pd.get_dummies(input_df)

        # Reindex to match training columns
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Scale the input
        input_df_scaled = scaler.transform(input_df)

        # Predict the salary
        predicted_salary = model.predict(input_df_scaled)[0]

        # Render result
        return render_template('result.html', salary=round(predicted_salary, 2))

    except Exception as e:
        return f"Prediction error: {e}"

if __name__ == '__main__':
    app.run(debug=True)

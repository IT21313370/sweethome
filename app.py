from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('best_model_001.pkl')  

# Render the HTML form on the home page
@app.route('/')
def home():
    return render_template('index.html')

# Function to preprocess input data and make predictions
def predict_house_price(data):
    input_features = [
        float(data['longitude']), float(data['latitude']), float(data['housing_median_age']),
        float(data['total_rooms']), float(data['total_bedrooms']), float(data['population']),
        float(data['households']), float(data['median_income']),
        int(data['ocean_proximity'])
    ]

    # Make prediction
    predicted_price = model.predict(np.array(input_features).reshape(1, -1))
    return predicted_price[0]

# Handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        predicted_price = predict_house_price(data)
        return jsonify({'predicted_house_price': f'${predicted_price:.2f}'})  # Sending JSON response

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import joblib
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model, scaler, and label encoders
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        app.logger.info("Model loaded successfully")

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = joblib.load(scaler_file)
        app.logger.info("Scaler loaded successfully")

    with open('label_encoders.pkl', 'rb') as le_file:
        label_encoders = pickle.load(le_file)
        app.logger.info("Label encoders loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading model, scaler, or label encoders: {e}")
    raise e

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        app.logger.info('Received data: %s', data)

        # Validate input data
        if not data or 'features' not in data:
            raise ValueError("Invalid input data: 'features' key not found.")

        features = data['features']
        app.logger.info('Raw features: %s', features)

        # Validate the length of features
        if len(features) != 5:
            raise ValueError("Invalid number of features provided.")

        # Extract and encode features
        institute_id, city, state, ranking_category, rank = features

        if institute_id not in label_encoders['institute_id'].classes_:
            raise ValueError(f"Institute ID '{institute_id}' not recognized.")
        if city not in label_encoders['city'].classes_:
            raise ValueError(f"City '{city}' not recognized.")
        if state not in label_encoders['state'].classes_:
            raise ValueError(f"State '{state}' not recognized.")
        if ranking_category not in label_encoders['ranking_category'].classes_:
            raise ValueError(f"Ranking Category '{ranking_category}' not recognized.")

        institute_id_encoded = label_encoders['institute_id'].transform([institute_id])[0]
        city_encoded = label_encoders['city'].transform([city])[0]
        state_encoded = label_encoders['state'].transform([state])[0]
        ranking_category_encoded = label_encoders['ranking_category'].transform([ranking_category])[0]

        # Ensure rank is an integer
        try:
            rank = int(rank)
        except ValueError:
            raise ValueError("Rank should be an integer.")

        # Convert features to numpy array and reshape
        features_array = np.array([institute_id_encoded, city_encoded, state_encoded, ranking_category_encoded, rank]).reshape(1, -1)
        app.logger.info('Features array: %s', features_array)

        # Standardize the features
        features_scaled = scaler.transform(features_array)
        app.logger.info('Transformed features: %s', features_scaled)

        # Make prediction using loaded model
        prediction = model.predict(features_scaled)
        app.logger.info('Prediction: %s', prediction)

        # Convert prediction to a more user-friendly format
        rounded_prediction = round(prediction[0], 2)  # Round to 2 decimal places

        return jsonify({'prediction': rounded_prediction})
    except ValueError as ve:
        error_message = f"Value Error: {str(ve)}"
        app.logger.error(error_message)
        return jsonify({'error': error_message}), 400
    except Exception as e:
        error_message = f"Error during prediction: {str(e)}"
        app.logger.error(error_message)
        return jsonify({'error': error_message}), 500

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

if __name__ == '__main__':
    app.run(debug=True,port=5001, use_reloader=False)

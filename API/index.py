from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Load preprocessor and model
with open('assets/preprocesser.pkl', 'rb') as f:
    preprocesser = pickle.load(f)

with open('assets/dtr_model.pkl', 'rb') as f:
    dtr = pickle.load(f)

@app.route('/')
def home():
    return "Welcome to the Crop Yield Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve JSON data from the request
        data = request.get_json()
        
        # Convert JSON data to DataFrame
        features = pd.DataFrame([data])
        
        # Preprocess the data
        transformed_features = preprocesser.transform(features)
        
        # Predict using the trained model
        prediction = dtr.predict(transformed_features).reshape(-1, 1)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction[0][0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

from flask import Flask, request, render_template
import pandas as pd
import pickle
import sys
import os
import numpy as np

# Add the root directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)

# Load models and preprocessor
def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

lr_model = load_model('models/linear_regression_model.pkl')
knn_model = load_model('models/knn_regressor_model.pkl')
rfc_model = load_model('models/random_forest_regressor_model.pkl')
preprocessor = load_model('models/preprocessor.pkl')

# Mean Absolute Errors for the models
mae_lr = 272.22350475281047
mae_knn = 246.9068862745098
mae_rfc = 165.6973155816993

# Compute weights based on MAE
weights = {
    'Linear Regression': 1 / mae_lr,
    'KNN Regression': 1 / mae_knn,
    'Random Forest': 1 / mae_rfc
}

# Normalize weights
total_weight = sum(weights.values())
normalized_weights = {model: weight / total_weight for model, weight in weights.items()}

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded.', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file.', 400
    
    data = pd.read_csv(file)
    print("Data loaded:")
    print(data.head())  # Print the first few rows for debugging

    try:
        # Preprocess data using the loaded preprocessor
        X = preprocessor.transform(data)
        print(f"Processed feature shape: {X.shape}")  # Debugging output to check shape
        
        # Make predictions
        lr_prediction = lr_model.predict(X)
        knn_prediction = knn_model.predict(X)
        rfc_prediction = rfc_model.predict(X)

        # Compute weighted average of predictions
        weighted_avg_prediction = (
            normalized_weights['Linear Regression'] * lr_prediction +
            normalized_weights['KNN Regression'] * knn_prediction +
            normalized_weights['Random Forest'] * rfc_prediction
        )
        
        # Convert to list for rendering
        weighted_avg_prediction_list = weighted_avg_prediction.tolist()

        # Generate HTML with index and prediction values
        prediction_html = "<ul>"
        for index, value in enumerate(weighted_avg_prediction_list):
            prediction_html += f"<li>Laptop {index}: {value}</li>"
        prediction_html += "</ul>"

    except Exception as e:
        return f"Error in making predictions: {e}", 500
    
    return f"""
        Predicted Laptop Prices (Euro):<br>
        {prediction_html}
    """

if __name__ == '__main__':
    app.run(debug=True)

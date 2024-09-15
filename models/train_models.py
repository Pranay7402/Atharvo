import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import os
import sys

# Ensure the scripts directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from preprocessing import preprocess_data

# Define the file path
data_path = r'.\data\laptop_price - dataset.csv'

# Load your dataset
if not os.path.exists(data_path):
    raise FileNotFoundError(f"The file at {data_path} does not exist.")
    
data = pd.read_csv(data_path)


#high correlation with Weigth and GPU_company but lower with Prices (Euro)
data = data.drop(['Inches','GPU_Type'],axis = 1)

X = data.drop('Price (Euro)', axis=1)
y = data['Price (Euro)']

# Load preprocessor
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Preprocess the features
X_processed = preprocess_data(data, preprocessor)

columns = data.columns[:-1]
# Convert to DataFrame
X = pd.DataFrame(X_processed, columns=columns)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train models
def create_model_pipeline(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

# Create models
lr_model = create_model_pipeline(LinearRegression())
knn_model = create_model_pipeline(KNeighborsRegressor())
rfc_model = create_model_pipeline(RandomForestRegressor())

# Fit models
lr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
rfc_model.fit(X_train, y_train)

# Save models to pickle files
with open('models/linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('models/knn_regressor_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)

with open('models/random_forest_regressor_model.pkl', 'wb') as f:
    pickle.dump(rfc_model, f)

print("Models and preprocessor have been trained and saved successfully.")

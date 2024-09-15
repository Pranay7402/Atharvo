import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(data, preprocessor):
    # Drop the target column if it's there (useful if preprocessing data from different sources)
    print(type(data))
    if 'Price (Euro)' in data.columns:
        data = data.drop('Price (Euro)', axis=1)
    # Transform the data using the preprocessor
    X = preprocessor.transform(data)
    print(f"Processed feature shape: {X.shape}")  # Print shape to debug
    return X

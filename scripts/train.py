from preprocessing import preprocess_data
from models import train_linear_regression, train_knn_regressor, train_random_forest_regressor
from evaluation import evaluate_model
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def main():
    data_path = r'.\data\laptop_price - dataset.csv'
    data = pd.read_csv(data_path)
    
    #high correlation with Weigth and GPU_company but lower with Prices (Euro)
    data = data.drop(['Inches','GPU_Type'],axis = 1)

    # Define features and target
    X = data.drop('Price (Euro)', axis=1)
    y = data['Price (Euro)']

    # Load preprocessor
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    # Preprocess the features
    X_processed = preprocess_data(X, preprocessor)

    columns = data.columns[:-1]
    # Convert to DataFrame
    X = pd.DataFrame(X_processed, columns=columns)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train.head(5))
    
    # Train and evaluate Linear Regression
    lr_model = train_linear_regression(X_train, y_train)
    print("Linear Regression Model:")
    evaluate_model(lr_model, X_test, y_test)
    
    # Train and evaluate KNN Regressor
    knn_model = train_knn_regressor(X_train, y_train)
    print("KNN Regression Model:")
    evaluate_model(knn_model, X_test, y_test)
    
    # Train and evaluate Random Forest Regressor
    rfc_model = train_random_forest_regressor(X_train, y_train)
    print("Random Forest Regression Model:")
    evaluate_model(rfc_model, X_test, y_test)

if __name__ == '__main__':
    main()

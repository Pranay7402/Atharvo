from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    with open('models/linear_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

def train_knn_regressor(X_train, y_train):
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    with open('models/knn_regressor_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

def train_random_forest_regressor(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    with open('models/random_forest_regressor_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

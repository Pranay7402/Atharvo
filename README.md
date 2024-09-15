# Laptop Price Prediction

This project aims to predict laptop prices using multiple regression models. The dataset used for this project is the [Laptop Price - dataset](https://www.kaggle.com/datasets/ironwolf404/laptop-price-dataset/data) provided by Iron Wolf on Kaggle.

## Project Overview

The project includes the following components:
- Data preprocessing
- Model training and evaluation
- Model predictions with a Flask web application

## Data

The dataset used in this project contains various features related to laptops, and the target variable is the price in Euros. 

**Dataset:** [Laptop Price - dataset](https://www.kaggle.com/datasets/ironwolf404/laptop-price-dataset/data)

## Models Used

1. **Linear Regression**
2. **K-Nearest Neighbors (KNN) Regression**
3. **Random Forest Regression**

## Evaluation Metrics

The performance of each model is evaluated using the following metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared (R²)

### Evaluation Results

- **Linear Regression Model:**
  - Mean Absolute Error: 272.22
  - Mean Squared Error: 151,366.90
  - R-squared: 0.695

- **KNN Regression Model:**
  - Mean Absolute Error: 246.91
  - Mean Squared Error: 154,095.97
  - R-squared: 0.690

- **Random Forest Regression Model:**
  - Mean Absolute Error: 165.70
  - Mean Squared Error: 64,670.43
  - R-squared: 0.870

## Web Application

A Flask web application is used to make predictions. The application allows users to upload a CSV file containing laptop features and receive predictions from the trained models. The predictions are combined using a weighted average based on the Mean Absolute Error (MAE) of each model.

### How to Run the Flask Application

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Pranay7402/Atharvo.git
   cd Atharvo
  
2. **Install Dependencies:**

Make sure you have Flask and other required packages installed. You can install them using pip:

   ```bash
   pip install -r requirements.txt
  ```
3. **Run the Flask Application:**

Start the Flask application with the following command:
```bash
   python app.py
```
The application will start running on http://127.0.0.1:5000/.

4. **Access the Web Interface:**

Open a web browser and navigate to http://127.0.0.1:5000/. You will see an interface where you can upload a CSV file and get predictions.

## File Structure
**app.py:** Flask application script.
**models/:** Directory containing the saved models and preprocessor.
**data/:** Directory where the dataset is located.
**templates/:** Directory containing HTML templates for the web application.
**requirements.txt:** List of required Python packages.

"""
Module for preprocessing and visualizing supermarket sales data.
This script includes functionality to clean, transform, and prepare the dataset for analysis
or machine learning tasks, along with visualizations for exploratory data analysis (EDA).
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
def train_and_predict_model(input_file, model_file='random_forest_regressor.pkl'):
    """
    Trains a RandomForestRegressor on the given supermarket dataset,
    saves the model, and makes predictions on the test set.
    
    Parameters:
    input_file (str): Path to the input Excel file containing the supermarket dataset.
    model_file (str): Path to save the trained model (default is 'random_forest_regressor.pkl').
    
    Returns:
    dict: A dictionary containing the model's evaluation metrics (MSE and R²), 
    the model file path, and a sample prediction.
    """
    # Step 1: Load Data
    df = pd.read_excel(input_file)
    # Step 2: Clean column names by stripping any spaces or unwanted characters
    df.columns = df.columns.str.strip()
    # Step 3: Convert Date and Create DateTime Column
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = df['Time'].astype(str)  # Convert Time to string format
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])
    df.drop(columns=['Date', 'Time'], inplace=True)
    # Step 4: Handle Missing Values
    df.fillna(0, inplace=True)
    # Step 5: Encode Categorical Variables
    label_encoders = {}
    categorical_columns = ['Customer type', 'Gender', 'Payment']
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    # Step 6: Feature Engineering: Date-Time Features
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['Day'] = df['DateTime'].dt.day
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Hour'] = df['DateTime'].dt.hour
    # Step 7: Add Additional Columns
    df['RevenuePerUnit'] = df['Unit price'] * df['Quantity']
    df['TotalTaxAmount'] = df['Total'] * 0.05
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df.drop(columns=['DateTime'], inplace=True)
    # Step 8: Log Transformation
    df['LogQuantity'] = np.log1p(df['Quantity'])
    df['LogTotal'] = np.log1p(df['Total'])
    df['LogGrossIncome'] = np.log1p(df['gross income'])
    # Step 9: Interaction Feature: Quantity * Unit Price
    df['QuantityUnitPrice'] = df['Quantity'] * df['Unit price']
    # Step 10: Outlier Treatment for 'Total'
    q1 = df['Total'].quantile(0.25)
    q3 = df['Total'].quantile(0.75)
    iqr = q3 - q1
    df = df[(df['Total'] >= q1 - 1.5 * iqr) & (df['Total'] <= q3 + 1.5 * iqr)]
    # Step 11: Scaling Numerical Features (Exclude 'Product line', 'City', 'Branch')
    numerical_features = ['Unit price', 'Quantity', 'Tax 5%', 'gross income', 'Rating',
    'RevenuePerUnit','TotalTaxAmount', 'LogQuantity', 'LogTotal', 'LogGrossIncome',
    'QuantityUnitPrice','IsWeekend', 'Year', 'Month', 'Day', 'DayOfWeek', 'Hour']
    # Ensure that all categorical columns have been encoded properly
    df[numerical_features] = df[numerical_features].apply(pd.to_numeric, errors='coerce')
    # Step 12: Check for Non-Numeric Columns in X
    x = df.drop(columns=['Total'])  # Features
    y = df['Total']  # Target
    # Check for any non-numeric columns in X
    non_numeric_columns = x.select_dtypes(include=['object']).columns
    if len(non_numeric_columns) > 0:
        # Handle these columns, e.g., encode or drop them.
        x = x.drop(columns=non_numeric_columns)  # Drop non-numeric columns for now
    # Step 13: Split Data into Training and Testing Sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Step 14: Scale the Data (Standardize)
    scaler = StandardScaler()
    x_train[numerical_features] = scaler.fit_transform(x_train[numerical_features])
    x_test[numerical_features] = scaler.transform(x_test[numerical_features])
    # Step 15: Train Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(x_train, y_train)
    # Step 16: Make Predictions
    y_pred = rf_regressor.predict(x_test)
    # Step 17: Evaluate the Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Step 18: Save the Trained Model
    joblib.dump(rf_regressor, model_file)
    # Step 19: Example: Load and Use the Saved Model
    loaded_model = joblib.load(model_file)
    new_data = x_test.iloc[0:1]  # Example: Use the first row of the test set
    new_prediction = loaded_model.predict(new_data)
    # Return the model evaluation metrics and prediction results
    return {
        'Mean Squared Error': mse,
        'R-squared': r2,
        'Model Path': model_file,
        'Prediction': new_prediction
    }
def hyperparameter_tuning(file_path,output_file='profits.csv',model_file='random_forest_model.pkl'):
    """
    Perform sales forecasting and profit optimization using machine learning.
    Args:
        file_path (str): Path to the input Excel file containing sales data.
        output_file (str): Path to save the output CSV file with new columns. 
        Default is 'profits.csv'.
        model_file (str): Path to save the trained model as a .pkl file. 
        Default is 'random_forest_model.pkl'.
    Returns:
        None: Outputs evaluation metrics, saves processed data and trained model.
    """
    # Step 1: Load Data
    df = pd.read_excel(file_path)
    df = df.sample(frac=0.1, random_state=42)  # Use 10% of the data for faster processing
    # Step 2: Preprocess Data
    df['Date'] = pd.to_datetime(df['Date'])
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df.drop(columns=['Date', 'Time', 'Invoice ID'], inplace=True)
    # Label Encoding
    categorical_columns = ['Customer type', 'Gender', 'Payment', 'City', 'Branch', 'Product line']
    for column in categorical_columns:
        df[column] = LabelEncoder().fit_transform(df[column])
    # Feature Engineering
    df['RevenuePerUnit'] = df['Unit price'] * df['Quantity']
    df['LogTotal'] = np.log1p(df['Total'])
    # Step 2.1: Calculate Profit Margin
    df['ProfitMargin'] = (df['Total'] - df['cogs']) / df['Total']
    df['ProfitMargin'] = df['ProfitMargin'].fillna(0)  # Handle any NaN values
    # Step 2.2: Calculate Optimized Sales
    df['OptimizedSales']=df['Total']+(df['Total']*df['ProfitMargin'])
    # Prepare Features and Target
    x = df[['Unit price', 'Quantity', 'Tax 5%', 'Rating',
    'RevenuePerUnit', 'ProfitMargin', 'OptimizedSales']]
    y = df['LogTotal']
    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Scale Data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    # Step 3: Hyperparameter Tuning
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=10,  # Reduced iterations
        cv=3,       # Reduced folds
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(x_train, y_train)
    best_rf = random_search.best_estimator_
    # Step 4: Model Evaluation
    y_pred = best_rf.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    y_train_pred = best_rf.predict(x_train)
    y_test_pred = best_rf.predict(x_test)
    # Calculate Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred) * 100  # Convert R² to percentage
    test_r2 = r2_score(y_test, y_test_pred) * 100  # Convert R² to percentage
    print("Model Performance:")
    print(f"Training Mean Squared Error: {train_mse:.4f}")
    print(f"Testing Mean Squared Error: {test_mse:.4f}")
    print(f"Training Accuracy: {train_r2:.2f}%")
    print(f"Testing Accuracy: {test_r2:.2f}%")
    # Step 5: Learning Curve
    def plot_learning_curve(estimator, x, y):
        """
        Plot the learning curve for the given model and data.
        """
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, x, y, cv=3, scoring='neg_mean_squared_error',
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
        )
        train_scores_mean = -train_scores.mean(axis=1)
        test_scores_mean = -test_scores.mean(axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, label="Training Error", marker='o')
        plt.plot(train_sizes, test_scores_mean, label="Validation Error", marker='o')
        plt.title("Learning Curve")
        plt.xlabel("Training Set Size")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.grid()
        plt.show()
    # Call the learning curve plot
    plot_learning_curve(best_rf, x_train, y_train)
    # Performance Analysis - Predictions and Residuals Plot
    def plot_predictions_and_residuals(y_test, y_pred):
        """
        Plots the Actual vs Predicted values, Residual Plot, and Residual Distribution.
        """
        # Plot the Predictions vs Actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
        plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red',linestyle='--')
        plt.title("Actual vs Predicted")
        plt.xlabel("Actual Log Total")
        plt.ylabel("Predicted Log Total")
        plt.grid(True)
        plt.show()
        # Calculate residuals
        residuals = y_test - y_pred
        # Plot residuals
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, color='blue')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()
        # Residuals Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, color='blue', edgecolor='black', alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.show()
    # Define y_pred using the best model
    y_pred = best_rf.predict(x_test)
    # Call the predictions and residuals plot function
    plot_predictions_and_residuals(y_test, y_pred)
    # Step 6: Save the dataset with new columns
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    # Step 7: Save the trained model to a .pkl file
    joblib.dump(best_rf, model_file)
    print(f"Model saved to {model_file}")
    # Step 8: (Optional) Load the model back to verify
    loaded_model = joblib.load(model_file)
    y_pred_loaded = loaded_model.predict(x_test)
    print(f"Loaded model MSE: {mean_squared_error(y_test, y_pred_loaded)}")

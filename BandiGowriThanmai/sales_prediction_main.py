"""
Module for preprocessing and visualizing supermarket sales data.
This script includes functionality to clean, transform, and prepare the dataset for analysis
or machine learning tasks, along with visualizations for exploratory data analysis (EDA).
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from model_training_performance_analysis import train_and_predict_model,hyperparameter_tuning
def preprocess_sales_data(input_path, output_path):
    """
    Preprocess the supermarket sales dataset.

    Parameters:
        input_path (str): Path to the input Excel file.
        output_path (str): Path to save the preprocessed Excel file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
        dict: Dictionary of LabelEncoders for categorical columns.
    """
    # Step 1: Load the Excel file
    df = pd.read_excel(input_path)
    # Step 2: Convert Date and Create DateTime Column
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = df['Time'].astype(str)  # Convert Time to string format
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])
    df.drop(columns=['Date', 'Time'], inplace=True)
    # Step 3: Handle Missing Values
    df.fillna(0, inplace=True)
    # Step 4: Encode Categorical Variables (Exclude 'Product line', 'City', 'Branch')
    label_encoders = {}
    categorical_columns = ['Customer type', 'Gender', 'Payment']
    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        label_encoders[column] = encoder
    # Step 5: Feature Engineering: Date-Time Features
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['Day'] = df['DateTime'].dt.day
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Hour'] = df['DateTime'].dt.hour
    # Step 6: Add Additional Columns
    df['RevenuePerUnit'] = df['Unit price'] * df['Quantity']
    df['TotalTaxAmount'] = df['Total'] * 0.05
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df.drop(columns=['DateTime'], inplace=True)
    # Step 7: Log Transformation
    df['LogQuantity'] = np.log1p(df['Quantity'])
    df['LogTotal'] = np.log1p(df['Total'])
    df['LogGrossIncome'] = np.log1p(df['gross income'])
    # Step 8: Interaction Feature: Quantity * Unit Price
    df['QuantityUnitPrice'] = df['Quantity'] * df['Unit price']
    # Outlier Treatment for 'Total'
    q1 = df['Total'].quantile(0.25)
    q3 = df['Total'].quantile(0.75)
    iqr = q3 - q1
    df = df[(df['Total'] >= q1 - 1.5 * iqr) & (df['Total'] <= q3 + 1.5 * iqr)]
    # Step 9: Scaling Numerical Features (Exclude 'Product line', 'City', 'Branch')
    numerical_features = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'gross income', 'Rating']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    # Step 10: Save Preprocessed Data to Excel
    df.to_excel(output_path, index=False)
    return df, label_encoders
def visualize_sales_data(input_path):
    """
    Generate visualizations for supermarket sales data.
    Parameters:
        input_path (str): Path to the input Excel file.
    """
    # Step 1: Load the Excel file
    df = pd.read_excel(input_path)
    # Step 2: Ensure 'Total' and 'Product line' columns exist
    if 'Total' not in df.columns or 'Product line' not in df.columns:
        raise ValueError("The input dataset must contain 'Total' and 'Product line' columns.")
    # Step 3: Sales Distribution by Product Line
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Product line', y='Total', data=df, estimator=sum, ci=None)
    plt.title('Total Sales by Product Line')
    plt.xticks(rotation=45)
    plt.ylabel('Total Sales')
    plt.show()
    # Step 4: Sales Trends Over Time
    df['Month'] = pd.to_datetime(df['Date']).dt.month  # Extract month from the 'Date' column
    sales_trend = df.groupby('Month')['Total'].sum()
    sales_trend.plot(kind='line', figsize=(10, 6), marker='o')
    plt.title('Monthly Sales Trend')
    plt.ylabel('Total Sales')
    plt.xlabel('Month')
    plt.show()
    # Step 5: Branch/City-Wise Sales
    branch_sales = df.groupby('Branch')['Total'].sum()
    city_sales = df.groupby('City')['Total'].sum()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    branch_sales.plot(kind='bar', color=['blue', 'green', 'orange'])
    plt.title('Total Sales by Branch')
    plt.ylabel('Total Sales')
    plt.xlabel('Branch')
    plt.subplot(1, 2, 2)
    city_sales.plot(kind='bar', color=['blue', 'green', 'orange'])
    plt.title('Total Sales by City')
    plt.ylabel('Total Sales')
    plt.xlabel('City')
    plt.show()
    # Step 6: Quantity vs. Total Sales
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Quantity', y='Total', data=df, alpha=0.7)
    plt.title('Quantity vs. Total Sales')
    plt.xlabel('Quantity Sold')
    plt.ylabel('Total Sales')
    plt.show()
    # Step 7: Payment Method Distribution
    payment_counts = df['Payment'].value_counts()
    payment_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
    plt.title('Payment Method Distribution')
    plt.ylabel('')
    plt.show()
    # Step 8: Heatmap of Correlations
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()
# Example usage
if __name__ == "__main__":
    INPUT_FILE = 'supermarket dataset.xlsx'
    OUTPUT_FILE = 'preprocessed_sales_data.xlsx'
    MODEL_FILE = 'random_forest_regressor.pkl'
    preprocessed_data, encoders = preprocess_sales_data(INPUT_FILE, OUTPUT_FILE)
    print("Preprocessing complete. Preprocessed data saved to", OUTPUT_FILE)
    print("Generating visualizations...")
    visualize_sales_data(INPUT_FILE)
    result=train_and_predict_model(INPUT_FILE)
    print(result)
    print("RESULTS AFTER HYPERPARAMETER TUNING")
    hyperparameter_tuning(INPUT_FILE)

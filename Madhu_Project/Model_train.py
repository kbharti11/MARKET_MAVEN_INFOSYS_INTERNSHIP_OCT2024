import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Function to load and preprocess the data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    st.write("Data Loaded Successfully")
    st.write(df.head())
    
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # Handle missing values by dropping rows
    df = df.dropna()
    
    # Encode categorical columns
    categorical_columns = ['Category', 'Sub Category', 'City', 'Region', 'State']
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    # Create new feature: Profit Margin
    df['Profit Margin'] = df['Profit'] / df['Sales']
    
    return df

# Function to train the model
def train_model(df):
    X = df[['Sales', 'Discount', 'Profit Margin', 'Category', 'Sub Category', 'City', 'Region', 'State']]
    y = df['Profit']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f'Mean Absolute Error: {mae}')
    st.write(f'Mean Squared Error: {mse}')
    st.write(f'R-squared: {r2}')
    
    # Save the model
    joblib.dump(model, 'random_forest_model.pkl')
    
    return model, y_test, y_pred

# Function to display visualizations
def show_visualizations(df, y_test, y_pred):
    # Correlation Heatmap
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(10, 6))
    numerical_df = df.select_dtypes(include=np.number)
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, ax=ax)
    st.pyplot(fig)

    # Profit Distribution
    st.subheader('Profit Distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Profit'], kde=True, color='blue', ax=ax)
    ax.set_title('Profit Distribution')
    st.pyplot(fig)

    # Sales vs Profit
    st.subheader('Sales vs Profit')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Sales', y='Profit', data=df, ax=ax)
    ax.set_title('Sales vs Profit')
    st.pyplot(fig)

    # Discount vs Profit
    st.subheader('Discount vs Profit')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Discount', y='Profit', data=df, ax=ax)
    ax.set_title('Discount vs Profit')
    st.pyplot(fig)

    # Actual vs Predicted Profit
    st.subheader('Actual vs Predicted Profit')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(range(len(y_test)), y_test, label='Actual Profit', color='blue', alpha=0.6)
    ax.scatter(range(len(y_pred)), y_pred, label='Predicted Profit', color='orange', alpha=0.6)
    ax.set_title('Actual vs Predicted Profit')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Profit')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')
    st.pyplot(fig)

# Streamlit Interface
def main():
    st.title("Supermarket Profit Prediction and Analysis")
    
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load and preprocess the data
        df = load_and_preprocess_data(uploaded_file)
        
        # Train model
        model, y_test, y_pred = train_model(df)
        
        # Display visualizations
        show_visualizations(df, y_test, y_pred)
        
        st.sidebar.subheader("Recommendations")
        st.sidebar.write("1. Adjust Discounts: Lower discounts if they are negatively impacting profits.")
        st.sidebar.write("2. Focus on High Profit Categories: Target categories and regions that yield higher profits.")
        st.sidebar.write("3. Improve Profit Margins: Look for segments where profit margins are low and work on improving them.")
        
        st.sidebar.subheader("Download Results")
        if st.sidebar.button("Download Model"):
            with open("random_forest_model.pkl", "rb") as file:
                st.sidebar.download_button(label="Download Model", data=file, file_name="random_forest_model.pkl")
        
        st.sidebar.subheader("Download Optimized Data")
        if st.sidebar.button("Download Optimized Data"):
            df.to_csv('optimized_sales_data.csv', index=False)
            st.sidebar.write("Optimized sales data saved as 'optimized_sales_data.csv'")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import re

# --- Data Loading and Preprocessing (copied from your original notebook) ---

# Load the dataset
# Note: In a real-world app, you'd host this file online or package it with the app.
try:
    df = pd.read_csv("laptop_price.csv", encoding='latin-1')
except FileNotFoundError:
    st.error("The 'laptop_price.csv' file was not found. Please ensure it is in the same directory as this script.")
    st.stop()

# Function to remove outliers using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    return df_cleaned

# Function to remove outliers using Z-score
def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    df_cleaned = df[z_scores < threshold].copy()
    return df_cleaned

# Convert 'Weight' to numeric before outlier removal
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

# Apply outlier removal to 'Price_euros', 'Weight' and 'Inches'
df = remove_outliers_iqr(df, 'Price_euros')
df = remove_outliers_iqr(df, 'Weight')
df = remove_outliers_zscore(df, 'Inches')

# Convert 'Ram' from string to integer
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)

# --- Feature Engineering (Copied and adapted from your notebook) ---

# Engineer 'ScreenResolution' and add new 'Touchscreen' and 'PPI' features
df[['Screen_Resolution_Width', 'Screen_Resolution_Height']] = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)').astype(int)
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
df['PPI'] = ((df['Screen_Resolution_Width']**2 + df['Screen_Resolution_Height']**2)**0.5 / df['Inches']).astype(float)
df['Screen_Resolution_Type'] = df['ScreenResolution'].apply(lambda x: ' '.join(x.split()[:-1]))
df = df.drop('ScreenResolution', axis=1)

# Engineer 'Memory'
def parse_storage(storage_string):
    ssd_gb = 0
    hdd_gb = 0
    flash_gb = 0
    storage_types = storage_string.split('+')
    for item in storage_types:
        if 'SSD' in item:
            match = re.search(r'(\d+)\s*(GB|TB)', item)
            if match:
                size = int(match.group(1))
                unit = match.group(2)
                ssd_gb = size * 1024 if unit == 'TB' else size
        if 'HDD' in item:
            match = re.search(r'(\d+)\s*(GB|TB)', item)
            if match:
                size = int(match.group(1))
                unit = match.group(2)
                hdd_gb = size * 1024 if unit == 'TB' else size
        if 'Flash' in item:
            match = re.search(r'(\d+)\s*(GB|TB)', item)
            if match:
                size = int(match.group(1))
                unit = match.group(2)
                flash_gb = size * 1024 if unit == 'TB' else size
    return pd.Series([ssd_gb, hdd_gb, flash_gb])
df[['SSD_GB', 'HDD_GB', 'Flash_Storage_GB']] = df['Memory'].apply(parse_storage)
df = df.drop('Memory', axis=1)

# Engineer 'Cpu'
def extract_cpu_brand(cpu_string):
    if 'Intel' in cpu_string:
        return 'Intel'
    elif 'AMD' in cpu_string:
        return 'AMD'
    elif 'Samsung' in cpu_string:
        return 'Samsung'
    else:
        return 'Other'
df['Cpu_Brand'] = df['Cpu'].apply(extract_cpu_brand)
def extract_cpu_type(cpu_string):
    if 'Core i' in cpu_string:
        return 'Core i'
    elif 'Ryzen' in cpu_string:
        return 'Ryzen'
    elif 'Celeron' in cpu_string:
        return 'Celeron'
    elif 'Pentium' in cpu_string:
        return 'Pentium'
    elif 'Atom' in cpu_string:
        return 'Atom'
    elif 'Xeon' in cpu_string:
        return 'Xeon'
    elif 'FX' in cpu_string:
        return 'FX'
    elif 'E-Series' in cpu_string:
        return 'E-Series'
    elif 'A' in cpu_string:
        return 'A'
    elif 'M' in cpu_string:
        return 'M'
    else:
        return 'Other'
df['Cpu_Type'] = df['Cpu'].apply(extract_cpu_type)
def extract_clock_speed(cpu_string):
    match = re.search(r'(\d+\.?\d*)GHz', cpu_string)
    return float(match.group(1)) if match else np.nan
df['Cpu_Clock_Speed'] = df['Cpu'].apply(extract_clock_speed)
df = df.drop('Cpu', axis=1)

# Engineer 'Gpu'
def get_gpu_brand(gpu_string):
    if 'Nvidia' in gpu_string:
        return 'Nvidia'
    elif 'AMD' in gpu_string:
        return 'AMD'
    elif 'Intel' in gpu_string:
        return 'Intel'
    else:
        return 'Other'
df['Gpu_Brand'] = df['Gpu'].apply(get_gpu_brand)
df = df.drop('Gpu', axis=1)

# Engineer 'OpSys'
def get_os_type(os_string):
    if 'Windows' in os_string:
        return 'Windows'
    elif 'Mac' in os_string:
        return 'Mac'
    elif 'Linux' in os_string:
        return 'Linux'
    else:
        return 'Other'
df['OpSys_Type'] = df['OpSys'].apply(get_os_type)
df = df.drop('OpSys', axis=1)

# One-Hot Encoding of all Categorical Variables
categorical_cols = ['Company', 'TypeName', 'Screen_Resolution_Type', 'Cpu_Brand', 'Cpu_Type', 'Gpu_Brand', 'OpSys_Type']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
df = df.drop(columns=['Product'], errors='ignore')

# Separate features (X) and target variable (y)
X = df.drop('Price_euros', axis=1)
y = df['Price_euros']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# --- Streamlit Application UI and Logic ---

st.title("Laptop Price Predictor")
st.write("Enter the specifications of a laptop to get a price prediction.")

# Create input widgets for all features
with st.sidebar:
    st.header("Laptop Specifications")
    company = st.selectbox('Company', sorted(df['Company'].unique()))
    type_name = st.selectbox('Type', sorted(df['TypeName'].unique()))
    ram = st.slider('RAM (GB)', 2, 64, 8)
    weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=2.0)
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    inches = st.number_input('Screen Size (Inches)', min_value=10.0, max_value=20.0, value=15.6)
    cpu_brand = st.selectbox('CPU Brand', sorted(df['Cpu_Brand'].unique()))
    cpu_type = st.selectbox('CPU Type', sorted(df['Cpu_Type'].unique()))
    cpu_clock = st.number_input('CPU Clock Speed (GHz)', min_value=1.0, max_value=5.0, value=2.5)
    ssd = st.number_input('SSD Storage (GB)', min_value=0, max_value=2048, value=256, step=128)
    hdd = st.number_input('HDD Storage (GB)', min_value=0, max_value=2048, value=0, step=128)
    gpu_brand = st.selectbox('GPU Brand', sorted(df['Gpu_Brand'].unique()))
    os_type = st.selectbox('Operating System', sorted(df['OpSys_Type'].unique()))
    screen_width = st.number_input('Screen Width (px)', min_value=1366, max_value=3840, value=1920)
    screen_height = st.number_input('Screen Height (px)', min_value=768, max_value=2160, value=1080)

# Function to get the correct one-hot encoded dictionary
def get_input_data():
    # Initialize all columns to 0
    input_data = pd.Series(np.zeros(len(X.columns)), index=X.columns)

    # Set user-selected values
    input_data['Inches'] = inches
    input_data['Ram'] = ram
    input_data['Weight'] = weight
    input_data['Screen_Resolution_Width'] = screen_width
    input_data['Screen_Resolution_Height'] = screen_height
    input_data['Cpu_Clock_Speed'] = cpu_clock
    input_data['SSD_GB'] = ssd
    input_data['HDD_GB'] = hdd
    input_data['Flash_Storage_GB'] = 0
    input_data['Touchscreen'] = 1 if touchscreen == 'Yes' else 0
    input_data['PPI'] = np.sqrt(screen_width**2 + screen_height**2) / inches

    # Set one-hot encoded values
    try:
        input_data[f'Company_{company}'] = 1
        input_data[f'TypeName_{type_name}'] = 1
        input_data[f'Cpu_Brand_{cpu_brand}'] = 1
        input_data[f'Cpu_Type_{cpu_type}'] = 1
        input_data[f'Gpu_Brand_{gpu_brand}'] = 1
        input_data[f'OpSys_Type_{os_type}'] = 1
    except KeyError as e:
        st.warning(f"Warning: The selected category '{e.args[0]}' was not found in the training data. The model may not be as accurate for this selection.")
    
    return pd.DataFrame([input_data])

# Prediction button
if st.button('Predict Price'):
    with st.spinner('Predicting...'):
        input_df = get_input_data()
        predicted_price = model.predict(input_df)[0]
        st.success(f"The predicted price is: **€{predicted_price:,.2f}**")
```
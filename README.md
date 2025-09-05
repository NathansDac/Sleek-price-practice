# Importing all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

# Mounting google drive to the notebook
from google.colab import drive
drive.mount('/content/drive')

# Loading the dataset into the notebook and reading it. Displaying the first five rows of the dataset.
df = pd.read_csv("//content/drive/MyDrive/CPE 221 Lab/datasets/laptop_price.csv", encoding='latin-1')

df.head()

DATA CLEANING AND PROCESSING


# Check for missing values in each column
df.isnull().sum()

df.isnull().values.any()

df.info()

# Since there are no missing values we can look for and address outliers

# Visualize outliers with box plots
numerical_cols = ['Inches', 'Weight', 'Price_euros']

plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

 Analyzing Outliers

Based on the box plots:

- **Inches:** There are a few data points below the lower whisker around 10-11 inches. These might represent smaller, more portable laptops or netbooks which are valid entries.
- **Weight:** There are several data points above the upper whisker, with some values exceeding 4 kg. These could represent heavier gaming laptops or workstations, which are plausible. However, extremely high values might indicate data entry errors.
- **Price_euros:** There are many data points above the upper whisker, indicating a wide range of prices for laptops, including very expensive models. While some of these high prices are expected for high-end laptops, extremely high values could be outliers or represent genuinely expensive specialized machines.






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


# Apply outlier removal to 'Price_euros' and 'Weight' using IQR
df_cleaned_iqr = remove_outliers_iqr(df, 'Price_euros')
df_cleaned_iqr_weight = remove_outliers_iqr(df_cleaned_iqr, 'Weight')

# Apply outlier removal to 'Inches' using Z-score
df_cleaned = remove_outliers_zscore(df_cleaned_iqr_weight, 'Inches')


print(f"Original shape: {df.shape}")
print(f"Shape after removing outliers: {df_cleaned.shape}")

display(df_cleaned.head())

### Justification for Using Different Outlier Removal Methods

The choice of outlier removal method for each feature is based on the visual analysis of the box plots and the nature of the potential outliers observed:

- **Price_euros and Weight (IQR Method):** The box plots for 'Price_euros' and 'Weight' show a significant number of data points far above the upper whisker, suggesting a right-skewed distribution and potential extreme values that could be genuine (high-end laptops) or data errors. The IQR method is robust to skewed distributions and focuses on the spread of the central portion of the data, making it suitable for identifying and removing these more distant outliers without being overly affected by the extreme values themselves. This helps in creating a dataset less influenced by these potentially impactful outliers.

- **Inches (Z-score Method):** The box plot for 'Inches' shows fewer potential outliers, located closer to the main distribution, primarily below the lower whisker. These likely represent smaller form-factor laptops which are genuine entries. The Z-score method is effective for distributions that are closer to normal and identifies data points based on their distance from the mean in terms of standard deviations. For 'Inches', this method can help identify values that are statistically unusual relative to the average screen size, which aligns with the observation of a few smaller-than-average laptops. Using the Z-score here helps retain most of the data while flagging values that are significantly different from the typical screen size.

By using a combination of IQR for features with potentially skewed distributions and more extreme outliers ('Price_euros', 'Weight') and Z-score for a feature with a less skewed distribution and less extreme outliers ('Inches'), we tailor the outlier removal approach to the specific characteristics of each numerical column, aiming for a cleaner dataset that is more suitable for subsequent analysis or modeling.

# Getting information on the columns of the current dataset
df.info()

# Convert 'Ram' from string to integer
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)

# Display the updated DataFrame
display(df.head())

# Perform one-hot encoding on the 'Company' column
df = pd.get_dummies(df, columns=['Company'], drop_first=True, dtype=int)

# Display the updated DataFrame with the new one-hot encoded columns
display(df.head())

# Perform one-hot encoding on the 'TypeName' column
df = pd.get_dummies(df, columns=['TypeName'], drop_first=True, dtype=int)

# Display the updated DataFrame with the new one-hot encoded columns
display(df.head())

# Feature engineer 'ScreenResolution'
# Extract the resolution dimensions
df[['Screen_Resolution_Width', 'Screen_Resolution_Height']] = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)').astype(int)

# Create 'Screen_Resolution_Dimension' column
df['Screen_Resolution_Dimension'] = df['Screen_Resolution_Width'].astype(str) + 'x' + df['Screen_Resolution_Height'].astype(str)

# Extract screen type
df['Screen_Resolution_Type'] = df['ScreenResolution'].apply(lambda x: ' '.join(x.split()[:-1]))

# Drop original 'ScreenResolution' column
df = df.drop('ScreenResolution', axis=1)

# Display the updated DataFrame
display(df.head())

# Perform one-hot encoding on the 'Screen_Resolution_Type' column
df = pd.get_dummies(df, columns=['Screen_Resolution_Type'], drop_first=True, dtype=int)

# Display the updated DataFrame with the new one-hot encoded columns
display(df.head())

# Display unique values and their counts in the 'Cpu' column to understand the data
display(df['Cpu'].value_counts())

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

display(df.head())

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

display(df.head())

import re

def extract_clock_speed(cpu_string):
    match = re.search(r'(\d+\.?\d*)GHz', cpu_string)
    if match:
        return float(match.group(1))
    return np.nan

df['Cpu_Clock_Speed'] = df['Cpu'].apply(extract_clock_speed)

display(df.head())

df = pd.get_dummies(df, columns=['Cpu_Brand'], drop_first=True, dtype=int)
df = pd.get_dummies(df, columns=['Cpu_Type'], drop_first=True, dtype=int)

display(df.head())

df = df.drop('Cpu', axis=1)

display(df.head())

import re

def parse_storage(storage_string):
    # Initialize all storage types to 0
    ssd_gb = 0
    hdd_gb = 0
    flash_gb = 0

    # Split the string to handle multiple storage types (e.g., '128GB SSD + 1TB HDD')
    storage_types = storage_string.split('+')

    for item in storage_types:
        # Check for SSD
        if 'SSD' in item:
            size_gb = 0
            # Use regex to find a number followed by 'GB' or 'TB'
            match = re.search(r'(\d+)\s*(GB|TB)', item)
            if match:
                size = int(match.group(1))
                unit = match.group(2)
                if unit == 'TB':
                    size_gb = size * 1024 # Convert TB to GB
                else:
                    size_gb = size
            ssd_gb = size_gb

        # Check for HDD
        if 'HDD' in item:
            size_gb = 0
            match = re.search(r'(\d+)\s*(GB|TB)', item)
            if match:
                size = int(match.group(1))
                unit = match.group(2)
                if unit == 'TB':
                    size_gb = size * 1024
                else:
                    size_gb = size
            hdd_gb = size_gb

        # Check for Flash Storage
        if 'Flash' in item:
            size_gb = 0
            match = re.search(r'(\d+)\s*(GB|TB)', item)
            if match:
                size = int(match.group(1))
                unit = match.group(2)
                if unit == 'TB':
                    size_gb = size * 1024
                else:
                    size_gb = size
            flash_gb = size_gb

    return pd.Series([ssd_gb, hdd_gb, flash_gb])

# Apply the function to the 'Memory' column and create new columns
df[['SSD_GB', 'HDD_GB', 'Flash_Storage_GB']] = df['Memory'].apply(parse_storage)

# You can now drop the original 'Memory' column
df = df.drop('Memory', axis=1)

# Display the updated DataFrame to see the new columns
df.head()

# --- Corrected code for 'Gpu' ---

def get_gpu_brand(gpu_string):
    """
    Extracts the main GPU brand from a string.
    """
    if 'Nvidia' in gpu_string:
        return 'Nvidia'
    elif 'AMD' in gpu_string:
        return 'AMD'
    elif 'Intel' in gpu_string:
        return 'Intel'
    else:
        return 'Other'

# Create the new 'Gpu_Brand' column
df['Gpu_Brand'] = df['Gpu'].apply(get_gpu_brand)

# One-hot encode the new, simplified 'Gpu_Brand' column
df = pd.get_dummies(df, columns=['Gpu_Brand'], drop_first=True, dtype=int)

# Drop the original 'Gpu' column as it's no longer needed
df = df.drop('Gpu', axis=1)

# --- Corrected code for 'OpSys' ---

def get_os_type(os_string):
    """
    Extracts the general operating system type.
    """
    if 'Windows' in os_string:
        return 'Windows'
    elif 'Mac' in os_string:
        return 'Mac'
    elif 'Linux' in os_string:
        return 'Linux'
    else:
        return 'Other'

# Create the new 'OpSys_Type' column
df['OpSys_Type'] = df['OpSys'].apply(get_os_type)

# One-hot encode the new, simplified 'OpSys_Type' column
df = pd.get_dummies(df, columns=['OpSys_Type'], drop_first=True, dtype=int)

# Drop the original 'OpSys' column
df = df.drop('OpSys', axis=1)

# Display the updated DataFrame to see the changes
df.head()



# Drop 'Product' column as it is too granular for a robust model.
df = df.drop('Product', axis=1)

# Display the final DataFrame with all engineered features
df.info()
df.head()


# --- Phase 2: Exploratory Data Analysis (EDA) ---

# Plot a correlation matrix to see how all numerical features relate to each other and to 'Price_euros'
# We will use the cleaned and engineered DataFrame
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()

# Visualize the relationship between key features and 'Price_euros' using scatter plots.
# The 'Ram' feature is a strong indicator of price.
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Ram', y='Price_euros', data=df)
plt.title('Ram (GB) vs. Price (Euros)')
plt.xlabel('Ram (GB)')
plt.ylabel('Price (Euros)')
plt.show()

# Let's see the relationship between weight and price.
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Weight', y='Price_euros', data=df)
plt.title('Weight (kg) vs. Price (Euros)')
plt.xlabel('Weight (kg)')
plt.ylabel('Price (Euros)')
plt.show()

# Visualize the relationship between Screen Dimensions and Price
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Screen_Resolution_Width', y='Price_euros', data=df)
plt.title('Screen Resolution Width vs. Price (Euros)')
plt.xlabel('Screen Resolution Width')
plt.ylabel('Price (Euros)')
plt.show()

### Analysis of Feature Relationships with Price

Based on the correlation matrix and scatter plots:

*   **Correlation Matrix:** The heatmap shows the correlation coefficients between all numerical features. We can observe the following significant correlations with `Price_euros`:
    *   **Strong Positive Correlation:** `Ram` has the highest positive correlation with `Price_euros` (around 0.74), indicating that laptops with more RAM tend to be more expensive. `Cpu_Clock_Speed` also shows a notable positive correlation (around 0.42).
    *   **Moderate Positive Correlation:** `Inches`, `Weight`, `Screen_Resolution_Width`, and `Screen_Resolution_Height` all show moderate positive correlations with `Price_euros`, suggesting that larger, heavier laptops with higher resolution screens are generally more expensive.
    *   **Other Correlations:** The one-hot encoded categorical features (like `Company`, `TypeName`, `Screen_Resolution_Type`, `Cpu_Brand`, `Cpu_Type`, `Gpu_Brand`, and `OpSys_Type`) show varying levels of correlation, indicating that brand, type, and components significantly influence the price. For example, certain brands (like Apple and Razer) and types (like Workstation and Gaming) might be associated with higher prices, while others (like Netbook) might correlate with lower prices.

*   **Ram (GB) vs. Price (Euros):** The scatter plot clearly shows a strong positive relationship between RAM and price. As the amount of RAM increases, the price of the laptop generally increases as well. The points are clustered at common RAM sizes (e.g., 4GB, 8GB, 16GB), with higher RAM capacities corresponding to higher price ranges.

*   **Weight (kg) vs. Price (Euros):** The scatter plot indicates a general positive trend between weight and price, though the relationship is not as strong as with RAM. Heavier laptops tend to be more expensive, likely due to larger screens, more powerful components, or more robust build quality. However, there is a considerable spread in price for any given weight, suggesting other factors are also very influential.

*   **Screen Resolution Width vs. Price (Euros):** The scatter plot shows a positive relationship between screen resolution width and price. Laptops with wider screens (and generally higher resolutions) are associated with higher prices. This is expected as higher resolution displays are often found in more premium and expensive laptops. There are distinct clusters of points corresponding to common screen widths, with higher widths generally reaching higher price points.

In summary, the exploratory data analysis reveals that RAM, CPU clock speed, screen resolution, and weight are important numerical features influencing laptop price. Additionally, the categorical features related to brand, type, and components play a significant role, as indicated by their correlations in the heatmap.

# Engineer 'Cpu'
def extract_cpu_brand(cpu_string):
    # ... code for extracting brand ...
    pass # Added pass to fix indentation

def extract_cpu_type(cpu_string):
    # ... code for extracting type ...
    pass # Added pass to fix indentation

def extract_clock_speed(cpu_string):
    match = re.search(r'(\d+\.?\d*)GHz', cpu_string)
    return float(match.group(1)) if match else np.nan

# df['Cpu_Clock_Speed'] = df['Cpu'].apply(extract_clock_speed)
# df = df.drop('Cpu', axis=1)

# Calculate PPI (Pixels Per Inch)
df['PPI'] = ((df['Screen_Resolution_Width']**2 + df['Screen_Resolution_Height']**2)**0.5 / df['Inches']).astype(float)

# Create 'Touchscreen' column based on Screen_Resolution_Type columns
touchscreen_cols = [col for col in df.columns if 'Touchscreen' in col]
df['Touchscreen'] = df[touchscreen_cols].any(axis=1).astype(int)

display(df.head())

# --- Final Phase: Model Building and Evaluation ---

# Import necessary libraries for model building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Drop any remaining non-numeric columns and the 'laptop_ID' column before modeling
# This is a safeguard in case any non-numeric data was missed
# Make sure to drop the 'Product' and 'Screen_Resolution_Dimension' columns as well, as they are not numeric or too granular
df_final = df.drop(columns=['laptop_ID', 'Product', 'Screen_Resolution_Dimension'], errors='ignore')

# Separate features (X) and target variable (y)
X = df_final.drop('Price_euros', axis=1)
y = df_final['Price_euros']

# Split the data into training and testing sets
# We use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# --- Model Evaluation ---
# Evaluate the model's performance using key metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f} euros")
print(f"Mean Squared Error (MSE): {mse:.2f} euros^2")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} euros")
print(f"R-squared (R2): {r2:.2f}")

# --- Interpretation of R-squared ---
print("\nInterpretation of R-squared (R2):")
print("R-squared represents the proportion of the variance in the target variable (Price_euros)")
print("that is predictable from the features. An R-squared of 0.85, for example, means that")
print("85% of the variability in the price can be explained by our model's features.")
print("A value closer to 1.0 is better.")

# --- Final Phase: Using the Trained Model for Prediction ---

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Recreate the data from your notebook to have the model available
# (This assumes the previous code block was run and 'model' is in memory)

# Let's create a new DataFrame for a single, new laptop
# This must have the same columns as the training data (X_train)

# Get the column names from the training data
train_cols = X_train.columns

# Create a dictionary for the new laptop with all columns initialized to 0
new_laptop_data = dict.fromkeys(train_cols, 0)

# Fill in the specific values for the new laptop
new_laptop_data['Inches'] = 15.6
new_laptop_data['Ram'] = 16
new_laptop_data['Weight'] = 2.2
new_laptop_data['Screen_Resolution_Width'] = 1920
new_laptop_data['Screen_Resolution_Height'] = 1080
new_laptop_data['Cpu_Clock_Speed'] = 2.8
new_laptop_data['SSD_GB'] = 512
new_laptop_data['HDD_GB'] = 0
new_laptop_data['Flash_Storage_GB'] = 0
new_laptop_data['Touchscreen'] = 0
# Calculate PPI for the new laptop
new_laptop_data['PPI'] = ((new_laptop_data['Screen_Resolution_Width']**2 + new_laptop_data['Screen_Resolution_Height']**2)**0.5 / new_laptop_data['Inches'])

# Example: Dell laptop
new_laptop_data['Company_Dell'] = 1
# Example: Notebook
new_laptop_data['TypeName_Notebook'] = 1
# Example: Full HD screen
new_laptop_data['Screen_Resolution_Type_Full HD'] = 1
# Example: Intel Core i CPU
new_laptop_data['Cpu_Brand_Intel'] = 1
new_laptop_data['Cpu_Type_Core i'] = 1
# Example: Nvidia GPU
new_laptop_data['Gpu_Brand_Nvidia'] = 1
# Example: Windows OS
new_laptop_data['OpSys_Type_Windows'] = 1


# Create a DataFrame from the new laptop data, ensuring the column order matches the training data
new_laptop_df = pd.DataFrame([new_laptop_data], columns=train_cols)


# Use the trained model to predict the price
predicted_price = model.predict(new_laptop_df)

# Print the predicted price
print(f"The predicted price of this new laptop is: {predicted_price[0]:.2f} euros")

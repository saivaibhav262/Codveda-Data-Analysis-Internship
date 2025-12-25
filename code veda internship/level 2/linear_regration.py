# ==========================================
# LEVEL 2 - TASK 1: REGRESSION ANALYSIS
# ==========================================

# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load dataset
df = pd.read_csv("iris_cleaned_dataset.csv")

# Step 3: Display dataset overview
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Step 4: Select independent and dependent variables
X = df[['sepal_length']]   # Independent variable
y = df['petal_length']     # Dependent variable

# Step 5: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict values
y_pred = model.predict(X_test)

# Step 8: Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R-squared Score (R²):", r2)

# Step 9: Model parameters
print("\nModel Parameters:")
print("Coefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)

# Step 10: Conclusion
print("\nConclusion:")
print("Linear Regression successfully predicts petal_length from sepal_length.")

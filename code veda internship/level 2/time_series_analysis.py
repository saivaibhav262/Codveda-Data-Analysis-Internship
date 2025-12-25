print("🔥 SCRIPT FILE IS EXECUTING 🔥")

# ==========================================
# LEVEL 2 - TASK 2: TIME SERIES ANALYSIS
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

print("Program started...")

# Step 1: Load dataset
df = pd.read_csv("Stock Prices Data Set.csv")
print("Dataset loaded successfully!")

# Step 2: Show columns (VERY IMPORTANT)
print("\nColumns in dataset:")
print(df.columns)

# Step 3: Convert Date column
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
print("\nDate column converted and set as index")

# Step 4: Select price column safely
if 'close' in df.columns:
    price_series = df['close']
elif 'Adj close' in df.columns:
    price_series = df['Adj close']
elif 'Price' in df.columns:
    price_series = df['Price']
else:
    raise Exception("❌ No price column found (Close / Adj Close / Price)")

print("Price column selected successfully")

# Step 5: Plot stock prices
plt.figure()
plt.plot(price_series)
plt.title("Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.savefig("stock_price_trend.png")
plt.show()

print("✔ Stock price trend graph displayed & saved")

# Step 6: Seasonal decomposition
decomposition = seasonal_decompose(price_series, model='additive', period=30)
decomposition.plot()
plt.savefig("time_series_decomposition.png")
plt.show()

print("✔ Time series decomposition completed")

# Step 7: Moving average
df['Moving_Average'] = price_series.rolling(window=20).mean()

plt.figure()
plt.plot(price_series, label='Original Price')
plt.plot(df['Moving_Average'], label='Moving Average')
plt.legend()
plt.title("Moving Average Smoothing")
plt.savefig("moving_average_plot.png")
plt.show()

print("✔ Moving average smoothing applied")

print("\n🎉 TIME SERIES ANALYSIS COMPLETED SUCCESSFULLY 🎉")

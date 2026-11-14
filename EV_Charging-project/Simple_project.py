import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

print("1. Generating simple EV charging data...")
# Create 100 days of random hourly data
np.random.seed(42)
total_hours = 100 * 24

# Create simple features: hour of the day and temperature
hours = np.tile(np.arange(24), 100)
temperatures = np.random.uniform(15, 30, total_hours)

# Create EV demand: high demand during peak hours (e.g., 8 AM and 6 PM)
base_demand = 15
peak_hours_impact = np.where((hours == 8) | (hours == 18), 25, 0)
random_noise = np.random.normal(0, 3, total_hours)
ev_demand = base_demand + peak_hours_impact + random_noise
ev_demand = np.clip(ev_demand, 0, None)  # Ensure no negative numbers

# Put everything into a clean table (DataFrame)
df = pd.DataFrame({
    'hour': hours,
    'temperature': temperatures,
    'energy_needed_kwh': ev_demand
})

print("2. Splitting data into Training and Testing sets...")
# Use the first 80 days for training, and the last 20 days for testing
train_df = df.iloc[:80*24]
test_df = df.iloc[80*24:]

X_train = train_df[['hour', 'temperature']]
y_train = train_df['energy_needed_kwh']

X_test = test_df[['hour', 'temperature']]
y_test = test_df['energy_needed_kwh']

print("3. Training the AI Model (XGBoost)...")
model = XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
model.fit(X_train, y_train)

print("4. Making Predictions...")
predictions = model.predict(X_test)

print("5. Plotting results...")
# Plot the first 2 days of the test results to see how well it did
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:48], label='Actual Demand', color='blue', marker='o')
plt.plot(predictions[:48], label='AI Predicted Demand', color='orange', linestyle='--', marker='x')
plt.title('EV Charging Demand: Actual vs Predicted (2-Day Snapshot)')
plt.xlabel('Hours')
plt.ylabel('Energy (kWh)')
plt.legend()
plt.grid(True)

# Save the graph as an image
plt.savefig('project_result.png')
print("Success! Your graph has been saved as 'project_result.png'")

# --- Add this at the very bottom to print a text output table ---
print("\n--- SAMPLE TEXT OUTPUT (First 10 Hours of Predictions) ---")
output_table = pd.DataFrame({
    'Hour of Day': X_test['hour'].iloc[:10].values,
    'Actual Demand (kWh)': y_test.iloc[:10].values.round(2),
    'AI Predicted Demand (kWh)': predictions[:10].round(2)
})
print(output_table.to_string(index=False))


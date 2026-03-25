# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ML models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Advanced model
from xgboost import XGBRegressor


# =========================
# LOAD DATASET
# =========================
# Read CSV file into dataframe
df = pd.read_csv("logistics_classification_dataset.csv")


# =========================
# CREATE TARGET VARIABLE (ETA)
# =========================
# Since dataset has bad time values, we generate realistic delivery time
df['delivery_time_hours'] = (
    df['distance_km'] / 50                  # base time (speed assumption)
    + df['package_weight_kg'] * 0.02        # weight impact
    + df['weather_condition'].map({         # weather delay
        'clear': 0.5,
        'cold': 0.8,
        'foggy': 1.0,
        'hot': 0.7,
        'rainy': 1.5,
        'stormy': 2.0
    })
    + np.random.uniform(0.5, 1.5, len(df))  # random delay
)


# =========================
# ENCODE CATEGORICAL DATA
# =========================
# Convert text to numbers for ML models
le = LabelEncoder()

df['weather_condition'] = le.fit_transform(df['weather_condition'])
df['vehicle_type'] = le.fit_transform(df['vehicle_type'])


# =========================
# SELECT FEATURES & TARGET
# =========================
# Input features
X = df[['distance_km', 'package_weight_kg', 'weather_condition', 'vehicle_type']]

# Target variable (what we predict)
y = df['delivery_time_hours']


# =========================
# TRAIN-TEST SPLIT
# =========================
# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# FEATURE SCALING
# =========================
# Normalize data for better model performance
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================
# DEFINE MODELS
# =========================
# Multiple ML models for comparison
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "SVM (SVR)": SVR(),
    "XGBoost": XGBRegressor(n_estimators=100)
}

# Store results
results = {}
trained_models = {}

print("\n📊 Model Comparison:\n")


# =========================
# TRAIN & EVALUATE MODELS
# =========================
# Loop through each model
for name, model in models.items():
    model.fit(X_train, y_train)  # train model
    y_pred = model.predict(X_test)  # predict

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store results
    results[name] = r2
    trained_models[name] = model

    # Print results
    print(f"{name}:")
    print(f"   MAE = {mae:.3f}")
    print(f"   R2  = {r2:.3f}\n")


# =========================
# GRAPH: MODEL COMPARISON
# =========================
# Plot accuracy of all models
model_names = list(results.keys())
r2_scores = list(results.values())

plt.figure()
plt.bar(model_names, r2_scores)

# Show values on top of bars
for i, v in enumerate(r2_scores):
    plt.text(i, v, f"{v:.2f}", ha='center')

plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.title("Model Comparison (Accuracy)")
plt.xticks(rotation=20)
plt.show()


# =========================
# SELECT BEST MODEL
# =========================
# Choose model with highest R2 score
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print("🏆 Best Model:", best_model_name)


# =========================
# USER INPUT FOR PREDICTION
# =========================
print("\n--- Enter Shipment Details ---")

# Take user input
# Take distance input with validation
while True:
    distance = float(input("Distance (km): "))
    
    if distance > 599:
        print("⚠️ Distance should be less than 600 km. Try again.\n")
    else:
        break
weight = float(input("Weight (kg): "))
weather = int(input("Weather (0-clear,1-cold,2-foggy,3-hot,4-rainy,5-stormy): "))
vehicle = int(input("Vehicle (0-bike,1-ev bike,2-ev van,3-scooter,4-truck,5-van): "))

# Create input dataframe
sample = pd.DataFrame([[distance, weight, weather, vehicle]],
                      columns=['distance_km', 'package_weight_kg', 'weather_condition', 'vehicle_type'])

# Apply same scaling
sample = scaler.transform(sample)

# Predict using best model
prediction = best_model.predict(sample)


# =========================
# FINAL OUTPUT
# =========================
# Show result in 2 decimal places
print(f"\n🚚 Predicted ETA using {best_model_name}: {prediction[0]:.2f} hours")
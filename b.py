import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# XGBoost
from xgboost import XGBRegressor

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("logistics_classification_dataset.csv")

# =========================
# CREATE ETA
# =========================
df['delivery_time_hours'] = (
    df['distance_km'] / 50
    + df['package_weight_kg'] * 0.02
    + df['weather_condition'].map({
        'clear': 0.5,
        'cold': 0.8,
        'foggy': 1.0,
        'hot': 0.7,
        'rainy': 1.5,
        'stormy': 2.0
    })
    + np.random.uniform(0.5, 1.5, len(df))
)

# =========================
# ENCODE
# =========================
le = LabelEncoder()
df['weather_condition'] = le.fit_transform(df['weather_condition'])
df['vehicle_type'] = le.fit_transform(df['vehicle_type'])

# =========================
# FEATURES
# =========================
X = df[['distance_km', 'package_weight_kg', 'weather_condition', 'vehicle_type']]
y = df['delivery_time_hours']

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODELS
# =========================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "SVM (SVR)": SVR(),
    "XGBoost": XGBRegressor(n_estimators=100)
}

results = {}
trained_models = {}

print("\n📊 Model Comparison:\n")

# =========================
# TRAIN & EVALUATE
# =========================
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = r2
    trained_models[name] = model

    print(f"{name}:")
    print(f"   MAE = {mae:.3f}")
    print(f"   R2  = {r2:.3f}\n")

# =========================
# GRAPH (MODEL COMPARISON)
# =========================
model_names = list(results.keys())
r2_scores = list(results.values())

plt.figure()
plt.bar(model_names, r2_scores)

# Add values on top
for i, v in enumerate(r2_scores):
    plt.text(i, v, f"{v:.2f}", ha='center')

plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.title("Model Comparison (Accuracy)")
plt.xticks(rotation=20)
plt.show()

# =========================
# BEST MODEL
# =========================
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]

print("🏆 Best Model:", best_model_name)

# =========================
# USER INPUT
# =========================
print("\n--- Enter Shipment Details ---")

distance = float(input("Distance (km): "))
weight = float(input("Weight (kg): "))
weather = int(input("Weather (0-clear,1-cold,2-foggy,3-hot,4-rainy,5-stormy): "))
vehicle = int(input("Vehicle (0-bike,1-ev bike,2-ev van,3-scooter,4-truck,5-van): "))

sample = pd.DataFrame([[distance, weight, weather, vehicle]],
                      columns=['distance_km', 'package_weight_kg', 'weather_condition', 'vehicle_type'])

# scale input
sample = scaler.transform(sample)

prediction = best_model.predict(sample)

# =========================
# FINAL OUTPUT (2 DECIMAL)
# =========================
print(f"\n🚚 Predicted ETA using {best_model_name}: {prediction[0]:.2f} hours")
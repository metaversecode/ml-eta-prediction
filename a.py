import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =========================
# LOAD ORIGINAL DATASET
# =========================
df = pd.read_csv("logistics_classification_dataset.csv")

# =========================
# CREATE CLEAN ETA
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
# ENCODE ONLY REQUIRED
# =========================
le = LabelEncoder()

df['weather_condition'] = le.fit_transform(df['weather_condition'])
df['vehicle_type'] = le.fit_transform(df['vehicle_type'])

# =========================
# FEATURES (IMPORTANT)
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
# TRAIN
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

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

prediction = model.predict(sample)

print("\n🚚 Predicted ETA:", round(prediction[0], 2), "hours")

# =========================
# GRAPH
# =========================
plt.figure()
plt.scatter(y_test, y_pred, s=10)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
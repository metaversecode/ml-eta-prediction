import pandas as pd
import numpy as np

# Load original dataset
df = pd.read_csv("logistics_classification_dataset.csv")

# =========================
# 1. CREATE CLEAN DELIVERY TIME
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
# 2. DROP USELESS COLUMNS
# =========================
df = df.drop(columns=[
    'delivery_id',
    'expected_time_hours',
    'delivery_partner',
    'package_type',
    'delivery_status'
])

# =========================
# 3. SAVE CLEAN DATASET
# =========================
df.to_csv("final_clean_dataset.csv", index=False)

print("✅ Final clean dataset created: final_clean_dataset.csv")
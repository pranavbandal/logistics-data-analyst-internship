"""
generate_raw_data.py
Simulates a raw logistics shipment dataset with realistic data-quality
problems: missing values, outliers, duplicate rows, and inconsistent
text formatting. This mimics data as it might arrive from a warehouse
management system (WMS) before any cleaning.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 500  # number of shipment records

cities = ["Mumbai", "mumbai", "Pune", "PUNE", "Delhi", "Bangalore",
          "bangalore", "Chennai", "Hyderabad", "Kolkata"]
carriers = ["BlueDart", "Delhivery", "DTDC", "Ekart", "XpressBees"]
modes = ["Road", "Rail", "Air", "road", "AIR"]

# --- Core fields ---
shipment_id = [f"SHIP{100000+i}" for i in range(N)]
origin_city = np.random.choice(cities, N)
dest_city = np.random.choice(cities, N)
carrier = np.random.choice(carriers, N)
mode = np.random.choice(modes, N, p=[0.4, 0.15, 0.15, 0.2, 0.1])

# Distance (km) - roughly realistic, with a few extreme outliers injected
distance_km = np.random.normal(450, 180, N).clip(20, None)
outlier_idx = np.random.choice(N, 6, replace=False)
distance_km[outlier_idx] = distance_km[outlier_idx] * np.random.uniform(4, 7, 6)

# Delivery time (days) - correlated loosely with distance + noise
delivery_days = (distance_km / 120) + np.random.normal(1.5, 1.0, N)
delivery_days = delivery_days.clip(0.5, None)
# inject a few unrealistic outliers (data entry errors)
err_idx = np.random.choice(N, 4, replace=False)
delivery_days[err_idx] = np.random.uniform(30, 60, 4)

# Shipment weight (kg)
weight_kg = np.random.gamma(shape=2, scale=45, size=N)

# Shipping cost (INR) - depends on distance & weight + noise
shipping_cost = (distance_km * 8 + weight_kg * 15) * np.random.uniform(0.85, 1.2, N)

# On-time flag derived loosely
on_time = np.where(delivery_days <= (distance_km / 120 + 3), "Yes", "No")

df = pd.DataFrame({
    "Shipment_ID": shipment_id,
    "Origin_City": origin_city,
    "Destination_City": dest_city,
    "Carrier": carrier,
    "Transport_Mode": mode,
    "Distance_KM": np.round(distance_km, 1),
    "Weight_KG": np.round(weight_kg, 2),
    "Delivery_Time_Days": np.round(delivery_days, 2),
    "Shipping_Cost_INR": np.round(shipping_cost, 2),
    "On_Time": on_time,
})

# --- Inject missing values (MCAR-style, ~ a few % per column) ---
for col, frac in [("Weight_KG", 0.04), ("Shipping_Cost_INR", 0.03),
                   ("Delivery_Time_Days", 0.02), ("Carrier", 0.015),
                   ("Destination_City", 0.01)]:
    miss_idx = np.random.choice(N, int(N * frac), replace=False)
    df.loc[miss_idx, col] = np.nan

# --- Inject duplicate rows ---
dup_rows = df.sample(8, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

# --- Shuffle ---
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("raw_logistics_data.csv", index=False)
print(f"Generated raw_logistics_data.csv with {len(df)} rows, {df.shape[1]} columns")
print("\nMissing values per column:\n", df.isna().sum())
print("\nDuplicate rows (excluding Shipment_ID):", df.drop(columns=["Shipment_ID"]).duplicated().sum())

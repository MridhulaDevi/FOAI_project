import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from clean_data import load_and_clean

# ---- 1. Load cleaned data ----
df = load_and_clean()

# ---- 2. Filter to a few common instance types (keeps things fast) ----
TARGET_INSTANCES = ["m4.xlarge", "c3.4xlarge", "r3.large"]
df = df[df["InstanceType"].isin(TARGET_INSTANCES)]

print("Filtered shape:", df.shape)
print(df["InstanceType"].value_counts())
# ---- 3. Encode categorical columns ----
df = pd.get_dummies(df, columns=["InstanceType", "AvailabilityZone", "OperatingSystem"], drop_first=True)

# ---- 4. Sort by time, then split chronologically (80% train, 20% test) ----
df = df.sort_values("Timestamp").reset_index(drop=True)
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print("\nTrain size:", len(train_df))
print("Test size:", len(test_df))
# ---- 5. Define features and target ----
feature_cols = [c for c in df.columns if c not in ["Timestamp", "SpotPrice"]]

X_train = train_df[feature_cols]
y_train = train_df["SpotPrice"]

X_test = test_df[feature_cols]
y_test = test_df["SpotPrice"]

# ---- 6. Train regression model ----
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained.")
# ---- 7. Evaluate ----
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nMAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# ---- 8. Residual std (this is your uncertainty band for chance nodes) ----
residuals = y_test - y_pred
residual_std = residuals.std()
print(f"Residual Std Dev: {residual_std:.4f}")

# ---- 9. Save model + feature columns + residual std for later use ----
joblib.dump({
    "model": model,
    "feature_cols": feature_cols,
    "residual_std": residual_std
}, "src/utils/price_model.pkl")

print("\nModel saved to src/utils/price_model.pkl")
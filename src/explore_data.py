import pandas as pd

column_names = ["Timestamp", "InstanceType", "OperatingSystem", "AvailabilityZone", "SpotPrice"]
df = pd.read_csv("data/us-east-1.csv", header=None, names=column_names)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMemory usage (MB):", df.memory_usage(deep=True).sum() / 1024**2)
print("Row count:", len(df))
import pandas as pd
import numpy as np


def load_and_clean(filepath="data/us-east-1.csv"):
    column_names = ["Timestamp", "InstanceType", "OperatingSystem",
                    "AvailabilityZone", "SpotPrice"]

    df = pd.read_csv(filepath, header=None, names=column_names)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["SpotPrice"] = pd.to_numeric(df["SpotPrice"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "SpotPrice"])

    df["InstanceType"] = df["InstanceType"].str.strip()
    df["OperatingSystem"] = df["OperatingSystem"].str.strip()
    df["AvailabilityZone"] = df["AvailabilityZone"].str.strip()

    df = df.drop_duplicates()
    df = df.sort_values("Timestamp").reset_index(drop=True)

    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

    return df


# ---------------- cached data access ----------------
_CACHE = {}


def get_data(filepath="data/us-east-1.csv"):
    """Loads and cleans the CSV once, then reuses it from memory."""
    if filepath not in _CACHE:
        _CACHE[filepath] = load_and_clean(filepath)
    return _CACHE[filepath]


def get_price_stats(instance_type, availability_zone, operating_system,
                    hour=None, min_rows=50):
    """
    Returns a SORTED numpy array of historical spot prices for the chosen
    instance / zone / OS. If enough rows exist for the given hour of day,
    only that hour is used. No model, no training - just filtering data.
    """
    df = get_data()
    seg = df[
        (df["InstanceType"] == instance_type) &
        (df["AvailabilityZone"] == availability_zone) &
        (df["OperatingSystem"] == operating_system)
    ]
    if hour is not None:
        seg_hour = seg[seg["Hour"] == hour]
        if len(seg_hour) >= min_rows:
            seg = seg_hour
    return np.sort(seg["SpotPrice"].to_numpy())


def get_price_history(instance_type, availability_zone, filepath="data/us-east-1.csv"):
    """Historical SpotPrice + Timestamp for the dashboard chart."""
    df = get_data(filepath)
    filtered = df[
        (df["InstanceType"] == instance_type) &
        (df["AvailabilityZone"] == availability_zone)
    ][["Timestamp", "SpotPrice"]].sort_values("Timestamp")
    return filtered


if __name__ == "__main__":
    df = load_and_clean()
    print("Cleaned shape:", df.shape)
    print(df.dtypes)
    print("Date range:", df["Timestamp"].min(), "to", df["Timestamp"].max())
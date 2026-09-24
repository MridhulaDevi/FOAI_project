import pandas as pd

def load_and_clean(filepath="data/us-east-1.csv"):
    column_names = ["Timestamp", "InstanceType", "OperatingSystem", "AvailabilityZone", "SpotPrice"]
    
    df = pd.read_csv(filepath, header=None, names=column_names)

    # Convert Timestamp to proper datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Ensure SpotPrice is numeric
    df["SpotPrice"] = pd.to_numeric(df["SpotPrice"], errors="coerce")

    # Drop rows where Timestamp or SpotPrice failed to parse
    df = df.dropna(subset=["Timestamp", "SpotPrice"])

    # Clean string columns (strip whitespace, standardize case if needed)
    df["InstanceType"] = df["InstanceType"].str.strip()
    df["OperatingSystem"] = df["OperatingSystem"].str.strip()
    df["AvailabilityZone"] = df["AvailabilityZone"].str.strip()

    # Remove duplicate rows if any
    df = df.drop_duplicates()

    # Sort chronologically (important for time-series work later)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Feature engineering for ML step (Day 3)
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

    return df

if __name__ == "__main__":
    df = load_and_clean()
    
    print("Cleaned shape:", df.shape)
    print("\nDtypes:")
    print(df.dtypes)
    print("\nDate range:", df["Timestamp"].min(), "to", df["Timestamp"].max())
    print("\nUnique InstanceTypes:", df["InstanceType"].nunique())
    print("Sample InstanceTypes:", df["InstanceType"].unique()[:10])
    print("\nUnique AvailabilityZones:", df["AvailabilityZone"].unique())
    print("\nUnique OperatingSystems:", df["OperatingSystem"].unique())
    print("\nSpotPrice range:", df["SpotPrice"].min(), "to", df["SpotPrice"].max())
    print("\nMissing values check:")
    print(df.isnull().sum())
def get_price_history(instance_type, availability_zone, filepath="data/us-east-1.csv"):
    """
    Returns historical SpotPrice + Timestamp for a given instance/zone,
    used for the dashboard's price trend chart.
    """
    df = load_and_clean(filepath)
    filtered = df[
        (df["InstanceType"] == instance_type) &
        (df["AvailabilityZone"] == availability_zone)
    ][["Timestamp", "SpotPrice"]].sort_values("Timestamp")

    return filtered
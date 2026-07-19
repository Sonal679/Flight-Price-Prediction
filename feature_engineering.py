import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

# Load cleaned data
df = pd.read_csv("Data/cleaned_flight_data.csv")

print("Columns in cleaned data:")
print(df.columns.tolist())

# -----------------------------
# Date of Journey
# -----------------------------
df["Journey_Day"] = pd.to_datetime(
    df["Date_of_Journey"],
    dayfirst=True
).dt.day

df["Journey_Month"] = pd.to_datetime(
    df["Date_of_Journey"],
    dayfirst=True
).dt.month

df.drop("Date_of_Journey", axis=1, inplace=True)

# -----------------------------
# Departure Time
# -----------------------------
df["Dep_Hour"] = pd.to_datetime(
    df["Dep_Time"],
    format="%H:%M"
).dt.hour

df["Dep_Min"] = pd.to_datetime(
    df["Dep_Time"],
    format="%H:%M"
).dt.minute

df.drop("Dep_Time", axis=1, inplace=True)

# -----------------------------
# Arrival Time
# -----------------------------
df["Arrival_Hour"] = pd.to_datetime(
    df["Arrival_Time"],
    format="%H:%M",
    errors="coerce"
).dt.hour

df["Arrival_Min"] = pd.to_datetime(
    df["Arrival_Time"],
    format="%H:%M",
    errors="coerce"
).dt.minute

df.drop("Arrival_Time", axis=1, inplace=True)

# -----------------------------
# Duration
# -----------------------------
df["Duration_Hours"] = (
    df["Duration"]
    .str.extract(r"(\d+)h")
    .fillna(0)
    .astype(int)
)

df["Duration_Mins"] = (
    df["Duration"]
    .str.extract(r"(\d+)m")
    .fillna(0)
    .astype(int)
)

df.drop("Duration", axis=1, inplace=True)

# -----------------------------
# Total Stops
# -----------------------------
df["Total_Stops"] = (
    df["Total_Stops"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({
        "non-stop": 0,
        "1 stop": 1,
        "2 stops": 2,
        "3 stops": 3,
        "4 stops": 4
    })
    .astype(int)
)

# -----------------------------
# Drop unwanted columns
# -----------------------------
df.drop(["Route", "Additional_Info"], axis=1, inplace=True)

# -----------------------------
# Final Check
# -----------------------------
print("\nFinal Columns:")
print(df.columns.tolist())
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# -----------------------------
# Save Final Dataset
# -----------------------------
df.to_csv("Data/final_flight_data.csv", index=False)

print(" Feature Engineering completed successfully!")
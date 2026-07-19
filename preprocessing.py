import pandas as pd

df = pd.read_excel("Data/Data_Train.xlsx")   

print(df.columns.tolist())
print(df.head())

print("Shape before cleaning:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print("Columns after cleaning:", df.columns.tolist())

print("\nShape after cleaning:", df.shape)

df.to_csv("Data/cleaned_flight_data.csv", index=False)

print("Data cleaning completed successfully!")
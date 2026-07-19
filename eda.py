import pandas as pd
# Load Dataset
df=pd.read_excel("data/Data_Train.xlsx")
print("First 5 rows:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nColumns:")
print(df.columns)
print("\nMissing values:")
print(df.isnull().sum())
print("\n Dataset Shape:")
print(df.shape)
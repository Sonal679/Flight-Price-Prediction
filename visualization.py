import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_excel("data/Data_Train.xlsx")
plt.figure(figsize=(8,5))
plt.hist(df["Price"],bins=30)
plt.title("Flight Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Flights")
plt.show()
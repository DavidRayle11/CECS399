import numpy as np
import pandas as pd
from scipy.io import loadmat

# nirs_path = r"E:\Projects\CECS399\data\raw_nirs\S1\S171_run01.nirs"

# data = loadmat(nirs_path)

# t = data["t"].flatten()
# aux = data["aux"]

# print("Loaded:", nirs_path)
# print("Time points:", t.shape)
# print("AUX shape:", aux.shape)

# for i in range(aux.shape[1]):
#     print(f"Channel {i}: min={aux[:,i].min():.4f}, max={aux[:,i].max():.4f}")


df = pd.read_excel("data/TC_BehavioralData.xlsx")

print("SubNum dtype:", df["SubNum"].dtype)
print("Total unique SubNums:", df["SubNum"].nunique())
print("Min SubNum:", df["SubNum"].min())
print("Max SubNum:", df["SubNum"].max())
print("First 20 unique SubNums:")
print(sorted(df["SubNum"].unique())[:20])

print()

print(1174 in df["SubNum"].values)
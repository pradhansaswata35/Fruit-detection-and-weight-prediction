import numpy as np
import pandas as pd

df = pd.read_csv("../datasets/apple_weight_dataset.csv")
df.drop(["red_and_green"], axis="columns", inplace=True)

X = df.drop(["weight"], axis="columns")
y = df["weight"]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X[["height", "width"]] = scaler.fit_transform(df[["height", "width"]])

min_vals = scaler.data_min_     # array: [min_height, min_width]
print(f"apple [min_height, min_width] : {min_vals}")

scale_vals = scaler.data_max_ - scaler.data_min_  # range for scaling
print(f"apple scale vals : {scale_vals}")

scaler_obj = scaler # Keep scaler accessible too
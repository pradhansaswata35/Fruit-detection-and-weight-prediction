import numpy as np
import pandas as pd

df = pd.read_csv("datasets/banana_weight_dataset.csv")
df.drop(["yellow_and_green"], axis="columns", inplace=True)

X = df.drop(["weight"], axis="columns")
y = df["weight"]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X[["height", "width"]] = scaler.fit_transform(df[["height", "width"]])

min_vals = scaler.data_min_     # array: [min_height, min_width]
# print(f"banana [min_height, min_width] : {min_vals}")

scale_vals = scaler.data_max_ - scaler.data_min_  # range for scaling
# print(f"banana scale vals : {scale_vals}")

scaler_obj = scaler # Keep scaler accessible too
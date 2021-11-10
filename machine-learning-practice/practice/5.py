
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

print(os.getcwd())

train_data = pd.read_csv("data/train.csv")
print(train_data)
print(train_data.shape)
print(train_data.index.values)
print(train_data.columns)
print(train_data.head())
print(train_data.info())
print(train_data.describe())
print(train_data.corr())
plt.matshow(train_data.corr())

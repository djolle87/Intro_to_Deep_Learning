import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Step 1 load data

df = pd.read_csv('data/data.csv')
df = df.drop(['index', 'price', 'sq_price'], axis=1)
df = df[0:10]

# Step 2 - add labels

# 1 is good buy and 0 is bad buy
df.loc[:, ('y1')] = [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
# y2 is a negation of y1, opposite
df.loc[:, ('y2')] = df['y1'] == 0
# turn True/False values to 1s and 0s
df.loc[:, ('y2')] = df['y2'].astype(int)

# Step 3 - prepare data for tensorflow

# tensors are a generic version of vectors and matrices
# vector is a list of numbers (1D Tensor)
# matrix is a list of list of numbers (2D Tensor)
# list of list of list of numbers (3D Tensor)
# .....
# Convert features to input tensor
inputX = df.loc[:, ['area', 'bathrooms']].as_matrix()
# Convert labels to input tensors
inputY = df.loc[:, ['y1', 'y2']].as_matrix()

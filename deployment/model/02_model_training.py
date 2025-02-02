import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

'''
super simple model to use with
- flask
- docker
- kubernetes
'''

# read the data 
df = pd.read_csv('Real estate.csv', header = None)
print(df.head())

# segreate features and target
# ft_drop = ["No", "Y house price of unit area", "X1 transaction date"]

X = df.iloc[1:, 2:6]
# X = df.drop(ft_drop, axis=1)

y = df.iloc[1:, 7]
# y = df['Y house price of unit area']

print (X.head())

# train the model
lr = LinearRegression().fit(X, y)

# output the score
print(f"algorithm score is: {lr.score(X, y)}")

# create pickle file
joblib.dump(lr, 'model.pkl')
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib


# read in the model
model_path = 'model.pkl'
model = open(model_path, "rb")
lr_model = joblib.load(model)
model.close()

print(lr_model.predict([[1, 2, 3, 4]]))
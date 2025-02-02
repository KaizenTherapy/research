import pandas as pd 
import numpy as np
import sklearn 
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        print(request.form.get('var 1'))
        print(request.form.get('var 2'))
        print(request.form.get('var 3'))
        print(request.form.get('var 4'))
        print(request.form.get('var 5'))
    
        try:
            var_1 = float(request.form.get('var 1'))
            var_2 = float(request.form.get('var 2'))
            var_3 = float(request.form.get('var 3'))
            var_4 = float(request.form.get('var 4'))
            var_5 = float(request.form.get('var 5'))
            
            pred_args = [var_1, var_2, var_3, var_4, var_5]
            preds = pred_args.reshape(1, -1)

            model = open("../model/model.pkl", "rb")
            lr_model = joblib.load(model)
            model.close()

            model_prediction = lr_model.predict(preds)
            model_prediction = round(float(model_prediction), 2)

        except ValueError:
            return "Please enter valid values"
    
    return render_template('predict.html', prediction = model_prediction)

if __name__ == '__main__':
        app.run(host = "0.0.0.0")
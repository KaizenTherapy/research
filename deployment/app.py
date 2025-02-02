import pandas as pd 
import numpy as np
import sklearn 
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        print(request.form.get('var_1'))
        print(request.form.get('var_2'))
        print(request.form.get('var_3'))
        print(request.form.get('var_4'))
        print(request.form.get('var_5'))
    
        try:
            var_1 = float(request.form.get('var_1'))
            var_2 = float(request.form.get('var_2'))
            var_3 = float(request.form.get('var_3'))
            var_4 = float(request.form.get('var_4'))
            # var_5 = float(request.form.get('var_5'))

            # print(var_1, var_2, var_3, var_4, var_5)
            
            # pred_args = [var_1, var_2, var_3, var_4, var_5]


            print(var_1, var_2, var_3, var_4)
            
            pred_args = [[var_1, var_2, var_3, var_4]]
            # preds = pred_args.reshape(1, -1)
            preds = pred_args

            model = open("model/model.pkl", "rb")
            lr_model = joblib.load(model)
            model.close()

            # pred_args = pd.DataFrame([pred_args])
            print(lr_model.predict([[1, 2, 3, 4]]))

            model_prediction = lr_model.predict(preds)
            model_prediction = round(float(model_prediction), 2)
            
            print(f"model_prediction: {model_prediction}")

        except ValueError:
            return "Please enter valid values"
    
    return render_template('predict.html', prediction = model_prediction)

if __name__ == '__main__':
        app.run(host = "0.0.0.0")
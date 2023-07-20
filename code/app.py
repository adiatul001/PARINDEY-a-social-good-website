from flask import Flask,request, url_for, redirect, render_template
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
scaler=pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]

    cols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']


    df = pd.DataFrame([int_features], columns=cols)
    numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
    df[numerical_features] = scaler.transform(df[numerical_features])


    final=[np.array(int_features)] 
    print(int_features)
    print(final)
    prediction=model.predict_proba(df)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('index.html',pred='Your heart is at risk.\nProbability of cardiovascular diseases is {}'.format(output),msg="you need to do something")
    else:
        return render_template('index.html',pred='Your heart is safe.\n Probability of cardiovascular diseases is {}'.format(output),msg="your heart is Safe for now")
    

if __name__ == '__main__':
    app.run()
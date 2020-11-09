#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 08:53:49 2020

@author: hungerbox
"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import os
import librosa
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("model.pkl","rb")
model=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All- New"

@app.route('/predict_audio',methods=["GET"])
def predict_audio():
    """Let's Authenticate the High Frequency of Gun's Sound 
    This is using docstrings for specifications.
    We have Model Trained on 9 different sounds and their frequency.
    The Model is Capable of Identifying 9 different types of Sounds 
    ---
    parameters:  
      - name: X_test
        in: qyery
        type: file
        required: true
    responses:
        200:
            description: The Result  should be - 0 (Air Conditioner),1(Car Honking),2(Children Playing),3(Dog Barking),4(Drilling),5(Engine Idling),6(Gun Shot),7(JackHammer),8(Siren),9(Street Music)
        
    """
    df_test=librosa.load(request.files.get('Audio_Frequency_Indicator'))
    prediction=model.predict(df_test)
    print(prediction)
    return "Hello The answer is"+str(prediction)

    
if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)
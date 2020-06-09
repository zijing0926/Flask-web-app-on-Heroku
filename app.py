#import pickle
#import os
#import numpy as np
import numpy as np
from flask import Flask, request, jsonify, render_template
import dill
import spacy
import re


app = Flask(__name__)
model = dill.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/stats')
def stats():
    return render_template('stats.html')

@app.route('/comov')
def plot():
    return render_template('comov.html')

@app.route('/word')
def word():
    return render_template('word.html')




def ValuePredictor(to_predict_list):

    features = [to_predict_list]
    prediction = model.predict_proba(features)

    return int(prediction[0][1]*100)


@app.route('/result')
def result():

    to_predict_list = request.args.get('News',0)

    result = ValuePredictor(to_predict_list)


    return render_template("result.html", prediction='The Probability of Price Increase is {}%'.format(result))


if __name__ == '__main__':
  app.run(port=33508)

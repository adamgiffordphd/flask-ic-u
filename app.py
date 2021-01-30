from flask import Flask, render_template, request, redirect, url_for
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import CDN
import numpy as np
import pandas as pd

app = Flask(__name__)
app.vars = {}

# load model

def make_prediction(x_test):
  # get prediction from model
  y_pred = []
  return make_plots(y_pred)

def make_plots(y_pred):
  # make plots

@app.route('/')
def intro():
  return render_template('intro.html')

@app.route('/howto')
def howto():
  return render_template('howto.html')

@app.route('/about')
def about():
  return render_template('about.html')

@app.route('/forms')
def forms():
  return render_template('forms.html')

@app.route('/data', methods=['POST'])
def predict():
  app.vars['SUBJECT_ID'] = request.form.get('patientid')
  app.vars['ADMIT_AGE'] = request.form.get('age')
  app.vars['GENDER'] = request.form.get('gender')
  app.vars['RELIGION'] = request.form['religion']
  app.vars['LANGUAGE'] = request.form.get('language')
  app.vars['MARITAL_STATUS'] = request.form['maritalstatus']
  app.vars['ETHNICITY'] = request.form.get('ethnicity')

  app.vars['ADMISSION_TYPE'] = request.form.get('admissiontype')
  app.vars['ADMISSION_LOCATION'] = request.form.get('admissionlocation')
  app.vars['INSURANCE'] = request.form['insurance']

  app.vars['DIAGNOSIS'] = request.form.get('diagnosis')
  app.vars['NOTES'] = request.form.get('clinicalnotes')

  return redirect(url_for('display')) 

@app.route('/display')
def display():
  # x_test = pd.from_dict(app.vars)

  # y_pred = make_prediction(x_test)
  
  # plots = make_plots(y_pred)

  # n = 100

  x = np.random.random(n) * 10
  y = np.random.random(n) * 10
  s = np.random.random(n)

  p = figure(width=300, height=300)
  p.circle(x, y, radius=s, fill_alpha=0.6)

  script, div = components(p)

  return render_template('display.html',
                          script=script, div=div)

if __name__ == '__main__':
  app.run(port=33507, debug=True)

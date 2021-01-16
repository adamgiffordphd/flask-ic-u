from flask import Flask, render_template, request, redirect
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import CDN
import numpy as np

app = Flask(__name__)

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

@app.route('/display')
def display():
  n = 100

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

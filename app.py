from flask import Flask, render_template, request, redirect

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

if __name__ == '__main__':
  app.run(port=33507, debug=True)

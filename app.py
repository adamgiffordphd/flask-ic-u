from flask import Flask, render_template, request, current_app, url_for
from bokeh.plotting import figure
from bokeh.palettes import Spectral4
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.embed import components
# from bokeh.resources import CDN
from bokeh.layouts import Column
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
app.vars = {}

slidedir = '.static/'

def generate_page_list():
    pages = [
        # {"name": "Intro", "url": url_for("")},
        {"name": "About", "url": url_for("about")},
        {"name": "Slide Deck", "url": url_for("iframe")},
        {"name": "How to", "url": url_for("howto")},
        {"name": "Get Started", "url": url_for("forms")},
        # {"name": "Performance", "url": url_for("performance")}
    ]
    return pages

# load urgency model data
urg_model = './models/log__URGENCY__20210130.pkl'
los_model = './models/log__URGENCY__20210130.pkl'
with open(urg_model, 'rb') as f:
  urg_dict = pickle.load(f)
  num_cols = urg_dict['numeric_cols']
  cat_cols = urg_dict['categorical_cols']
  diagn_col = urg_dict['diagnosis_col']
  urg_ohe = urg_dict['ohe_categoricals']
  urg_diagn_vect = urg_dict['diagn_vect']
  urg_feats_un = urg_dict['feature_union']
  urg_est = urg_dict['estimator']
  del urg_dict

# with open(los_model, 'rb') as f:
#   los_dict = pickle.load(f)
#   los_ohe = los_dict['ohe_categoricals']
#   los_ohe = los_dict['ohe_categoricals']
#   los_feat_un = los_dict['feature_union']
#   los_count_vect = los_dict['count_vectorizor']
#   los_est = los_dict['estimator']
#   del los_dict

def getFeatureNames(categorical,bow,numerical=['ADMIT_AGE']):
    feature_names = numerical
    feature_names.extend([c.split('_')[-1] for c in categorical.get_feature_names()])
    feature_names.extend(bow.get_feature_names())
    
    return feature_names

def createFeatureCoeffDict(coefs,features):
  feats_coeffs = [(f, c) for c,f in zip(coefs,features)]
  return feats_coeffs

def getMostImportantFeaturesUrg(feature_imp_urg, x_test, n_most=4):
  x_test = urg_feats_un.transform(x_test).toarray()[0]
  total = sum([abs(fs[1]) for fs in feature_imp_urg])
  features_scores = [(f[0], x * f[1] / total) for x,f in zip(x_test,feature_imp_urg)]
  most_important = sorted(features_scores, key=lambda x: abs(x[1]))
  return most_important[-n_most:]

def make_predictions(x_test, urgency=True, los=False):
  # get prediction from model
  y_pred_urg = int(urg_est.predict(x_test)[0])
  y_pred_proba_urg = urg_est.predict_proba(x_test)[0]
  # y_pred_los = los_est.predict(x_test)

  return y_pred_urg, y_pred_proba_urg #, y_pred_los

def make_urgency_plot(y_pred_proba):
  factors = ['stable','questionable','urgent','immediate']
  source = ColumnDataSource(data=dict(factors=factors, probs=y_pred_proba, color=Spectral4))
  
  hover = HoverTool(
      tooltips=[
          ("Level", "@factors"),
          ("Probability", "@probs"),
      ]
  )
  
  p = figure(x_range=factors, plot_height=250,
          plot_width=450, tools=[hover])

  p.vbar(x='factors', top='probs', color='color', width=0.4, source=source)

  p.title.text_font_size = '16pt'
  p.yaxis.axis_label_text_font_size = '14pt'
  p.yaxis.major_label_text_font_size = '11pt'
  p.xaxis.major_label_text_font_size = '11pt'
  p.xgrid.grid_line_color = None
  p.y_range.start = 0
  p.yaxis.axis_label = 'P(category)'

  return p, factors[y_pred_proba.argmax()].upper()

def make_urgency_factors_plot(most_imp_feats_urg):
  y = [m[0][:15] for m in most_imp_feats_urg]
  x = [abs(m[1]) for m in most_imp_feats_urg]
  x = [val/max(x) for val in x]

  source = ColumnDataSource(dict(factors=y, scores=x))

  hover = HoverTool(
      tooltips=[
          ("Factor", "@factors"),
          ("Scaled Score", "@scores"),
      ]
  )
  
  p = figure(y_range=y, plot_height=250, title="Main Factors",
          plot_width=450, tools=[hover])

  p.hbar(y='factors', right='scores', height=0.4, source=source)

  p.title.text_font_size = '16pt'
  p.yaxis.axis_label_text_font_size = '14pt'
  p.yaxis.major_label_text_font_size = '11pt'
  p.xaxis.major_label_text_font_size = '11pt'
  p.ygrid.grid_line_color = None
  p.x_range.start = 0
  p.xaxis.visible = False
  return p

@app.route('/')
def intro():
  return render_template('intro.html', pages=generate_page_list())

@app.route('/howto')
def howto():
  return render_template('howto.html')

@app.route('/examples')
def examples():
  ex = request.args.get('ex')
  update = request.args.get('update')
  if ex is None:
    page = 'examples.html'
  elif update is None:
    page = 'example-' + ex + '.html'
  else:
    page = 'example-' + ex + '-update.html'
  return render_template(page)

@app.route('/example-urg')
def exampleurg():
  return render_template('example-urg.html')

@app.route('/about')
def about():
  return render_template('about.html')

@app.route('/forms')
def forms():
  return render_template('forms.html')

@app.route('/iframe')
def iframe():
  return render_template('iframe.html')

@app.route('/display', methods=['POST'])
def display():
  # need all 26 columns to use estimator data transformations (even though only 
  # 10 data columns are actually used in the model)
  app.vars['SUBJECT_ID'] = [int(request.form.get('patientid'))]
  app.vars['GENDER'] = [request.form.get('gender')]
  app.vars['HADM_ID'] = [999999]
  app.vars['ADMITTIME'] = [4]
  app.vars['DISCHTIME'] = [4]
  app.vars['ADMISSION_TYPE'] = [request.form.get('admissiontype')]
  app.vars['ADMISSION_LOCATION'] = [request.form.get('admissionlocation')]
  app.vars['INSURANCE'] = [request.form['insurance']]
  app.vars['LANGUAGE'] = [request.form.get('language')]
  app.vars['RELIGION'] = [request.form['religion']]
  app.vars['MARITAL_STATUS'] = [request.form['maritalstatus']]
  app.vars['ETHNICITY'] = [request.form.get('ethnicity')]
  app.vars['DIAGNOSIS'] = [request.form.get('diagnosis')]
  app.vars['HOSPITAL_EXPIRE_FLAG'] = [0]
  app.vars['HAS_CHARTEVENTS_DATA'] = [0]
  app.vars['HOSPITAL_DAYS'] = [0]
  app.vars['ADMIT_AGE'] = [request.form.get('age')]
  app.vars['ICUSTAY_ID'] = [888888]
  app.vars['DBSOURCE'] = ['']
  app.vars['INTIME'] = [4]
  app.vars['LOS'] = [0]
  app.vars['DAYS_ADM_TO_ICU'] = [0]
  app.vars['SAMEDAY_ADM_TO_ICU'] = [0]
  app.vars['ADM_TO_ICU_100p'] = [0]
  app.vars['ADM_TO_ICU_90m'] = [0]
  app.vars['ICU_URGENCY'] = [0]
  
  # app.vars['TEXT'] = request.form.get('clinicalnotes') # not implemented yet, so do nothing 
                                                         # with clinical notes at this time
  x_test = pd.DataFrame.from_dict(app.vars, orient = 'columns')
  print(x_test.columns)
  # y_pred_urg, y_pred_proba_urg, y_pred_los = make_predictions(x_test)
  y_pred_urg, y_pred_proba_urg = make_predictions(x_test)

  feature_names_urg = getFeatureNames(urg_ohe, urg_diagn_vect.named_steps.count,
                                    num_cols)
  # feature_names_los = getFeatureNames(los_ohe, los_diagn_vect.named_steps.count,
  #                                   num_cols)                 
  feature_imp_urg = createFeatureCoeffDict(urg_est.best_estimator_.named_steps.reg.coef_[y_pred_urg],
                                        feature_names_urg)
  most_imp_feats_urg = getMostImportantFeaturesUrg(feature_imp_urg, x_test)

  p_urg = make_urgency_plot(y_pred_proba_urg)
  p_urg_factors = make_urgency_factors_plot(most_imp_feats_urg)
  layout = Column(p_urg, p_urg_factors)
  script, div = components(layout)

  return render_template('display.html',
                          script=script, div=div)

@app.route('/performance')
def performance():
  return render_template('performance.html')

@app.route('/display-example', methods=['POST'])
def displayexample():
  exs_dict = {}
  exs_dict['immediate'] = 'inactive'
  exs_dict['urgent'] = 'inactive'
  exs_dict['questionable'] = 'inactive'
  exs_dict['stable'] = 'inactive'

  if int(request.form.get('patientid'))==43320:
    example_file = './examples/stable_examp_loc68757_20210216.pkl'
    exs_dict['stable'] = 'active'
    color = 'color:rgb(43, 131, 186);'
    text = '''
    Patient 43320 has been identified as 'stable'. Stable status indicates that
    the need for ICU admission is NOT likely within 5 days. The main factors 
    contributing to this estimate are: gender ("M"), admission type 
    ("ELECTIVE"), clinical notes ("cystectomy"), and language ("ENGL").
    '''
    los = 4.88
  elif request.form.get('patientid')=='5285':
    example_file = './examples/questionable_pre-urgent_examp_loc10545_20210216.pkl'
    exs_dict['questionable'] = 'active'
    color = 'color:rgb(171, 221, 164);'
    text = '''
    Patient 5285 has been identified as having 'questionable' need of intensive care. 
    Questionable status indicates a likely need for ICU admission within 5 days. 
    The main factors contributing to this estimate are: gender ("F"), 
    diagnosis information ("aortic"), and clinical notes ("avr", "cabg").
    '''
    los = 4.02
  elif request.form.get('patientid')=='5285 ':
    example_file = './examples/urgent_examp_loc10547_20210216.pkl'
    exs_dict['urgent'] = 'active'
    color = 'color:rgb(253, 174, 97);'
    text = '''
    Patient 5285 has been identified as having 'urgent' need of intensive care. 
    Urgent status indicates a likely need for ICU admission within 24 hours. 
    The main factors contributing to this estimate are: gender ("F"), 
    diagnosis information ("aortic", "aorta", "asymmetric").
    '''
    los = 4.04
  elif int(request.form.get('patientid'))==3986:
    example_file = './examples/immediate_examp_loc8065_20210216.pkl'
    exs_dict['immediate'] = 'active'
    color = 'color:rgb(215, 25, 28);'
    text = '''
    Patient 3986 has been identified as having 'immediate' need of intensive care. 
    Immediate status indicates a likely need for ICU admission within 1 hour. 
    The main factors contributing to this estimate are: diagnoisis information
    ("aneurysm"), admission type ("EMERGENCY"), admission location ("CLINICAL 
    REFERRAL/PREMATURE"), and gender ("M").
    '''
    los = 5.19
  else:
    return render_template('example-error.html',id=request.form.get('patientid'))


  y_pred_proba_urg, most_imp_feats_urg = pickle.load(open(example_file,'rb'))

  p_urg, risk = make_urgency_plot(y_pred_proba_urg)
  p_urg_factors = make_urgency_factors_plot(most_imp_feats_urg)
  layout = Column(p_urg, p_urg_factors)
  script, div = components(layout)

  return render_template('display.html',
                          script=script, div=div, text=text, los=los,
                          risk=risk,exs=exs_dict,color=color)

if __name__ == '__main__':
  app.run(port=33507, debug=True)

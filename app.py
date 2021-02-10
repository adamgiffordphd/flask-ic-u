from flask import Flask, render_template, request, current_app #,url_for
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
  
  ttl = "IC-U Risk Factor: {}".format(factors[y_pred_proba.argmax()].upper())
  p = figure(x_range=factors, plot_height=250, title=ttl,
          plot_width=450, tools=[hover])

  p.vbar(x='factors', top='probs', color='color', width=0.4, source=source)

  p.title.text_font_size = '16pt'
  p.yaxis.axis_label_text_font_size = '14pt'
  p.yaxis.major_label_text_font_size = '11pt'
  p.xaxis.major_label_text_font_size = '11pt'
  p.xgrid.grid_line_color = None
  p.y_range.start = 0
  p.yaxis.axis_label = 'P(category)'

  return p

def make_urgency_factors_plot(most_imp_feats_urg):
  y = [m[0] for m in most_imp_feats_urg]
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
  return render_template('intro.html')

@app.route('/howto')
def howto():
  return render_template('howto.html')

@app.route('/example-urg')
def exampleurg():
  return render_template('example-urg.html')

@app.route('/about')
def about():
  return render_template('about.html')

@app.route('/forms')
def forms():
  return render_template('forms.html')

@app.route('/slides')
def slides():
    slide = request.args.get('slide')
    html = 'slide' + slide + '.html'
    return current_app.send_static_file(html)
    # return url_for('static', filename=slide)

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
  # return redirect(url_for('display')) 

@app.route('/display-example', methods=['POST'])
def displayexample():
  if int(request.form.get('patientid'))==88409:
    example_file = './examples/urg_examp_ix28_20210204.pkl'
    
    text = '''
    Patient 88409 has been identified as having 'urgent' need of intensive care.<br />
    Urgent status indicates a likely need for ICU admission within 24 hours.<br />
    The main factors contributing to this estimate are: gender ("male"), <br />
    diagnosis information ("arm infection"), and insurance type ("Government").
    '''
    los = 3.49

  y_pred_proba_urg, most_imp_feats_urg = pickle.load(open(example_file,'rb'))

  p_urg = make_urgency_plot(y_pred_proba_urg)
  p_urg_factors = make_urgency_factors_plot(most_imp_feats_urg)
  layout = Column(p_urg, p_urg_factors)
  script, div = components(layout)

  return render_template('display.html',
                          script=script, div=div, text=text, los=los)

if __name__ == '__main__':
  app.run(port=33507, debug=True)

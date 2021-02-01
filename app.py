from flask import Flask, render_template, request, redirect, url_for
from bokeh.plotting import figure
from bokeh.palettes import Spectral4
from bokeh.models import ColumnDataSource
from bokeh.embed import components
from bokeh.resources import CDN
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
app.vars = {}

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
  features_scores = [(f[0], x * f[1]) for x,f in zip(x_test,feature_imp_urg)]
  most_important = sorted(features_scores, key=lambda x: -abs(x[1]))
  return most_important[:n_most]

def make_predictions(x_test, urgency=True, los=False):
  # get prediction from model
  y_pred_urg = int(urg_est.predict(x_test)[0])
  y_pred_proba_urg = urg_est.predict_proba(x_test)[0]
  # y_pred_los = los_est.predict(x_test)

  return y_pred_urg, y_pred_proba_urg #, y_pred_los

def make_urgency_plot(y_pred_proba):
  factors = ['stable','questionable','urgent','immediate']
  source = ColumnDataSource(data=dict(factors=factors, probs=y_pred_proba, color=Spectral4))

  p = figure(x_range=factors, plot_height=250, title="IC-U Risk Factor",
            plot_width=300, toolbar_location=None, tools="")

  p.vbar(x='factors', top='probs', color='color', width=0.4, source=source)

  p.xgrid.grid_line_color = None
  p.y_range.start = 0
  p.yaxis.axis_label = 'P(category)'

  return p

def make_urgency_factors_plot(most_imp_feats_urg):
  y = [m[0] for m in most_imp_feats_urg]
  x = [abs(m[1]) for m in most_imp_feats_urg]
  x = [val/np.round(max(x)) for val in x]

  source = ColumnDataSource(dict(factors=y, scores=x,))

  p = figure(y_range=y, plot_height=250, title="Main Factors",
            plot_width=400, toolbar_location=None, tools="")

  p.hbar(y='factors', right='scores', height=0.4, source=source)

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

@app.route('/about')
def about():
  return render_template('about.html')

@app.route('/forms')
def forms():
  return render_template('forms.html')

@app.route('/data', methods=['POST'])
def display():
  # need all 26 columns to use estimator data transformations (even though only 
  # 10 data columns are actually used in the model)
  app.vars['SUBJECT_ID'] = request.form.get('patientid')
  app.vars['GENDER'] = request.form.get('gender')
  app.vars['HADM_ID'] = 999999
  app.vars['ADMITTIME'] = np.nan
  app.vars['DISCHTIME'] = np.nan
  app.vars['ADMISSION_TYPE'] = request.form.get('admissiontype')
  app.vars['ADMISSION_LOCATION'] = request.form.get('admissionlocation')
  app.vars['INSURANCE'] = request.form['insurance']
  app.vars['LANGUAGE'] = request.form.get('language')
  app.vars['RELIGION'] = request.form['religion']
  app.vars['MARITAL_STATUS'] = request.form['maritalstatus']
  app.vars['ETHNICITY'] = request.form.get('ethnicity')
  app.vars['DIAGNOSIS'] = request.form.get('diagnosis')
  app.vars['HOSPITAL_EXPIRE_FLAG'] = np.nan
  app.vars['HAS_CHARTEVENTS_DATA'] = np.nan
  app.vars['HOSPITAL_DAYS'] = np.nan
  app.vars['ADMIT_AGE'] = request.form.get('age')
  app.vars['ICUSTAY_ID'] = np.nan
  app.vars['DBSOURCE'] = np.nan
  app.vars['INTIME'] = np.nan
  app.vars['LOS'] = np.nan
  app.vars['DAYS_ADM_TO_ICU'] = np.nan
  app.vars['SAMEDAY_ADM_TO_ICU'] = np.nan
  app.vars['ADM_TO_ICU_100p'] = np.nan
  app.vars['ADM_TO_ICU_90m'] = np.nan
  app.vars['ICU_URGENCY'] = np.nan
  
  # app.vars['TEXT'] = request.form.get('clinicalnotes') # not implemented yet, so do nothing 
                                                         # with clinical notes at this time
  x_test = pd.DataFrame(app.vars, index = [0])
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

  return redirect(url_for('display')) 

# @app.route('/display')
# def display():
#   # x_test = pd.from_dict(app.vars)

#   # y_pred = make_prediction(x_test)
  
#   # plots = make_plots(y_pred)

#   n = 100

#   x = np.random.random(n) * 10
#   y = np.random.random(n) * 10
#   s = np.random.random(n)

#   p = figure(width=300, height=300)
#   p.circle(x, y, radius=s, fill_alpha=0.6)

#   script, div = components(p)

#   return render_template('display.html',
#                           script=script, div=div)

if __name__ == '__main__':
  app.run(port=33507, debug=True)

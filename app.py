from flask import Flask, render_template, request, url_for
from bokeh.embed import components
from bokeh.layouts import column
import pandas as pd
import pickle
import visualizations as viz

app = Flask(__name__)

# load urgency model data
urg_model = './models/log__URGENCY__20210304_forFlask_from_20210216.pkl'
with open(urg_model, 'rb') as f:
  text_dft, urg_est = pickle.load(f)
  # the following are neeeded for visualizations
  urg_feature_union = urg_est.named_steps.features
  urg_onehot = urg_est.named_steps.features.transformer_list[0][1].transformer_list[1][1]
  urg_text_cvt = urg_est.named_steps.features.transformer_list[1][1].named_steps.count

los_model = './models/gradientboost__LOS__20210305_forFlask_from_20210215.pkl'
with open(los_model, 'rb') as f:
  los_dft, los_est = pickle.load(f)

def make_predictions(x_test, urgency=True, los=True):
  # get prediction from model
  y_pred_urg = int(urg_est.predict(x_test)[0])
  y_pred_proba_urg = urg_est.predict_proba(x_test)[0]
  y_pred_los = (100*los_est.predict(x_test)[0]//1)/100

  return y_pred_urg, y_pred_proba_urg, y_pred_los

@app.route('/')
def intro():
  return render_template('intro.html')

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
  # need all 22 columns to use estimator data transformations (even though only 
  # 11 data columns are actually used in the model)
  app.vars = {
    'SUBJECT_ID': [int(request.form.get('patientid'))],
    'HADM_ID': [999999],
    'ICUSTAY_ID': [888888],
    'ADMITTIME': [4],
    'CHARTTIME': [4],
    'INTIME': [4],
    #  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # columns used in model:
    'GENDER': [request.form.get('gender')],
    'ADMIT_AGE': [int(request.form.get('age'))],
    'ADMISSION_TYPE': [request.form.get('admissiontype')],
    'ADMISSION_LOCATION': [request.form.get('admissionlocation')],
    'INSURANCE': [request.form.get('insurance')],
    'LANGUAGE': [request.form.get('language')],
    'RELIGION': [request.form.get('religion')],
    'MARITAL_STATUS': [request.form.get('maritalstatus')],
    'ETHNICITY': [request.form.get('ethnicity')],
    'DIAGNOSIS': [request.form.get('diagnosis')],
    'TEXT': request.form.get('clinicalnotes'),
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    'DBSOURCE': [''],
    'LOS': [-9999],
    'DAYS_EVENT_TO_ICU': [-9999],
    'ICU_URGENCY': [0],
    'SAMEDAY_EVENT_TO_ICU': [0]
  }
  facts_dict = {
        'immediate': 'inactive',
        'urgent': 'inactive',
        'questionable': 'inactive',
        'stable': 'inactive'
    }

  x_test = pd.DataFrame.from_dict(app.vars, orient = 'columns')

  y_pred_urg, y_pred_proba_urg, y_pred_los = make_predictions(x_test)
  # y_pred_urg, y_pred_proba_urg = make_predictions(x_test)

  feature_names_urg = viz.getFeatureNames(urg_onehot, urg_text_cvt, ['ADMIT_AGE'])
  # feature_names_los = viz.getFeatureNames(urg_onehot, los_text_cvt, ['ADMIT_AGE'])

  feature_imp_urg = viz.createFeatureCoeffDict(urg_est.named_steps.reg.coef_[y_pred_urg],
                                        feature_names_urg)
  most_imp_feats_urg = viz.getMostImportantFeaturesUrg(urg_feature_union, feature_imp_urg, x_test)

  p_urg, risk = viz.make_urgency_plot(y_pred_proba_urg)
  p_urg_factors = viz.make_urgency_factors_plot(most_imp_feats_urg)
  layout = column(p_urg, p_urg_factors)
  script, div = components(layout)
  color, text = viz.get_viz_metadata(app.vars['SUBJECT_ID'][0],
                                y_pred_urg, [feat[0] for feat in most_imp_feats_urg])

  return render_template('display.html',
                          script=script, div=div, text=text, los=y_pred_los,
                          risk=risk,exs=facts_dict,color=color)

@app.route('/performance')
def performance():
  urg_cf_file = './performance/cf__URGENCY__for_model_20210216.pkl'
  urg_data = pickle.load(open(urg_cf_file,'rb'))

  fig, select = viz.make_urg_performance_plot(urg_data)

  layout = column(select, fig)
  script, div = components(layout)

  return render_template('performance.html',
                          script=script, div=div)

@app.route('/display-example', methods=['POST'])
def displayexample():
  exs_dict = {
    'immediate': 'inactive',
    'urgent': 'inactive',
    'questionable': 'inactive',
    'stable': 'inactive'
  }

  if request.form.get('patientid')=='43320':
    example_file = './examples/stable_examp_loc68757_20210216.pkl'
    exs_dict['stable'] = 'active'
    color, text, los = viz.get_example('stable')

  elif request.form.get('patientid')=='5285':
    example_file = './examples/questionable_pre-urgent_examp_loc10545_20210216.pkl'
    exs_dict['questionable'] = 'active'
    color, text, los = viz.get_example('questionable')

  elif request.form.get('patientid')=='5285 ':
    example_file = './examples/urgent_examp_loc10547_20210216.pkl'
    exs_dict['urgent'] = 'active'
    color, text, los = viz.get_example('urgent')

  elif request.form.get('patientid')=='3986':
    example_file = './examples/immediate_examp_loc8065_20210216.pkl'
    exs_dict['immediate'] = 'active'
    color, text, los = viz.get_example('immediate')

  else:
    return render_template('example-error.html',id=request.form.get('patientid'))

  y_pred_proba_urg, most_imp_feats_urg = pickle.load(open(example_file,'rb'))

  p_urg, risk = viz.make_urgency_plot(y_pred_proba_urg)
  p_urg_factors = viz.make_urgency_factors_plot(most_imp_feats_urg)
  layout = column(p_urg, p_urg_factors)
  script, div = components(layout)

  return render_template('display.html',
                          script=script, div=div, text=text, los=los,
                          risk=risk,exs=exs_dict,color=color)

if __name__ == '__main__':
  app.run(port=33507, debug=True)

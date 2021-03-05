from bokeh.plotting import figure
from bokeh.palettes import Spectral4, Viridis256
from bokeh.models import (ColumnDataSource, HoverTool, 
      CustomJS, Select, LinearColorMapper, ColorBar)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # helper functions for plotting # # # # # # # # # # # # # # # #
def getFeatureNames(categorical,bow,numerical=['ADMIT_AGE']):
    '''gets the feature names from the output of the data 
    transformers for use in the visualizations'''

    feature_names = numerical
    feature_names.extend([c.split('_')[-1] for c in categorical.get_feature_names()])
    feature_names.extend(bow.get_feature_names())
    
    return feature_names

def createFeatureCoeffDict(coefs,features):
    '''gets the feature names from the output of the data 
    transformers for use in the visualizations'''
    feats_coeffs = [(f, c) for c,f in zip(coefs,features)]
    return feats_coeffs

def getMostImportantFeaturesUrg(urg_feature_union, feature_imp_urg, x_test, n_most=4):
    '''transforms inputs to feature space and then finds 
    which of those features have largest coefficients for 
    urgency factors plot'''

    x_test = urg_feature_union.transform(x_test).toarray()[0]
    total = sum([abs(fs[1]) for fs in feature_imp_urg])
    features_scores = [(f[0], x * f[1] / total) for x,f in zip(x_test,feature_imp_urg)]
    most_important = sorted(features_scores, key=lambda x: abs(x[1]))
    return most_important[-n_most:]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # helper functions for examples # # # # # # # # # # # # # # # #
def get_stable_ex():
    color = 'color:rgb(43, 131, 186);'
    text = '''
    Patient 43320 has been identified as 'stable'. Stable status indicates that
    the need for ICU admission is NOT likely within 5 days. The main factors 
    contributing to this estimate are: gender ("M"), admission type 
    ("ELECTIVE"), clinical notes ("cystectomy"), and language ("ENGL").
    '''
    los = 4.88
    return color, text, los

def get_questionable_ex():
    color = 'color:rgb(171, 221, 164);'
    text = '''
    Patient 5285 has been identified as having 'questionable' need of intensive care. 
    Questionable status indicates a likely need for ICU admission within 5 days. 
    The main factors contributing to this estimate are: gender ("F"), 
    diagnosis information ("aortic"), and clinical notes ("avr", "cabg").
    '''
    los = 4.02
    return color, text, los

def get_urgent_ex():
    color = 'color:rgb(253, 174, 97);'
    text = '''
    Patient 5285 has been identified as having 'urgent' need of intensive care. 
    Urgent status indicates a likely need for ICU admission within 24 hours. 
    The main factors contributing to this estimate are: gender ("F"), 
    diagnosis information ("aortic"), and clinical notes ("aorta", "asymmetric").
    '''
    los = 4.04
    return color, text, los

def get_immediate_ex():
    color = 'color:rgb(215, 25, 28);'
    text = '''
    Patient 3986 has been identified as having 'immediate' need of intensive care. 
    Immediate status indicates a likely need for ICU admission within 1 hour. 
    The main factors contributing to this estimate are: diagnoisis information
    ("aneurysm"), admission type ("EMERGENCY"), admission location ("CLINICAL 
    REFERRAL/PREMATURE"), and gender ("M").
    '''
    los = 5.19
    return color, text, los

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # plotting functions for individual predictions # # # # # # # # # # # # #
def make_urgency_plot(y_pred_proba):
    '''make the urgency plot showing urgency score'''

    factors = ['stable','questionable','urgent','immediate']
    source = ColumnDataSource(data=dict(factors=factors, probs=y_pred_proba, color=Spectral4))

    hover = HoverTool(
        tooltips=[
            ("Level", "@factors"),
            ("Probability", "@probs"),
        ]
    )

    p = figure(x_range=factors, plot_height=250,
            plot_width=450, tools=[hover], title="Urgency Score Probabilities")

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
    '''make the urgency factors plot showing largest 
    contributing factors to result'''

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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # plotting functions for model evaluations  # # # # # # # # # # # # #
def make_urg_performance_plot(urg_data):
    source = ColumnDataSource(data=urg_data)

    mapper = LinearColorMapper(
    palette=Viridis256,
    low=0,
    high=urg_data['true'].max()
    )
    color_bar = ColorBar( color_mapper=mapper, location=( 0, 0))

    TOOLTIPS = [
    ("(Pred., True)", "(@x, @y)"),
    ("Raw Count", "@raw"),
    ("Norm. by True", "@true{1.111}"),
    ("Norm. by Pred.", "@pred{1.111}"),
    ("Norm. by Total", "@all{1.111}")
    ]

    labels = ['stable','questionable','urgent','immediate']
    fig = figure(title="Confusion Matrix", tooltips=TOOLTIPS, toolbar_location=None,
            x_range=labels, y_range=labels[-1::-1], plot_height=500, plot_width=600)
    rect = fig.rect('y', 'x', source=source, fill_color={'field': 'color', 'transform': mapper}, line_color='black', width=1, height=1)
    fig.add_layout(color_bar, 'right')


    fig.xaxis.axis_label = 'Predicted'
    fig.yaxis.axis_label = 'True'
    fig.title.text_font_size = '16pt'
    fig.yaxis.axis_label_text_font_size = '14pt'
    fig.xaxis.axis_label_text_font_size = '14pt'
    fig.yaxis.major_label_text_font_size = '11pt'
    fig.xaxis.major_label_text_font_size = '11pt'

    opts = ['Raw Counts', 'Normalize by True Labels', 'Normalize by Predicted Labels', 'Normalize by Total']
    select = Select(title="Plot Options:", value=opts[1], options=opts)

    codec = """
        var data = source.data;
        var f = select.value;
        const {transform} = rect.glyph.fill_color;
        switch(f) {
        case "Raw Counts":
            data['color'] = data['raw'];
            break;
        case "Normalize by True Labels":
            data['color'] = data['true'];
            break;
        case "Normalize by Predicted Labels":
            data['color'] = data['pred'];
            break;
        case "Normalize by Total":
            data['color'] = data['all'];
            break;
        default:
            data['color'] = data['true'];
        }
        transform.low = 0;
        transform.high = Math.max.apply(Math,data['color']);
        rect.glyph.fill_color = {field: 'color', transform: transform};
        // necessary becasue we mutated source.data in-place
        source.change.emit();
        p.reset.emit()
    """
    update_cm = CustomJS(args=dict(source=source, select=select, rect=rect, 
                                fig=fig), code=codec)
    select.js_on_change('value', update_cm)

    return fig, select
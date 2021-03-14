# IC-U with Flask

This app is the web implementation for my capstone project , IC-U, for The Data Incubator. IC-U is an app that uses patient demographic and clinical information recorded during his/her stay at a hospital to predict an urgency score for potential intensive care needs and an etimated length of stay in the ICU. To see the GitHub page for project's data analysis and model implementation, go to [https://github.com/adamgiffordphd/imminent_icu_stays](https://github.com/adamgiffordphd/imminent_icu_stays). 

The repository contains a basic template for a Flask configuration that will
work on Heroku.

The app is hosted on Heroku here: [https://ic-u.herokuapp.com](https://ic-u.herokuapp.com).

## Introduction
ICU care costs and short-term capacity are major pain points for hospitals across the country. Using the MIMIC-III database and a combination of random- forest, ridge, and logistic regression, IC-U culminates in an web app that uses patient demographic and clinical information to produce an urgency score (aptly called "Intensive Care - Urgency", or IC-U) for needed intensive care and an estimated length-of-stay in the ICU. 

By predicting intensive care urgency and length of stay, hospitals can better optimize patient care, flow, and logistics, and better prepare for anticipated spikes in ICU capacity needs. In turn, this information has the potential to improve patient outcomes and decrease operating costs. 

Visit the MIMIC-III database here: [https://mimic.physionet.org](https://mimic.physionet.org).

## Motivation
### The problem
Over 5 million patients annually require some form of intensive care, which requires continual and often invasive monitoring, as well as near 1-to-1 nurse-to-patient staffing in order to treat patients effectively. As a result of the complicated nature of ICU care, ICUs are one of the leading drivers of hospital costs in the US today. Hospitals are currently spending $80B a year in ICU care, and these costs are expected to double by 2030. 

Additionally, median ICU occupancy rates in the US already sit around 75%, and can be as high as 86%. Given that the number of ICU beds in a hospital can range from as low as 6 to 67 beds, many hospitals face extreme challenges managing spikes in ICU needs. 

It is no surprise then that both ICU operating costs and occupancy rates are major pain points for hospitals across the country and that innovations are necessary to help drive costs down.

### My solution
The basis of IC-U is to quantify a risk factor (called the "IC-U" factor) reflecting intensive care urgency for patients that present to a hospital and predict an estimated length of stay in the ICU. My solution would take in a patient’s health and demographic information and estimate (1) the likelihood that s/he may need immediate intensive care and (2) the likely length of stay (LOS) ultimately required in the ICU.

### Value proposition
Intensive Care Urgency and LOS estimates can help hospital staff to more quickly identify patients most in need of intensive care and predict ICU occupancy rates in the near term. This will allow hospitals to optimize patient triaging and monitoring and better manage staffing and capacity concerns. Combined, these benefits could both improve patient care and reduce operating costs.

## Model Design
IC-U uses the MIMIC-III data base, a freely accessible dataset representing data from ~60,000 ICU stays across 40,000 patients. To quantify intensive care urgency, I calculate the time between hospital admission and ultimate ICU admission and bin these times into 4 distinct categories:

- Immediate: ICU admission <1 hour from hospital admission
- Urgent: ICU admission <24 hours from hospital admission
- Questionable: ICU admission <5 days from hospital admission
- Stable: ICU admission >5 days from hospital admission

Two separate models are then fit to the data:
- A multi-class "one-versus-all" logistic regression classifies patients into one of the four urgency categories.
- A gradient-boosting regression estimates the anticipated LOS in the ICU.

## [How it Works](https://ic-u.herokuapp.com/howto)
IC-U works as follows:

- Hospital staff inputs patient information into a web form
- Upon submission, the predictive models analyze the patient data and compute both the IC-U factor and an estimated length of stay.
- Finally, the app displays the results with a description of the main contributors to the IC-U score.

With more information about the size and occupancy of the hospital, these results can be utilized to optimize patient care and staffing and serve as an early warning system for ICU capacity issues.
And as more information for a patient is gathered, the inputs can be updated and a new IC-U factor and LOS can be calculated and displayed as needed.

## Getting Started
Visit the [Examples](https://ic-u.herokuapp.com/examples) page to see how data is inputted into the forms and results are visualized. Then visit the [Get Started](https://ic-u.herokuapp.com/forms) page to input custom data.


## Model Performance
View the [Model Performance](https://ic-u.herokuapp.com/performance) tab to see current information on the models' performance.

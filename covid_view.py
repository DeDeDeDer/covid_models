# from Central_Package.all_dc_package import *
import pandas as pd
import numpy as np
import plotly.express as px  # (version 4.7.0)

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import train_test_split


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

# DATA REFERENCES
# maindata_dir = '/Users/derrick/Desktop/All_Country_Covid19.csv'
maindata_dir = 'data/All_Country_Covid19.csv'
country_list = list(pd.read_csv(maindata_dir)['Country_Province'].unique())
country_list = [x for x in country_list if x]
country_list = [x for x in country_list if str(x) != 'nan']
country_dict = []
for aa in country_list:

    country_dict.append({"label": aa, "value": aa})
# country_dict = [{'label': 'Dominica', 'value': 'Dominica'}, {'label': 'Italy', 'value': 'Italy'}, {'label': 'US', 'value': 'US'}, {'label': 'United Kingdom', 'value': 'United Kingdom'}, {'label': 'Kenya', 'value': 'Kenya'}, {'label': 'Spain', 'value': 'Spain'}, {'label': 'China', 'value': 'China'}, {'label': 'Eswatini', 'value': 'Eswatini'}, {'label': 'Germany', 'value': 'Germany'}, {'label': 'Morocco', 'value': 'Morocco'}, {'label': 'Canada', 'value': 'Canada'}, {'label': 'Laos', 'value': 'Laos'}, {'label': 'Australia', 'value': 'Australia'}, {'label': 'Vietnam', 'value': 'Vietnam'}, {'label': 'Angola', 'value': 'Angola'}, {'label': 'Mauritius', 'value': 'Mauritius'}, {'label': 'Congo (Kinshasa)', 'value': 'Congo (Kinshasa)'}, {'label': 'Ghana', 'value': 'Ghana'}, {'label': 'Cyprus', 'value': 'Cyprus'}, {'label': 'France', 'value': 'France'}, {'label': 'Barbados', 'value': 'Barbados'}, {'label': 'Montenegro', 'value': 'Montenegro'}, {'label': 'Bolivia', 'value': 'Bolivia'}, {'label': 'Israel', 'value': 'Israel'}, {'label': 'MS Zaandam', 'value': 'MS Zaandam'}, {'label': 'Netherlands', 'value': 'Netherlands'}, {'label': 'Georgia', 'value': 'Georgia'}, {'label': 'El Salvador', 'value': 'El Salvador'}, {'label': 'Grenada', 'value': 'Grenada'}, {'label': 'Bangladesh', 'value': 'Bangladesh'}, {'label': 'Monaco', 'value': 'Monaco'}, {'label': 'Tajikistan', 'value': 'Tajikistan'}, {'label': 'Guatemala', 'value': 'Guatemala'}, {'label': 'Trinidad and Tobago', 'value': 'Trinidad and Tobago'}, {'label': 'Andorra', 'value': 'Andorra'}, {'label': 'Equatorial Guinea', 'value': 'Equatorial Guinea'}, {'label': 'Saint Vincent and the Grenadines', 'value': 'Saint Vincent and the Grenadines'}, {'label': 'Saint Kitts and Nevis', 'value': 'Saint Kitts and Nevis'}, {'label': 'Slovakia', 'value': 'Slovakia'}, {'label': 'Ukraine', 'value': 'Ukraine'}, {'label': 'Armenia', 'value': 'Armenia'}, {'label': 'Panama', 'value': 'Panama'}, {'label': 'Jamaica', 'value': 'Jamaica'}, {'label': 'Colombia', 'value': 'Colombia'}, {'label': 'Paraguay', 'value': 'Paraguay'}, {'label': 'Mongolia', 'value': 'Mongolia'}, {'label': 'Denmark', 'value': 'Denmark'}, {'label': 'New Zealand', 'value': 'New Zealand'}, {'label': 'Argentina', 'value': 'Argentina'}, {'label': 'Mozambique', 'value': 'Mozambique'}, {'label': "Cote d'Ivoire", 'value': "Cote d'Ivoire"}, {'label': 'Madagascar', 'value': 'Madagascar'}, {'label': 'Iceland', 'value': 'Iceland'}, {'label': 'Uganda', 'value': 'Uganda'}, {'label': 'Fiji', 'value': 'Fiji'}, {'label': 'Iraq', 'value': 'Iraq'}, {'label': 'Ireland', 'value': 'Ireland'}, {'label': 'South Sudan', 'value': 'South Sudan'}, {'label': 'Malaysia', 'value': 'Malaysia'}, {'label': 'Sierra Leone', 'value': 'Sierra Leone'}, {'label': 'South Africa', 'value': 'South Africa'}, {'label': 'Eritrea', 'value': 'Eritrea'}, {'label': 'Norway', 'value': 'Norway'}, {'label': 'Uzbekistan', 'value': 'Uzbekistan'}, {'label': 'Poland', 'value': 'Poland'}, {'label': 'Suriname', 'value': 'Suriname'}, {'label': 'Mexico', 'value': 'Mexico'}, {'label': 'Thailand', 'value': 'Thailand'}, {'label': 'Congo (Brazzaville)', 'value': 'Congo (Brazzaville)'}, {'label': 'Lebanon', 'value': 'Lebanon'}, {'label': 'Mauritania', 'value': 'Mauritania'}, {'label': 'Korea, South', 'value': 'Korea, South'}, {'label': 'Egypt', 'value': 'Egypt'}, {'label': 'Kyrgyzstan', 'value': 'Kyrgyzstan'}, {'label': 'Belize', 'value': 'Belize'}, {'label': 'Gabon', 'value': 'Gabon'}, {'label': 'Botswana', 'value': 'Botswana'}, {'label': 'Bahrain', 'value': 'Bahrain'}, {'label': 'Western Sahara', 'value': 'Western Sahara'}, {'label': 'Cuba', 'value': 'Cuba'}, {'label': 'Sweden', 'value': 'Sweden'}, {'label': 'Togo', 'value': 'Togo'}, {'label': 'Bhutan', 'value': 'Bhutan'}, {'label': 'Taiwan', 'value': 'Taiwan'}, {'label': 'Namibia', 'value': 'Namibia'}, {'label': 'Iran', 'value': 'Iran'}, {'label': 'Cameroon', 'value': 'Cameroon'}, {'label': 'Finland', 'value': 'Finland'}, {'label': 'Japan', 'value': 'Japan'}, {'label': 'Latvia', 'value': 'Latvia'}, {'label': 'Peru', 'value': 'Peru'}, {'label': 'Russia', 'value': 'Russia'}, {'label': 'Zambia', 'value': 'Zambia'}, {'label': 'Nigeria', 'value': 'Nigeria'}, {'label': 'Portugal', 'value': 'Portugal'}, {'label': 'Malawi', 'value': 'Malawi'}, {'label': 'Algeria', 'value': 'Algeria'}, {'label': 'Saint Lucia', 'value': 'Saint Lucia'}, {'label': 'Nepal', 'value': 'Nepal'}, {'label': 'Honduras', 'value': 'Honduras'}, {'label': 'Uruguay', 'value': 'Uruguay'}, {'label': 'Philippines', 'value': 'Philippines'}, {'label': 'Sao Tome and Principe', 'value': 'Sao Tome and Principe'}, {'label': 'Haiti', 'value': 'Haiti'}, {'label': 'Dominican Republic', 'value': 'Dominican Republic'}, {'label': 'Bahamas', 'value': 'Bahamas'}, {'label': 'Mali', 'value': 'Mali'}, {'label': 'Chad', 'value': 'Chad'}, {'label': 'Somalia', 'value': 'Somalia'}, {'label': 'Afghanistan', 'value': 'Afghanistan'}, {'label': 'Guinea-Bissau', 'value': 'Guinea-Bissau'}, {'label': 'Oman', 'value': 'Oman'}, {'label': 'Slovenia', 'value': 'Slovenia'}, {'label': 'Libya', 'value': 'Libya'}, {'label': 'Indonesia', 'value': 'Indonesia'}, {'label': 'Cambodia', 'value': 'Cambodia'}, {'label': 'Austria', 'value': 'Austria'}, {'label': 'Switzerland', 'value': 'Switzerland'}, {'label': 'Benin', 'value': 'Benin'}, {'label': 'Moldova', 'value': 'Moldova'}, {'label': 'Azerbaijan', 'value': 'Azerbaijan'}, {'label': 'Chile', 'value': 'Chile'}, {'label': 'Singapore', 'value': 'Singapore'}, {'label': 'Nicaragua', 'value': 'Nicaragua'}, {'label': 'Costa Rica', 'value': 'Costa Rica'}, {'label': 'Timor-Leste', 'value': 'Timor-Leste'}, {'label': 'Liechtenstein', 'value': 'Liechtenstein'}, {'label': 'Albania', 'value': 'Albania'}, {'label': 'Belarus', 'value': 'Belarus'}, {'label': 'San Marino', 'value': 'San Marino'}, {'label': 'Niger', 'value': 'Niger'}, {'label': 'Croatia', 'value': 'Croatia'}, {'label': 'Comoros', 'value': 'Comoros'}, {'label': 'Burundi', 'value': 'Burundi'}, {'label': 'Qatar', 'value': 'Qatar'}, {'label': 'India', 'value': 'India'}, {'label': 'Gambia', 'value': 'Gambia'}, {'label': 'Bulgaria', 'value': 'Bulgaria'}, {'label': 'Tunisia', 'value': 'Tunisia'}, {'label': 'Syria', 'value': 'Syria'}, {'label': 'Kosovo', 'value': 'Kosovo'}, {'label': 'Turkey', 'value': 'Turkey'}, {'label': 'Maldives', 'value': 'Maldives'}, {'label': 'Estonia', 'value': 'Estonia'}, {'label': 'United Arab Emirates', 'value': 'United Arab Emirates'}, {'label': 'Kuwait', 'value': 'Kuwait'}, {'label': 'Lesotho', 'value': 'Lesotho'}, {'label': 'Kazakhstan', 'value': 'Kazakhstan'}, {'label': 'Jordan', 'value': 'Jordan'}, {'label': 'North Macedonia', 'value': 'North Macedonia'}, {'label': 'Bosnia and Herzegovina', 'value': 'Bosnia and Herzegovina'}, {'label': 'Ethiopia', 'value': 'Ethiopia'}, {'label': 'Antigua and Barbuda', 'value': 'Antigua and Barbuda'}, {'label': 'Venezuela', 'value': 'Venezuela'}, {'label': 'Guyana', 'value': 'Guyana'}, {'label': 'Brunei', 'value': 'Brunei'}, {'label': 'Sri Lanka', 'value': 'Sri Lanka'}, {'label': 'West Bank and Gaza', 'value': 'West Bank and Gaza'}, {'label': 'Romania', 'value': 'Romania'}, {'label': 'Saudi Arabia', 'value': 'Saudi Arabia'}, {'label': 'Central African Republic', 'value': 'Central African Republic'}, {'label': 'Brazil', 'value': 'Brazil'}, {'label': 'Djibouti', 'value': 'Djibouti'}, {'label': 'Hungary', 'value': 'Hungary'}, {'label': 'Belgium', 'value': 'Belgium'}, {'label': 'Senegal', 'value': 'Senegal'}, {'label': 'Ecuador', 'value': 'Ecuador'}, {'label': 'Liberia', 'value': 'Liberia'}, {'label': 'Luxembourg', 'value': 'Luxembourg'}, {'label': 'Guinea', 'value': 'Guinea'}, {'label': 'Papua New Guinea', 'value': 'Papua New Guinea'}, {'label': 'Burma', 'value': 'Burma'}, {'label': 'Czechia', 'value': 'Czechia'}, {'label': 'Lithuania', 'value': 'Lithuania'}, {'label': 'Seychelles', 'value': 'Seychelles'}, {'label': 'Sudan', 'value': 'Sudan'}, {'label': 'Tanzania', 'value': 'Tanzania'}, {'label': 'Zimbabwe', 'value': 'Zimbabwe'}, {'label': 'Burkina Faso', 'value': 'Burkina Faso'}, {'label': 'Serbia', 'value': 'Serbia'}, {'label': 'Pakistan', 'value': 'Pakistan'}, {'label': 'Rwanda', 'value': 'Rwanda'}, {'label': 'Greece', 'value': 'Greece'}, {'label': 'Yemen', 'value': 'Yemen'}, {'label': 'Cabo Verde', 'value': 'Cabo Verde'}, {'label': 'Holy See', 'value': 'Holy See'}, {'label': 'Malta', 'value': 'Malta'}]

def decision_tree(x_train, y_train,param_value_1,param_value_2,param_value_3):
    dectree_crit = ['mse', 'friedman_mse', 'mae']
    dectree_max_depth = [1,2,3,4,5]
    dectree_min_samp_split = [2,3,4,5]
    dectree = DecisionTreeClassifier(criterion=param_value_1,
                                    max_depth=param_value_2,
                                    min_samples_split=param_value_3).fit(x_train, y_train)
    return dectree

# lr = LinearRegression().fit(x_train, y_train)

def logistic(x_train, y_train,param_value_1,param_value_2,param_value_3):
    logistic_C = [1, 3, 5, 10, 40, 60, 80, 100]
    logistic_tol = [0.0001,0.0004]
    logistic_solver = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']

    if param_value_1 is None or param_value_2 is None  or param_value_3 is None :
        logis = LogisticRegression(C=logistic_C[0], tol=logistic_tol[0], solver=logistic_solver[0]).fit(x_train, y_train)
    else:
        logis = LogisticRegression(C=param_value_1, tol=param_value_2, solver=param_value_3).fit(x_train, y_train)
    return logis

def ridge(x_train, y_train,param_value_1,param_value_2,param_value_3):
    ridge_alpha = [1,2,3,4,5]
    ridge_tol = [0.0001,0.0004]
    ridge_solver = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    ridge = RidgeClassifier(alpha=param_value_1, tol=param_value_2, solver=param_value_3).fit(x_train, y_train)
    return ridge

def svm_linear(x_train, y_train,param_value_1,param_value_2,param_value_3):
    svmlr_C = [1, 3, 5, 10, 40, 60, 80, 100]
    svmlr_loss = ['epsilon_insensitive', 'squared_epsilon_insensitive']
    svmlr_tolerance = [0.0001,0.0004]
    svmlr = LinearSVC(C=param_value_1, loss=param_value_2, tol=param_value_3).fit(x_train, y_train)
    return svmlr
def sup_vec_m(x_train, y_train,param_value_1,param_value_2,param_value_3):
    svc_C = [1, 3, 5, 10, 40, 60, 80, 100]
    svc_kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    svc_degree = [2,3,4,5,6]
    svc = SVC(C=param_value_1,kernel=param_value_2,degree=param_value_3).fit(x_train, y_train)
    return svc
def randomFor(x_train, y_train,param_value_1,param_value_2,param_value_3):
    rfc_n_est = [10,20,30,50,70,100]
    rfc_max_depth = [1,2,3,4,5]
    rfc_min_samp_split = [2,3,4,5]
    rfc = RandomForestClassifier(n_estimators=param_value_1,
                                 max_depth=param_value_2,
                                 min_samples_split=param_value_3).fit(x_train, y_train)
    return rfc
def GradienBoos(x_train, y_train,param_value_1,param_value_2,param_value_3):
    gbc_learn_rate = [0.1,0.2,0.3,0.4,0.5]
    gbc_crit = ['mse', 'friedman_mse', 'mae']
    gbc_sun_samp = [0.1,0.3,0.5,1,1.3,1.5]
    gbc = GradientBoostingClassifier(learning_rate=param_value_1,
                                     criterion=param_value_2,
                                     subsample=param_value_3).fit(x_train, y_train)
    return gbc

# Future Days
future_days = [{'label': '25', 'value': '25'},
{'label': '50', 'value': '50'},
{'label': '75', 'value': '75'}
                 ]
# Train test s[;ot
train_test_splitter = [{'label': '10%', 'value': '10%'},
{'label': '30%', 'value': '30%'},
{'label': '50%', 'value': '50%'}]

# Dash Dropdown references
models_ref_A = ['Decision Tree', 'Lasso', 'Ridge', 'SVM Linear', 'SVC', "Random Forest", 'Gradient Boosting']
model_list = []
for a in models_ref_A:
    model_list.append({"label": a, "value": a})

# dd = {'Decision Tree': {'Error func': ['mse', 'friedman_mse', 'mae']},
# 'Lasso': {'num': [1,2,3,4,5]}}
# nam = dd['Decision Tree'].keys()
# val = dd['Decision Tree'].items()
# print(list(nam))

mod_param_dict_1 = {
    'Decision Tree': {'criterion': ['mse', 'friedman_mse', 'mae']},
    'Logistic': {'C': [1, 3, 5, 10, 40, 60, 80, 100]},
    'Ridge': {'alpha': [1,2,3,4,5]},
    'SVM Linear': {'C': [1, 3, 5, 10, 40, 60, 80, 100]},
    'SVC': {'C': [1, 3, 5, 10, 40, 60, 80, 100]},
    'Random Forest': {'n_estimators':[10,20,30,50,70,100]},
    'Gradient Boosting': {'learning_rate': [0.1,0.2,0.3,0.4,0.5]},
               }

mod_param_dict_2 = {
    'Decision Tree': {'max_depth': [1,2,3,4,5]},
    'Logistic': {'tol': [0.0001,0.0004]},
    'Ridge': {'tol': [0.0001,0.0004]},
    'SVM Linear': {'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']},
    'SVC': {'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']},
    'Random Forest': {'max_depth':[1,2,3,4,5]},
    'Gradient Boosting': {'criterion': ['mse', 'friedman_mse', 'mae']},
               }
mod_param_dict_3 = {
    'Decision Tree': {'min_samples_split': [2,3,4,5]},
    'Logistic': {'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']},
    'Ridge': {'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']},
    'SVM Linear': {'tol': [0.0001,0.0004]},
    'SVC': {'degree': [2,3,4,5,6]},
    'Random Forest': {'min_samples_split':[2,3,4,5]},
    'Gradient Boosting': {'subsample': [0.1,0.3,0.5,1,1.3,1.5]},
               }

mod_param_model_names = list(mod_param_dict_1.keys())
mod_param_values = mod_param_dict_1[mod_param_model_names[0]]


app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1("Derrick's SGX Listed Companies Overview", style={'text-align': 'centre'}),
    # html.Div([])
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.P('Country'),
                    dcc.Dropdown(id="Country Province",
                                 options=country_dict,
                                 value='Singapore',
                                 style={'width': "40%"}),
                ], className='box-row-col'),

            ], className='box-row'),


            html.Div([
                html.Div([
                    html.P('Model'),
                    dcc.Dropdown(id="Model",
                                 options=[{'label': k, 'value': k} for k in mod_param_dict_1.keys()],
                                 value='Logistic',
                                 # style={'width': "40%"}
                                 ),
                ],className='box-row1-col1'),
                html.Div([
                    html.P('Future Days'),
                    dcc.Dropdown(id="Future_Days",
                                 options=future_days,
                                 value='50',
                                 # style={'width': "40%"}
                                 ),
                ],className='box-row1-col2'),

                html.Div([
                    html.P('Train Test Split'),
                    dcc.Dropdown(id="Train_Test_Split",
                                 options=train_test_splitter,
                                 value='10%',
                                 # style={'width': "40%"}
                                 ),
                ],className='box-row1-col3'),
            ], className='box-row1'),

            html.Div([
                html.Div([
                    html.P(id='param_1_name'),
                    dcc.Dropdown(id="param_1_value",
                                 multi=False,
                                 # value=1,
                                 # style={'width': "40%"}
                                 ),
                ],className='box-row2-col1'),
                html.Div([
                    html.P(id='param_2_name'),
                    dcc.Dropdown(id="param_2_value",
                                 # value='totalRevenue',
                                 # style={'width': "40%"}
                                 ),
                ],className='box-row2-col2'),
                html.Div([
                    html.P(id='param_3_name'),
                    dcc.Dropdown(id="param_3_value",
                                 # value='totalRevenue',
                                 # style={'width': "40%"}
                                 ),
                ],className='box-row2-col3'),



            ], className='box-row2'),



        ]),
        html.Div([
            dcc.Graph(id='my_bee_map', figure={}),
        ], className='chart-A-container'),
        # html.Div([
        #     dcc.Graph(id='my_bee_map2', figure={}),
        # ], className='chart-A-container'),

    ]),



], style={'background-color': 'Black', 'width': '100%', 'Height':'100%'}, className='BodyBack')

app.css.append_css({
    'external_url': "https://codepen.io/chriddyp/pen/bWLwgP.css"
})
# ------------------------------------------------------------------------------
# Model Parameter 1
@app.callback(
    [dash.dependencies.Output('param_1_name', 'children'),
     dash.dependencies.Output('param_1_value', 'options')],
    [dash.dependencies.Input('Model', 'value')])
def set_cities_options(selected_model):
    param_name = list(mod_param_dict_1[selected_model].keys())[0]
    param_values = list(mod_param_dict_1[selected_model].items())[0][1]
    param_values = [{'label': k, 'value': k} for k in param_values]
    return param_name, param_values
# Model Parameter 2
@app.callback(
    [dash.dependencies.Output('param_2_name', 'children'),
     dash.dependencies.Output('param_2_value', 'options')],
    [dash.dependencies.Input('Model', 'value')])
def set_cities_options(selected_model):
    param_name = list(mod_param_dict_2[selected_model].keys())[0]
    param_values = list(mod_param_dict_2[selected_model].items())[0][1]
    param_values = [{'label': k, 'value': k} for k in param_values]
    #print(param_name)
    #print(param_values)
    return param_name, param_values
# Model Parameter 3
@app.callback(
    [dash.dependencies.Output('param_3_name', 'children'),
     dash.dependencies.Output('param_3_value', 'options')],
    [dash.dependencies.Input('Model', 'value')])
def set_cities_options(selected_model):
    param_name = list(mod_param_dict_3[selected_model].keys())[0]
    param_values = list(mod_param_dict_3[selected_model].items())[0][1]
    param_values = [{'label': k, 'value': k} for k in param_values]
    return param_name, param_values


# Connect the Plotly graphs with Dash Component
@app.callback(
    # [Output(component_id='output_container', component_property='children'),
    #  Output(component_id='my_bee_map', component_property='figure')],
    Output(component_id='my_bee_map', component_property='figure'),

    [Input(component_id='Country Province', component_property='value'),
     Input(component_id='Model', component_property='value'),
        Input(component_id='Future_Days', component_property='value'),
        Input(component_id='Train_Test_Split', component_property='value'),
        Input(component_id='param_1_name', component_property='value'),
        Input(component_id='param_1_value', component_property='value'),
        Input(component_id='param_2_name', component_property='value'),
        Input(component_id='param_2_value', component_property='value'),
        Input(component_id='param_3_name', component_property='value'),
        Input(component_id='param_3_value', component_property='value'),
     ]
)
def update_graph(option_ctry, option_model, option_future_days, option_tran_split,
                 option_param_n1, option_param_v1,
                 option_param_n2, option_param_v2,
                 option_param_n3, option_param_v3):
    df = pd.read_csv(maindata_dir)
    df_original = df.loc[df['Country_Province']==option_ctry]

    # df_original = df_original.loc[df_original['day_since100'] > 0]

    df = df_original[1:][['case_cnt']]  # 'date',

    # src_dir2 = '/Users/derrick/PycharmProjects/DC_DB_1/covid19/research/DELPHI/data/processed/'
    #sectorName2 = 'Cases_Singapore_None'
    #src_file2 = src_dir2 + sectorName2 + '.csv'
    #df = pd.read_csv(src_file2)[1:][['case_cnt']]  # 'date',
    df_work = df.copy()
    df_work['Label'] = range(1, len(df_work) + 1)


    # DATA PREPARATION
    df_date_ref = df_original[1:].copy()
    df_date_ref['Label'] = range(1, len(df_work) + 1)

    y_response = 'case_cnt'
    y_prediction = 'Prediction'
    x_explain = 'Label'
    future_days = int(option_future_days) # 25
    option_tran_split = int(str(option_tran_split).replace("%",""))/100
    option_tran_split = float(option_tran_split)

    df_work[y_prediction] = df_work[[y_response]].shift(-future_days)
    # try:
    #     df_work.drop(['date'])
    # except:
    #     pass
    X = np.array(df_work.drop([y_prediction], 1))[:-future_days]
    y = np.array(df_work[y_prediction])[:-future_days]


    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=option_tran_split)

    param_name_1, param_name_2, param_name_3 = option_param_n1, option_param_n2, option_param_n3
    param_value_1, param_value_2, param_value_3 = option_param_v1, option_param_v2, option_param_v3

    model_i = LogisticRegression().fit(x_train, y_train)
    # Run default particular model

    # Run selected particular model
    if option_model == 'Decision Tree':
        model_i = decision_tree(x_train, y_train, param_value_1,param_value_2,param_value_3)
    if option_model == 'Lasso':
        model_i = logistic(x_train, y_train, param_value_1,param_value_2,param_value_3)
    if option_model == 'Ridge':
        model_i = ridge(x_train, y_train, param_value_1,param_value_2,param_value_3)
    if option_model == 'SVM Linear':
        model_i = svm_linear(x_train, y_train, param_value_1,param_value_2,param_value_3)
    if option_model == 'SVC':
        model_i = sup_vec_m(x_train, y_train, param_value_1,param_value_2,param_value_3)
    if option_model == 'Random Forest':
        model_i = randomFor(x_train, y_train, param_value_1,param_value_2,param_value_3)
    if option_model == 'Gradient Boosting':
        model_i = GradienBoos(x_train, y_train, param_value_1,param_value_2,param_value_3)

    x_future = df_work.drop([y_prediction], 1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)

    mod_i_prediction = model_i.predict(x_future)

    predictions = mod_i_prediction
    valid = df_work[X.shape[0]:]

    valid[y_prediction] = predictions

    date_col = 'date'
    #print(df_work.columns)


    df_work2 = pd.DataFrame()
    valid2 = pd.DataFrame()
    df_work2[date_col] = pd.merge(df_work, df_date_ref, on=[x_explain])[date_col]
    valid2[date_col] = pd.merge(valid, df_date_ref, on=[x_explain])[date_col]
    fig = px.scatter(x=list(df_work2[date_col]), y=list(df_work[y_response]),
                     template='plotly_dark',width=600, height=400)
    # fig = px.scatter(data_frame=df_work, x='date',y=y_response, template='plotly_dark')
    #fig.add_trace(px.line(data_frame=valid, x='Label', y=y_prediction))
    fig.add_scatter(mode='lines',x=valid2[date_col], y=valid[y_prediction])
    cht1_title = '{} model projection for {}'.format(option_model,option_ctry)
    fig.update_layout(title=cht1_title,
                      xaxis_title="Date",  yaxis_title="Infected Numbers",

                      )


    return fig








if __name__ == '__main__':
    app.run_server(debug=True)
    x=1
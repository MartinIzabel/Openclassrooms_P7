# -*- coding: utf-8 -*-
from xmlrpc.client import boolean
import pandas as pd
import streamlit as st
import numpy as np
import requests
from sklearn import preprocessing
import joblib
import lightgbm
import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib as plt
import seaborn as sns
from lime import lime_tabular

#######################################################################################
# To run this code, type in terminal at the file path: 
# cd C:\Users\User\Desktop\OC\P7_IZABEL_MARTIN\Openclassrooms_P7
# streamlit run app.py
# MDP heroku : 2V3kpt29v!
#######################################################################################

#Load the data
app_test = pd.read_csv('app_test.csv', sep = ",",  index_col='SK_ID_CURR')

# Load the model
model = joblib.load("model_vf.pkl")

#listes des features
features = list(app_test.columns)

#Preparation des predictions
seuil = 0.7539816036060938
X_test = preprocessing.StandardScaler().fit_transform(app_test)
# app_test['prediction'] = (model.predict_proba(X_test)[:,1])
# app_test_bool = app_test
# app_test_bool['prediction_label'] =  (model.predict_proba(X_test)[:,1] > seuil).astype(bool)

COLOR_BR_r = ['#EF553B', '#00CC96']
COLOR_BR = ['indianred', 'dodgerblue']


test_query = requests.get("https://fastapi-p7.herokuapp.com/score")
score_df = pd.DataFrame.from_dict(test_query.json())
score_df['prediction_label'] = (score_df['prediction'] > seuil).astype(bool)
# score_df
score_df.index = score_df.index.astype("int")

app_test = app_test.merge(score_df, left_index=True, right_index=True)
# app_test

# url = 'https://fastapi-p7.herokuapp.com/predict'
# input = X_test[0]
# st.markdown(type(input))

# resp = requests.post(url, data=input)
# st.markdown(resp.content)

#######################################################################################

#Select your ID 
option = st.sidebar.selectbox(
    'Select your Client SK_ID',
     app_test.index)

#######################################################################################

st.title("Bienvenue sur le dashboard Prêt à dépenser\n ----") 
st.subheader("Déterminez l'acceptation ou le rejet d'un dossier de nos crédit à la consommation")

list_virer = ['prediction', 'prediction_label']
excl_app = app_test.drop(list_virer , axis = 1)

df_expl = pd.DataFrame(X_test, index = app_test.index, columns = list(excl_app.columns) )    
z = np.array(df_expl.loc[{option}]).reshape(-1, )

left_column, right_column = st.columns(2)
left_column.markdown(f'ID Client: {option}')

prob_client = app_test['prediction'].loc[option]

if left_column.button('Predire !'):
    left_column.markdown(f'Score Client: {round(prob_client, 3)}')

    if prob_client > seuil:
        st.header("Prêt refusé  :disappointed:")
    else:
        st.header("Prêt accordé  :sunglasses:")

    fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value =  prob_client*100,
    mode = "gauge+number",
    gauge = {'axis': {'range': [None, 100]},
             'steps' : [
                        
                {'range': [0, 100], 'color': "gray"}],
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': seuil*100}}))

    st.subheader("Probabilité d'insolvabilité vs seuil.")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.pie(values=[prob_client, prob_client] , names=[0,1], color=[0,1], color_discrete_sequence=COLOR_BR_r, width=230, height=230)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))


    st.subheader("Ratio prêt accordé vs non-accordés")
    x = [1, 2, 3, 4, 10]
    fig, ax = plt.pyplot.subplots()
    ax.pie(app_test['prediction_label'].value_counts(), labels = ["Accordés", "Refusés"],
               explode = [0, 0.2],
               autopct = lambda x: str(round(x, 2)) + '%',
               pctdistance = 0.7, labeldistance = 1.4,
               shadow = True)
    st.pyplot(fig)

    st.subheader("Pourquoi ce score ?")
    explainer = lime_tabular.LimeTabularExplainer(training_data =  X_test, mode = 'classification', feature_names =  list(excl_app.columns))
    exp = explainer.explain_instance(data_row = z , predict_fn = model.predict_proba)

    # with plt.pyplot.style.context('grayscale'):
    #     fig = exp.as_pyplot_figure()
    #     st.pyplot(fig)
    
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)
   
 
# #######################################################################################

# #Extract feature importances
feature_importance_values = model.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
top_10_features = feature_importances.sort_values('importance', ascending=False)[:10]
top_5_features = feature_importances.sort_values('importance', ascending=False)[:5]
feature_client = list(top_10_features['feature'])

other_features = [('229_other_features', feature_importances[11:]['importance'].sum())]   
df_other_ft = pd.DataFrame(other_features, columns = ['feature', 'importance'] )
top_11_features = top_10_features.append(df_other_ft, ignore_index=True)

# #######################################################################################  

client_data = {}
for col in feature_client:
    client_data[col] = [app_test[col].loc[option]]   

df_client = pd.DataFrame(client_data, index = None)
 

st.subheader('Explorez vos données')


st.subheader("Importance des variables dans la prise de décision")
x = [1, 2, 3, 4, 10]
fig, ax = plt.pyplot.subplots(figsize = (6, 6))
ax.pie(top_11_features['importance'], labels = top_11_features['feature'],
               explode = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
               autopct = lambda x: str(round(x, 2)) + '%',
               pctdistance = 0.7, labeldistance = 1.4,
               shadow = True)
st.pyplot(fig)

st.subheader('Vos données')
df_client

# list_virer = ['prediction', 'prediction_label']
# excl_app = app_test.drop(list_virer , axis = 1)

# df_expl = pd.DataFrame(X_test, index = app_test.index, columns = list(excl_app.columns) )    
# z = np.array(df_expl.loc[{option}]).reshape(-1, )
    # z.shape
    # X_test[0,:].shape
    # explainer = lime_tabular.LimeTabularExplainer(training_data =  X_test, mode = 'classification', feature_names = list(app_test.columns))
    # exp = explainer.explain_instance(data_row = X_test[12,:], predict_fn = model.predict_proba)
    # with plt.style.context("ggplot"):
    #     fig = exp.as_pyplot_figure()
    #     st.pyplot(fig)

# explainer = lime_tabular.LimeTabularExplainer(training_data =  X_test, mode = 'classification', feature_names =  list(excl_app.columns))
# exp = explainer.explain_instance(data_row = z , predict_fn = model.predict_proba)

# with plt.style.context("ggplot"):
#     fig = exp.as_pyplot_figure()
#     st.pyplot(fig)

variable_select = st.selectbox('Selection de la variable à explorer', list(excl_app.columns))   
st.markdown(f'Votre valeur : {app_test[variable_select].loc[option]}')

vline = app_test[variable_select].loc[option]

fig, ax  = plt.pyplot.subplots()

# KDE plot of loans that were repaid on time
sns.kdeplot(app_test.loc[app_test['prediction_label'] == False, variable_select], label = 'target = 0')

# KDE plot of loans which were NOT repaid on time
sns.kdeplot(app_test.loc[app_test['prediction_label'] == True, variable_select], label = 'target = 1')  

# ADD la valeur du client
plt.pyplot.axvline(vline, c= 'black', ls = '--', linewidth=2, label = 'your data')

# Labeling of plot
plt.pyplot.xlabel(variable_select); 
plt.pyplot.ylabel('Densité'); 
plt.pyplot.title('Distribution'); 
plt.pyplot.legend()
st.pyplot(fig)

#######################################################################################
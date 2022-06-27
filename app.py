# -*- coding: utf-8 -*-
# import os
# import pandas as pd
import streamlit as st
# import numpy as np
# import requests
# from sklearn import preprocessing
# import joblib


# matplotlib and seaborn for plotting
# import matplotlib as plt
# import seaborn as sns

#######################################################################################
# To run this code, type in terminal at the file path: 
# cd C:\Users\User\Desktop\OC\P7_IZABEL_MARTIN
# streamlit run app.py
#######################################################################################

# pathabsolutedir = os.path.dirname(os.path.abspath(__file__))

#Load the data
# app_test = pd.read_csv('app_test.csv', sep = ",",  index_col='SK_ID_CURR')

# Load the model
# model = joblib.load("model_gb.pkl")

#listes des features
# features = list(app_test.columns)

#Preparation des predictions
# X_test = sklearn.preprocessing.StandardScaler().fit_transform(app_test)
# app_test['prediction'] = (model.predict_proba(X_test)[:,1] >= 0.48).astype(bool)

#######################################################################################

#Select your ID 
# option = st.sidebar.selectbox(
#     'Select your Client SK_ID',
#      app_test.index)

#######################################################################################

st.title("Bienvenue sur le dashboard Prêt à dépenser\n ----") 
st.subheader("Déterminez l'acceptation ou le rejet d'un dossier de nos crédit à la consommation")

# left_column, right_column = st.columns(2)

# left_column.markdown(f'ID Client: {option}')

# if left_column.button('Predire !'):
#     if app_test['prediction'].loc[option] == True:
#         st.header("Prêt non accordé  :disappointed:")
#     else:
#         st.header("Prêt accordé  :sunglasses:")
    
#     st.subheader("Ratio de prêt accordés")

#     x = [1, 2, 3, 4, 10]
#     fig, ax = plt.pyplot.subplots(figsize = (4, 4))
#     ax.pie(app_test['prediction'].value_counts(), labels = ["Accordé", "Refusé"],
#                explode = [0, 0.2],
#                autopct = lambda x: str(round(x, 2)) + '%',
#                pctdistance = 0.7, labeldistance = 1.4,
#                shadow = True)
#     st.pyplot(fig)

 
# #######################################################################################

# #Extract feature importances
# feature_importance_values = model.feature_importances_
# feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
# top_10_features = feature_importances.sort_values('importance', ascending=False)[:10]
# feature_client = list(top_10_features['feature'])

# other_features = [('229_other_features', feature_importances[11:]['importance'].sum())]   
# df_other_ft = pd.DataFrame(other_features, columns = ['feature', 'importance'] )
# top_11_features = top_10_features.append(df_other_ft, ignore_index=True)

# #######################################################################################  


# client_data = {}
# for col in feature_client:
#     client_data[col] = app_test[col].loc[option]   
 

# if left_column.button('Get details !'):
    
#     client_data

#     st.subheader("Importance des variables dans la prise de décision")

#     x = [1, 2, 3, 4, 10]
#     fig, ax = plt.pyplot.subplots(figsize = (6, 6))
#     ax.pie(top_11_features['importance'], labels = top_11_features['feature'],
#                explode = [0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
#                autopct = lambda x: str(round(x, 2)) + '%',
#                pctdistance = 0.7, labeldistance = 1.4,
#                shadow = True)
#     st.pyplot(fig)

# "EXT_SOURCE_1"
# "DAYS_BIRTH":
# "EXT_SOURCE_3":
# "EXT_SOURCE_2":
# "AMT_CREDIT":
# "AMT_GOODS_PRICE":
# "AMT_ANNUITY":
# "DAYS_ID_PUBLISH":
# "DAYS_EMPLOYED":
# "DAYS_LAST_PHONE_CHANGE":

# variable_select = st.selectbox(
# 'Select your feature to explore',feature_client)


# st.markdown(f'Votre valeur : {client_data[variable_select]}')


# chart_data = app_test[variable_select].value_counts()
# st.line_chart(chart_data)


#######################################################################################

# if __name__ == "__main__":
#     print("Script runned directly")
# else:
#     print("Script called by other")
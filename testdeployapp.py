#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from prediction import predict


# In[6]:


header = st.container()
with header:
    st.title('Prediciendo precio de casa')


# In[7]:


col1, col2 = st.columns(2)
with col1:
    longitude = st.number_input('Latitud del lugar', value=-124.35)
    latitude = st.number_input('Longitud del lugar', value=-124.35)
    housing_median_age = st.number_input('Años casa', value=-124.35)
    total_rooms = st.number_input('Total cuartos', value=-124.35)
    total_bedrooms = st.number_input('Total de baños', value=-124.35)
with col2:
    population = st.number_input('Poblacion', value=-124.35)
    households = st.number_input('casas', value=-124.35)
    median_income = st.number_input('ingreso medio', value=-124.35)
    ocean_proximity = st.selectbox('¿En qué zona te gustaría?', options=["ISLAND","NEAR BAY", "NEAR OCEAN", "INLAND", "<1H OCEAN"], index = 0)


# In[8]:


st.header('modelos')
model = st.selectbox('¿Qué tipo de modelo de Machine Learning quieeras usar para tu predicción?', ['Linear Regression','Decision Tree', 'Random Forest'], index = 0)


# In[9]:


if st.button ('Toca para predecir el precio de la casa'):
        data = pd.DataFrame({
            'longitude' : [longitude],
            'latitude' : [latitude],
            'housing_median_age' : [housing_median_age],
            'total_rooms' : [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population' : [population],
            'households': [households],
            'median_income' : [median_income],
            'ocean_proximity' : [ocean_proximity]
            })
        if model == 'Linear Regression':
            result = predict(data, 'lin_regression.sav')
        elif model == 'Decision Tree':
            result = predict(data, 'tree_regression.sav')
        elif model == 'Random Forest':
            result = predict(data, 'forest_regression.sav')
    
        st.text(f'El precio de la casa es de: ${result[0]}')


#!/usr/bin/env python
# coding: utf-8

# In[111]:


import streamlit as st
import pandas as pd
import os
import tarfile
import urllib.request
import joblib
from sklearn.linear_model import LinearRegression


# In[112]:


def predict(data, model_name):
    pipeline= joblib.load('pipeline.sav')
    transformed_data = pipeline.transform(data)
    return model.predict(transformed_data)


# In[113]:


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
fetch_housing_data()
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[114]:


header = st.container()
dataset = st.container()
inputs = st.container()
modelTraining = st.container()


# In[115]:


with header:
    st.title('Housing Project Prediction')


# In[116]:


with dataset:
    st.header('Housing Dataset')
    st.text('Se muestran la información que existe en nuestra base de datos')
    housing = load_housing_data()
    st.write(housing.head())


# In[117]:


with inputs:
    st.header('Inputs del modelo')
    st.text('Selecciona los inputs para poder predecir el precio de la casa de tus sueños')
    
    sel_col, disp_col= st.columns(2)
    rooms = sel_col.slider('Número de cuartos', min_value=1, max_value= 10, value= 1, step=1)
    bathrooms = sel_col.slider('Número de baños', min_value=1, max_value= 10, value= 1, step=1)
    location = sel_col.selectbox('¿En qué zona te gustaría?', options=["ISLAND","NEAR BAY", "NEAR OCEAN", "INLAND", "<1H OCEAN"], index = 0)
    model = sel_col.selectbox('¿Qué tipo de modelo de Machine Learning quieeras usar para tu predicción?', options=["Linear Regression","Decision Tree", "Random Forest"], index = 0)
    
   
    


# In[124]:


with modelTraining:
    st.header('Resultados del Modelo de ML')
  
    loaded_model = joblib.load('lin_regression.sav')
    result = loaded_model.score(housing_labels, housing_predictions)
    disp_col.subheader('Resultado del modelo de ML')
    disp_col.write(result)


# In[ ]:





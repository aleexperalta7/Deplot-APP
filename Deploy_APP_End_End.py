#!/usr/bin/env python
# coding: utf-8

# In[82]:


import streamlit as st
import pandas as pd
import os
import tarfile
import urllib.request
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


# In[83]:


def predict(data, model_name):
    model = joblib.load(f'{model_name}')
    pipeline= joblib.load('pipeline.sav')
    transformed_data = pipeline.transform(data)
    return model.predict(transformed_data)


# In[84]:


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


# In[92]:


housing = load_housing_data()


# In[100]:


imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)


# In[102]:


X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)


# In[101]:


rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): 
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self 
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[94]:


num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

# In[95]:


header = st.container()
dataset = st.container()
inputs = st.container()
modelTraining = st.container()


# In[96]:


with header:
    st.title('Housing Project Prediction')


# In[97]:


with dataset:
    st.header('Housing Dataset')
    st.text('Se muestran la información que existe en nuestra base de datos')
    housing = load_housing_data()
    st.write(housing.head())


# In[98]:


with inputs:
    st.header('Inputs del modelo')
    st.text('Selecciona los inputs para poder predecir el precio de la casa de tus sueños')
    
    sel_col1, sel_col2= st.columns(2)
    longitude = sel_col1.number_input('Latitud del lugar', value=-124.35)
    latitude = sel_col1.number_input('Longitud del lugar', value=32.54)
    housing_median_age = sel_col1.slider('Años promedios de la casa', min_value=1, max_value= 52, value= 1, step=1)
    total_rooms = sel_col1.slider('Total de cuartos', min_value=2, max_value= 39320, value= 2, step=1)
    total_bedrooms = sel_col1.slider('Total de baños', min_value=1, max_value= 6445, value= 1, step=1)
    population = sel_col2.slider('Poblacion total', min_value=3, max_value= 35682, value= 3, step=1)
    households = sel_col2.slider('Tamaño de personas viviendo en la casa', min_value=1, max_value= 6082, value= 1, step=1)
    median_income = sel_col2.number_input('Ingreso medios', value=0.4999)
    ocean_proximity = sel_col2.selectbox('¿En qué zona te gustaría?', ['ISLAND','NEAR BAY', 'NEAR OCEAN', 'INLAND', '<1H OCEAN'], index = 0)
    model = sel_col2.selectbox('¿Qué tipo de modelo de Machine Learning quieeras usar para tu predicción?', ['Linear Regression','Decision Tree', 'Random Forest'], index = 0)
    
    


# In[99]:


with modelTraining:
    st.header('Resultados del Modelo de ML')
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
  
  


#!/usr/bin/env python
# coding: utf-8

# In[58]:


import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from prediction import predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# In[59]:


columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] 
# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/aleexperalta7/Streamlit/main/Iris_project/iris.data', names=columns)
df.head()


# In[60]:


data = df.values
X = data[:,(2,3)]
Y = data[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[61]:


def predict(data, model_name):
    model = joblib.load(f'{model_name}')
    pipeline= joblib.load('iris_pipeline.sav')
    transformed_data = pipeline.transform(data)
    return model.predict(transformed_data)


# In[62]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

iris_pipeline = Pipeline([("std_scaler", StandardScaler())])


# In[63]:


X_train_tr = iris_pipeline.fit_transform(X_train)


# In[64]:


header = st.container()
dataset = st.container()
inputs = st.container()
modelTraining = st.container()


# In[65]:


with header:
    st.title('Clasificador flores')


# In[66]:


with dataset:
    st.text('información de la base de datos de Iris')
    st.write(df.head())


# In[67]:


with inputs:
    st.header('Iris Data Inputs')
    st.text('Seleccion el ancho o largo del pétalo de la flor: ')
   
    sel_col1, sel_col2= st.columns(2)
    petal_length = sel_col1.slider('Largo del pétalo', value=1.0, min_value=0.0, max_value = 6.9, step=0.01)
    petal_width = sel_col1.slider('Ancho del pétalo', value=1.0, min_value=0.0, max_value = 2.5, step=0.01)
    model = sel_col1.selectbox('¿Qué tipo de modelo de Machine Learning quieres usar para tu clasificación?', ['Logistic Regression','Support Vector Machine', 'Decision Tree', 'Voting Classifier'], index = 0)


# In[68]:


with modelTraining:
    st.header('Resultados del Modelo de ML')
    if st.button ('Toca para clasificar el tipo de flor'):
        data = pd.DataFrame({
            'Largo del pétalo' : [petal_length],
            'Ancho del pétalo' : [petal_width]
            })
        if model == 'Support Vector Machine':
            result = predict(data, 'SVM.sav')
        elif model == 'Logistic Regression':
            result = predict(data, 'Log_reg.sav')
        elif model == 'Decision Tree':
            result = predict(data, 'Dec_tree.sav')
        elif model == 'Voting Classifier':
            result = predict(data, 'vot_clf.sav')
            
        if result == 0:
            result = "Setosa"
        if result == 1:
            result = "Versicolor"
        if result == 2:
            result = "Virginica"
    
        st.text(f'La clasificación de la flor es: {result[0]}')


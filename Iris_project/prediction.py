#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib

def predict(data, model_name):
    model = joblib.load(f'{model_name}')
    pipeline= joblib.load('iris_pipeline.sav')
    transformed_data = pipeline.transform(data)
    return model.predict(transformed_data)


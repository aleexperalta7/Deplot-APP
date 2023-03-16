#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def predict(data, model_name):
    model = joblib.load({model_name})
    pipeline= joblib.load('pipeline.sav')
    transformed_data = pipeline.transform(data)
    return model.predict(transformed_data)


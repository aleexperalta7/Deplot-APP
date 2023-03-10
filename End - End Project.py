#!/usr/bin/env python
# coding: utf-8

# # END- END PROJECT ALEJANDRO PERALTA

# #### Primero debemos importar nuestras librerias y tambien importar nuestro DATASET que vamos a estar utilizando

# In[2]:


import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor



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


# ### Se usa el pd.read_csv para poder abrir nuestro DATASET que esta enfomrato .csv

# In[3]:


fetch_housing_data()
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# ### Podemos ver como esta distribuido nuesto DATASET

# In[4]:


housing = load_housing_data()
housing.head()


# ### Aqui podemos apreciar los valores proemdio y la cantidad de valores por parametro, en este caso queremos saber sobre el tag "Ocean_proximity"

# In[5]:


print(housing["ocean_proximity"].value_counts())
print(housing.describe())


# ### Representación gráfica sobre los datos que tenemos

# In[6]:


import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.title("attribute_histogram_plots")
plt.show()


# ### Aquí se hace un split de los datos para encontrar una distruición adecuada, ya que en las imagenes pasadas no nos dice mucho sobre el comportamiento de los datos y creamos una nueva variable "income_cat"

# In[7]:


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[8]:


train_set, test_set = split_train_test(housing, 0.2)
len(train_set)
len(test_set)
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[9]:


housing["income_cat"] = pd.cut(housing["median_income"], bins=[0.,1.5,3.0,4.5,6.,np.inf], labels=[1,2,3,4,5])
housing["income_cat"].hist()


# ### aqui hacemos un split random de nuestros datos para que los pueda agarrar como TEST

# In[10]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# ### Podemos ver en la representación de abajo, como es la visualización de los datos usando longitud y latitud, la cual nos muestra la costa de california 

# In[11]:


housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()


# ### se hace una matriz de correlación de los datos para ver cuales se tienen mas distribucion entre ellos

# In[12]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))


# In[13]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["popultaion_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# ### Aqui es donde empezamos a preparar nuestros datos para aplicar algun algoritmo de ML, los cuales tenemos que quitar los datos de predicción de pasos atrás. Se pueden usar técnicas para los datos faltantes como se muestra en la parte de abajo, ya que estos datos que no estan completos, nos pueden ocasionar ruido en nuestro modelo

# In[14]:


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing.dropna(subset=["total_bedrooms"])    # opcion 1
housing.drop("total_bedrooms", axis=1)       # opcion 2
median = housing["total_bedrooms"].median()  # opcion 3
housing["total_bedrooms"].fillna(median, inplace=True)


# ### Aqui podemos ver una funcion mas sencilla sobre el drop de los datos, solo debemos de poner que es lo que queremos hacer con ellos, en este caso usaremos la media para rellenar los espacios vacios

# In[15]:


imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)


# In[16]:


imputer.statistics_


# In[17]:


housing_num.median().values


# In[18]:


X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)


# ### Dentro de nuesto DATASET existe un parametro que es letra, entonces podemos cambiar eso por numero para asi nuestro modelo de ML no tenga ningun inconveniente

# In[19]:


housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


# In[20]:


ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# In[21]:


ordinal_encoder.categories_


# ### Aquí crearemos una matriz de 0 y 1 para mostrar si la categoria es la que busacamos, si es correcto se pone un 1, si no se pone un 0

# In[22]:


cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[23]:


housing_cat_1hot.toarray()


# ### Utilizamos una técnica de Transformers para poder crear nuevos atributos

# In[24]:



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


# ### También podemos usar pipline que nos ayuda a como ensamblar ciertos parametros

# In[25]:


num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[26]:


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# In[27]:


housing_prepared


# In[28]:


#obtener el sav

from joblib import dump
dump(housing_prepared, 'pipeline.sav')


# ### Ahora si ya podemos utilizar un modelo de ML para la parte de la predicción

# In[29]:


lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# ### Vemos nuestros datos de predicción

# In[30]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))


# In[31]:


print("Labels:", list(some_labels))


# ### Y vemos cual es nuestro error sobre la predicción

# In[32]:


housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[41]:


#GridSearchCV Linear Regression
from sklearn.model_selection import GridSearchCV

param_grid = {'fit_intercept':[True,False], 'copy_X':[True, False], 'positive': [True, False]}

lin_reg_grid = GridSearchCV(lin_reg, param_grid, cv=5)

lin_reg_grid.fit(housing_prepared, housing_labels)


# In[42]:


print('Best Score: ', lin_reg_grid.best_score_) 
print('Best Params: ', lin_reg_grid.best_params_)


# In[45]:


X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
lin_reg_grid_pred = lin_reg_grid.predict(X_test_prepared)

lin_reg_grid_pred_mse = mean_squared_error(y_test, lin_reg_grid_pred)
lin_reg_grid_pred_rmse = np.sqrt(lin_reg_grid_pred_mse)


# In[46]:


lin_reg_grid_pred_rmse


# In[47]:


from joblib import dump
dump(lin_reg_grid_pred_rmse, 'lin_regression.sav')


# ### También podemos usar otro modelo 

# In[48]:


tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# ### Podemos apreciar que nuestro error es 0.0, pero no necesarimente es correcto ya que puede ser que nuestro modelo este sobreentrenado

# In[49]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# ### Para validar nuestro error pasado, usamos un cross validation para hacer un split de test en el split de test

# In[50]:


scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# ### Y efectivamente podemos ver que nuestro error no es 0.0

# In[51]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


# In[52]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[53]:


#GridSearchCV Decission Tree Regressor
param_grid = [{'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 
     'splitter': ['best', 'random'], 'max_depth': [None, 2, 4, 6, 8],
     'max_features': [None, 'sqrt', 'log2']},]

tree_reg_grid = GridSearchCV(tree_reg, param_grid, cv=5)

tree_reg_grid.fit(housing_prepared, housing_labels)


# In[54]:


print('Best Score: ', tree_reg_grid.best_score_) 
print('Best Params: ', tree_reg_grid.best_params_)


# In[56]:


X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
tree_reg_grid_pred = tree_reg_grid.predict(X_test_prepared)

tree_reg_grid_pred_mse = mean_squared_error(y_test, tree_reg_grid_pred)
tree_reg_grid_pred_rmse = np.sqrt(tree_reg_grid_pred_mse)


# In[57]:


tree_reg_grid_pred_rmse


# In[58]:


from joblib import dump
dump(tree_reg_grid_pred_rmse, 'tree_regression.sav')


# ### Podemos usar otro modelo de ML

# In[59]:


forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


# ### y vemos que nuestro error es menor que el de LR  y que el DT

# In[60]:


housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[61]:


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# In[68]:


#GridSearchCV Random Forest Regressor
param_grid = { 'max_features': [None, 'sqrt'],
               'max_depth': [None, 2, 4, 6, 8],
               'min_samples_split': [2,5],
               'min_samples_leaf': [1,2],
               'bootstrap': [True, False]}

forest_reg_grid = GridSearchCV(forest_reg, param_grid)

forest_reg_grid.fit(housing_prepared, housing_labels)


# In[69]:


print('Best Score: ', forest_reg_grid.best_score_) 
print('Best Params: ', forest_reg_grid.best_params_)


# In[70]:


X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
forest_reg_grid_pred = forest_reg_grid.predict(X_test_prepared)

forest_reg_grid_pred_mse = mean_squared_error(y_test, forest_reg_grid_pred)
forest_reg_grid_pred_rmse = np.sqrt(forest_reg_grid_pred_mse)


# In[71]:


forest_reg_grid_pred_rmse


# In[72]:


from joblib import dump
dump(forest_reg_grid_pred_rmse, 'forest_regression.sav')


# ### Podemos concluir que primero debemso de tener nuestro DATASET preparado para poder ser utilizado, y despues se pueden aplicar modelos de ML, en este caso se implenetaron 3 tipos y obtuvimos sus resultados. Se hizo de manera manual, ya que para poder aplicar modelos de ML se puede utilizar un GridSerachCV, el cual podemos ponerle hiperparamtros y que nos diga con cual es la mejor combinación para tener un resultado mas óptimo

"""
Módulo para limpieza y enriquecimiento (Feature Engineering) usando Scikit-Learn Pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Índices para las nuevas características
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CreadorAtributos(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

def crear_pipeline_maestro(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Construye y devuelve el Pipeline completo basado en las columnas de entrenamiento.
    """
    housing_num = X_train.select_dtypes(include=[np.number])
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CreadorAtributos()),
        ('std_scaler', StandardScaler()),
    ])

    pipeline_maestro = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
    
    return pipeline_maestro

if __name__ == "__main__":
    print("Módulo de feature engineering configurado con Pipelines de Scikit-Learn.")
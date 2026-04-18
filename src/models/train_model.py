"""
Script de entrenamiento para el modelo final y evaluación básica.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# Importamos nuestra máquina de limpieza desde el módulo de features
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.build_features import crear_pipeline_maestro

def train_best_model(train_data_path: str, model_save_path: str):
    """
    Entrena el Random Forest usando el Pipeline y guarda el modelo final.
    """
    print(f"Cargando datos de entrenamiento desde {train_data_path}...")
    train_set = pd.read_csv(train_data_path)
    
    X_train = train_set.drop("median_house_value", axis=1)
    y_train = train_set["median_house_value"].copy()

    print("Configurando el Pipeline de preprocesamiento...")
    pipeline_maestro = crear_pipeline_maestro(X_train)

    print("Entrenando el RandomForestRegressor... (Esto puede tomar un momento)")
    modelo_final = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Empaquetamos todo en un solo tubo
    pipeline_definitivo = Pipeline([
        ('preprocesamiento', pipeline_maestro),
        ('modelo', modelo_final)
    ])
    
    pipeline_definitivo.fit(X_train, y_train)

    # Guardar modelo
    path = Path(model_save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_definitivo, path)
    print(f"¡Modelo guardado exitosamente en: {model_save_path}!")

def evaluate_model(model_path: str, test_data_path: str):
    """
    Evalúa el modelo contra los datos de prueba.
    """
    print(f"Evaluando modelo en {test_data_path}...")
    test_set = pd.read_csv(test_data_path)
    
    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()
    
    # El truco para que encuentre la clase personalizada al cargar
    from src.features.build_features import CreadorAtributos
    setattr(sys.modules["__main__"], "CreadorAtributos", CreadorAtributos)
    
    modelo = joblib.load(model_path)
    predicciones = modelo.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))

    print(f"RMSE Final en el Set de Prueba: ${rmse:,.2f}")


if __name__ == "__main__":
    TRAIN_PATH = "data/interim/train_set.csv"
    TEST_PATH = "data/interim/test_set.csv"
    MODEL_OUTPUT_PATH = "models/modelo_california.joblib"
    
    train_best_model(TRAIN_PATH, MODEL_OUTPUT_PATH)
    evaluate_model(MODEL_OUTPUT_PATH, TEST_PATH)
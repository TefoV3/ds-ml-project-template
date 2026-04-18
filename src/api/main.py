"""
API Básica usando FastAPI para servir el modelo entrenado de California Housing.
"""

from contextlib import asynccontextmanager # <-- 1. Importamos la nueva herramienta
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# --- Definición de la clase personalizada (IDÉNTICA AL NOTEBOOK) ---
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CreadorAtributos(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Aseguramos que X sea un array de numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

# --- Definición del Esquema de Datos ---
class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

# Variable global para el modelo
model = None

# --- 2. EL NUEVO MÉTODO LIFESPAN ---
# Esto reemplaza al viejo @app.on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Lo que pasa al ENCENDER el servidor ---
    global model
    try:
        import sys
        setattr(sys.modules["__main__"], "CreadorAtributos", CreadorAtributos)
        model = joblib.load("models/modelo_california.joblib")
        print("✅ Modelo cargado exitosamente en la memoria.")
    except Exception as e:
        print(f"❌ Error crítico: No se pudo cargar el modelo. Detalle: {e}")
    
    yield # Aquí la API se queda "pausada" recibiendo peticiones
    
    # --- Lo que pasa al APAGAR el servidor ---
    model = None
    print("🧹 Modelo descargado de la memoria.")

# --- 3. INICIALIZAR LA APP CON EL LIFESPAN ---
app = FastAPI(
    title="API de Predicción de Precios de Vivienda (California)", 
    description="API para estimar el valor medio de viviendas en distritos de California.",
    version="1.0",
    lifespan=lifespan # <-- Conectamos la función aquí
)

@app.get("/")
def home():
    return {
        "mensaje": "Bienvenido a la API del Proyecto Final de Ciencia de Datos",
        "docs": "Ve a la ruta /docs para probar la API"
    }

@app.post("/predict")
def predict_price(features: HousingFeatures):
    if model is None:
        return {"error": "El modelo no está disponible en el servidor."}
    
    try:
        data_df = pd.DataFrame([features.model_dump()]) # model_dump() es la forma moderna de dict()
        prediction = model.predict(data_df)
        
        return {
            "predicted_price_usd": round(float(prediction[0]), 2),
            "currency": "USD",
            "status": "success"
        }
        
    except Exception as e:
        return {"error": f"Ocurrió un error durante la predicción: {str(e)}"}
"""
Script para descargar y extraer los datos originales del proyecto.
"""

import os
import urllib.request
import tarfile
from pathlib import Path

def fetch_housing_data(housing_url: str, housing_path: str):
    """
    INSTRUCCIONES:
    1. Asegúrate de que el directorio `housing_path` exista (usa os.makedirs o Path.mkdir).
    2. Usa urllib.request.urlretrieve para descargar el archivo .tgz desde `housing_url`.
    3. Usa tarfile.open para extraer el contenido en `housing_path`.
    
    URL de los datos: "https://github.com/ageron/data/raw/main/housing.tgz"
    Ruta de destino recomendada: "data/raw/"
    """
    # 1. Crear el directorio si no existe (usando Pathlib para código más limpio)
    path = Path(housing_path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Definir la ruta exacta donde caerá el archivo comprimido
    tgz_path = path / "housing.tgz"
    
    # 2. Descargar el archivo
    print(f"Descargando datos desde: {housing_url}")
    urllib.request.urlretrieve(housing_url, tgz_path)
    
    # 3. Extraer el archivo tar.gz
    print(f"Extrayendo datos en: {housing_path}")
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=path)
        
    print("¡Descarga y extracción completadas con éxito!")

if __name__ == "__main__":
    URL = "https://github.com/ageron/data/raw/main/housing.tgz"
    PATH = r"C:\Users\Tefo\Desktop\Productivo\Clases\ds-ml-project-template\data\raw"
    fetch_housing_data(URL, PATH)
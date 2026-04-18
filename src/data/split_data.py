"""
Script para dividir los datos en conjunto de entrenamiento y conjunto de prueba.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

def split_and_save_data(raw_data_path: str, interim_data_path: str):
    """
    INSTRUCCIONES:
    1. Lee el archivo CSV descargado previamente en `raw_data_path` usando pandas.
    2. Separa los datos con `train_test_split()`. Te recomendamos un test_size=0.2 y random_state=42.
    3. (Opcional pero recomendado) Puedes usar `StratifiedShuffleSplit` basado en la variable
       del ingreso medio (median_income) para que la muestra sea representativa.
    4. Guarda los archivos resultantes (ej. train_set.csv y test_set.csv) en la carpeta `interim_data_path`.
    """
    print(f"Leyendo datos desde: {raw_data_path}")
    
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {raw_data_path}.")
        return

    # 1. Crear categorías de ingresos (bins) para estratificar
    # El ingreso en este dataset está escalado (ej. 1.5 = $15,000). 
    # Agrupamos en 5 categorías principales para evitar sesgos.
    df["income_cat"] = pd.cut(df["median_income"],
                              bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
                              labels=[1, 2, 3, 4, 5])

    # 2. Aplicar StratifiedShuffleSplit
    print("Realizando partición estratificada (Test size: 20%)...")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for train_index, test_index in split.split(df, df["income_cat"]):
        strat_train_set = df.loc[train_index].copy()
        strat_test_set = df.loc[test_index].copy()

    # 3. Eliminar la columna temporal 'income_cat' para dejar los datos originales
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # 4. Guardar los archivos en data/interim/
    interim_dir = Path(interim_data_path)
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = interim_dir / "train_set.csv"
    test_file = interim_dir / "test_set.csv"
    
    strat_train_set.to_csv(train_file, index=False)
    strat_test_set.to_csv(test_file, index=False)
    
    print("¡Partición completada con éxito!")
    print(f"- Entrenamiento guardado en: {train_file} ({len(strat_train_set)} filas)")
    print(f"- Prueba guardado en: {test_file} ({len(strat_test_set)} filas)")

if __name__ == "__main__":
    # Asegúrate de que la ruta coincida con donde se extrajo tu CSV en el paso anterior.
    # Si el script anterior lo dejó suelto en raw, usamos "data/raw/housing.csv"
    RAW_PATH = r"C:\Users\estef\OneDrive\Escritorio\Productivo\Clases\Maestria\Maestria-CienciaDeDatos\ds-ml-project-template\data\raw\housing.csv"
    INTERIM_PATH = r"C:\Users\estef\OneDrive\Escritorio\Productivo\Clases\Maestria\Maestria-CienciaDeDatos\ds-ml-project-template\data\interim"
    
    split_and_save_data(RAW_PATH, INTERIM_PATH)
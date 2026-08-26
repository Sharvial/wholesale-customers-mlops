import pandas as pd
import os

# Ruta al dataset original
DATA_PATH = "data/raw/wholesale_customers.csv"

def test_raw_data_exists():
    """Prueba 1: Verifica que el archivo de datos haya sido ingestado y exista."""
    assert os.path.exists(DATA_PATH), f"El archivo no se encontró en la ruta: {DATA_PATH}"

def test_data_columns():
    """Prueba 2: Verifica que el archivo contenga exactamente las columnas requeridas para el EDA."""
    df = pd.read_csv(DATA_PATH)
    expected_cols = ['Channel', 'Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    
    # Comprobamos que todas las columnas esperadas estén en el dataframe
    for col in expected_cols:
        assert col in df.columns, f"Falta la columna crítica: {col}"

def test_no_negative_values():
    """Prueba 3: Verifica que no haya datos imposibles (gastos negativos)."""
    df = pd.read_csv(DATA_PATH)
    spending_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    
    # Verificamos que todos los valores sean mayores o iguales a 0
    assert (df[spending_cols] >= 0).all().all(), "Error de calidad: Se detectaron gastos con valores negativos."
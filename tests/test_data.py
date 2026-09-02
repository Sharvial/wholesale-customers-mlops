import pandas as pd
import os


# Ruta al dataset original
DATA_PATH = "data/raw/wholesale_customers.csv"

SPENDING_COLS = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

EXPECTED_COLS = [
    "Channel",
    "Region",
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]


def test_raw_data_exists():
    """Verifica que el archivo de datos haya sido ingestado y exista."""
    assert os.path.exists(DATA_PATH), (
        f"El archivo no se encontró en la ruta: {DATA_PATH}"
    )


def test_data_columns():
    """Verifica que estén todas las columnas requeridas."""
    df = pd.read_csv(DATA_PATH)

    for col in EXPECTED_COLS:
        assert col in df.columns, f"Falta la columna crítica: {col}"


def test_no_negative_values():
    """Verifica que no haya gastos negativos."""
    df = pd.read_csv(DATA_PATH)

    assert (df[SPENDING_COLS] >= 0).all().all(), (
        "Error de calidad: Se detectaron gastos con valores negativos."
    )


def test_no_missing_values():
    """Verifica que no existan valores faltantes."""
    df = pd.read_csv(DATA_PATH)

    assert not df[EXPECTED_COLS].isnull().any().any(), (
        "Error de calidad: Se detectaron valores faltantes."
    )


def test_spending_columns_are_numeric():
    """Verifica que las variables de gasto sean numéricas."""
    df = pd.read_csv(DATA_PATH)

    for col in SPENDING_COLS:
        assert pd.api.types.is_numeric_dtype(df[col]), (
            f"Error de tipo: {col} debe ser numérica."
        )
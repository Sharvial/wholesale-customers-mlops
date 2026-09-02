import pandas as pd
import sys

def run_quality_gates(data_path):
    print(f"Iniciando Data Quality Gates para: {data_path}\n")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo en {data_path}")
        sys.exit(1)

    try:
        # Gate 1: Dataset con tamaño mínimo
        assert df.shape[0] > 100, "Falló Gate 1: El dataset tiene muy pocos registros (<100)."
        print("✅ Gate 1 superado: Cantidad de registros aceptable.")

        # Gate 2: Cero valores nulos
        assert df.isna().sum().sum() == 0, "Falló Gate 2: Se encontraron valores nulos (NaN)."
        print("✅ Gate 2 superado: No hay valores faltantes.")

        # Gate 3: Control de duplicados (menor al 5%)
        duplicated_ratio = df.duplicated().mean()
        assert duplicated_ratio < 0.05, f"Falló Gate 3: Demasiados duplicados ({duplicated_ratio:.1%})."
        print(f"✅ Gate 3 superado: Duplicados dentro del límite (< 5%). Actual: {duplicated_ratio:.1%}")

        # Gate 4: Valores de gasto estrictamente positivos o cero
        # Columnas correspondientes a categorías de gasto
        spending_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
        for col in spending_cols:
            assert (df[col] >= 0).all(), f"Falló Gate 4: Hay valores negativos en la columna {col}."
        print("✅ Gate 4 superado: Todos los valores de gasto son lógicos (>= 0).")

        # Gate 5: Tipos de datos correctos (todo debe ser numérico)
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Falló Gate 5: La columna {col} no es numérica."
        print("✅ Gate 5 superado: Todos los tipos de datos son numéricos.")

        print("\n🎉 ¡Todos los Data Quality Gates pasaron exitosamente! Los datos están listos para el EDA.")

    except AssertionError as msg:
        print(f"\n🚨 ALERTA DE CALIDAD DE DATOS: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    # Apuntamos a los datos crudos que acabas de descargar
    RAW_DATA_PATH = "data/raw/wholesale_customers.csv"
    run_quality_gates(RAW_DATA_PATH)
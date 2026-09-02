"""
Simulación controlada de contaminación de datos.

La simulación crea una copia temporal de los datos y agrega
diferentes tipos de contaminación sin modificar el dataset original.

Contaminaciones simuladas:
1. Valores faltantes
2. Filas duplicadas
3. Outlier extremo
4. Tipo de dato incorrecto
5. Categoría desconocida
6. Modificación del esquema
"""

from pathlib import Path
import pandas as pd
import numpy as np


# Rutas
RAW_PATH = Path("data/raw/wholesale_customers.csv")
PROCESSED_PATH = Path("data/processed/wholesale_customers_current.csv")


def load_data():
    """
    Carga los datos originales sin modificarlos.
    Se utiliza preferentemente el dataset raw porque contiene
    las variables categóricas Channel y Region.
    """

    if RAW_PATH.exists():
        return pd.read_csv(RAW_PATH)

    if PROCESSED_PATH.exists():
        return pd.read_csv(PROCESSED_PATH)

    raise FileNotFoundError(
        "No se encontró el dataset. Ejecuta primero la ingesta."
    )


def simulate_contamination(df):
    """
    Crea una copia contaminada del dataset.
    El DataFrame original nunca se modifica.
    """

    contaminated = df.copy()

    # ==========================================================
    # 1. VALOR FALTANTE
    # ==========================================================
    contaminated.loc[0, "Fresh"] = np.nan

    # ==========================================================
    # 2. FILA DUPLICADA
    # ==========================================================
    duplicated_row = contaminated.iloc[[0]].copy()
    contaminated = pd.concat(
        [contaminated, duplicated_row],
        ignore_index=True
    )

    # ==========================================================
    # 3. OUTLIER EXTREMO
    # ==========================================================
    contaminated.loc[1, "Grocery"] = (
        contaminated.loc[1, "Grocery"] * 100
    )

    # ==========================================================
    # 4. TIPO DE DATO INCORRECTO
    # ==========================================================
    contaminated["Milk"] = contaminated["Milk"].astype(object)
    contaminated.loc[2, "Milk"] = "ERROR"

    # ==========================================================
    # 5. CATEGORÍA DESCONOCIDA
    # ==========================================================
    if "Channel" in contaminated.columns:
        contaminated.loc[3, "Channel"] = 999

    if "Region" in contaminated.columns:
        contaminated.loc[4, "Region"] = 999

    # ==========================================================
    # 6. MODIFICACIÓN DEL ESQUEMA
    # ==========================================================
    contaminated["UnexpectedColumn"] = "SCHEMA_MODIFIED"

    return contaminated


def validate_contamination(original, contaminated):
    """
    Detecta las contaminaciones introducidas.
    """

    incidents = []

    # ----------------------------------------------------------
    # 1. Valores faltantes
    # ----------------------------------------------------------
    missing_values = contaminated.isna().sum()

    for column, count in missing_values.items():
        if count > 0:
            incidents.append(
                {
                    "type": "missing_values",
                    "column": column,
                    "message": f"Se detectaron {count} valores faltantes."
                }
            )

    # ----------------------------------------------------------
    # 2. Duplicados
    # ----------------------------------------------------------
    duplicate_count = contaminated.duplicated().sum()

    if duplicate_count > 0:
        incidents.append(
            {
                "type": "duplicates",
                "column": None,
                "message": (
                    f"Se detectaron {duplicate_count} filas duplicadas."
                )
            }
        )

    # ----------------------------------------------------------
    # 3. Outliers extremos
    # ----------------------------------------------------------
    numeric_columns = [
    column
    for column in original.select_dtypes(include=np.number).columns
    if column not in ["Channel", "Region"]
]

    for column in numeric_columns:
        if column not in contaminated.columns:
            continue

        original_median = original[column].median()

        if pd.isna(original_median) or original_median == 0:
            continue

        numeric_values = pd.to_numeric(
            contaminated[column],
            errors="coerce"
        )

        ratio = numeric_values / original_median

        extreme_values = (ratio > 50).sum()

        if extreme_values > 0:
            incidents.append(
                {
                    "type": "extreme_outlier",
                    "column": column,
                    "message": (
                        f"Se detectaron {extreme_values} "
                        "valores extremadamente altos."
                    )
                }
            )

    # ----------------------------------------------------------
    # 4. Tipo de dato incorrecto
    # ----------------------------------------------------------
    for column in original.select_dtypes(
        include=np.number
    ).columns:

        if column not in contaminated.columns:
            continue

        converted = pd.to_numeric(
            contaminated[column],
            errors="coerce"
        )

        invalid_types = (
            converted.isna()
            & contaminated[column].notna()
        ).sum()

        if invalid_types > 0:
            incidents.append(
                {
                    "type": "invalid_datatype",
                    "column": column,
                    "message": (
                        f"Se detectaron {invalid_types} "
                        "valores con tipo de dato incorrecto."
                    )
                }
            )

    # ----------------------------------------------------------
    # 5. Categorías desconocidas
    # ----------------------------------------------------------
    valid_categories = {
        "Channel": set(original["Channel"].dropna().unique())
        if "Channel" in original.columns
        else set(),

        "Region": set(original["Region"].dropna().unique())
        if "Region" in original.columns
        else set(),
    }

    for column, valid_values in valid_categories.items():

        if column not in contaminated.columns:
            continue

        unknown_values = set(
            contaminated[column].dropna().unique()
        ) - valid_values

        if unknown_values:
            incidents.append(
                {
                    "type": "unknown_category",
                    "column": column,
                    "message": (
                        f"Categorías desconocidas detectadas: "
                        f"{unknown_values}"
                    )
                }
            )

    # ----------------------------------------------------------
    # 6. Modificación del esquema
    # ----------------------------------------------------------
    original_columns = set(original.columns)
    contaminated_columns = set(contaminated.columns)

    added_columns = contaminated_columns - original_columns
    removed_columns = original_columns - contaminated_columns

    if added_columns:
        incidents.append(
            {
                "type": "schema_modification",
                "column": None,
                "message": (
                    f"Columnas nuevas detectadas: "
                    f"{added_columns}"
                )
            }
        )

    if removed_columns:
        incidents.append(
            {
                "type": "schema_modification",
                "column": None,
                "message": (
                    f"Columnas faltantes detectadas: "
                    f"{removed_columns}"
                )
            }
        )

    return incidents

def save_incidents(incidents):
    """
    Guarda los incidentes detectados en un archivo JSON
    para mantener trazabilidad de la contaminación.
    """

    import json
    from datetime import datetime

    log_path = Path("logs/contamination_incidents.json")

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "status": "BLOCKED",
        "incident_count": len(incidents),
        "incidents": incidents
    }

    with open(log_path, "w", encoding="utf-8") as file:
        json.dump(
            log_data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    print(f"\nIncidentes registrados en: {log_path}")

def main():

    print("=" * 60)
    print("SIMULACIÓN DE CONTAMINACIÓN DE DATOS")
    print("=" * 60)

    # Cargar datos originales
    original = load_data()

    print(f"\nDatos originales: {original.shape}")
    print("Dataset original cargado correctamente.")

    # Crear copia contaminada
    contaminated = simulate_contamination(original)

    print(f"Datos contaminados: {contaminated.shape}")

    # Validar
    incidents = validate_contamination(
        original,
        contaminated
    )

    print("\n" + "=" * 60)
    print("RESULTADOS DE LA VALIDACIÓN")
    print("=" * 60)

    if incidents:

        print(
            f"\nSe detectaron {len(incidents)} incidentes."
        )

        for i, incident in enumerate(incidents, start=1):

            print(
                f"\n{i}. [{incident['type']}]"
            )

            if incident["column"]:
                print(
                    f"   Columna: {incident['column']}"
                )

            print(
                f"   {incident['message']}"
            )

        print("\n" + "=" * 60)
        print("RESULTADO: BATCH BLOQUEADO")
        print("=" * 60)

        save_incidents(incidents)

        print("\nEl batch contaminado no debe continuar hacia el entrenamiento.")
        print("La simulación NO modifica el dataset original.")

    else:

        print("\nNo se detectaron incidentes.")

    print("\nLa simulación NO modifica el dataset original.")


if __name__ == "__main__":
    main()